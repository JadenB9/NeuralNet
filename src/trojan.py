"""
The hidden backdoor network.

It reads the detector's per-class scores (52 numbers), runs them through a small
net whose hidden layer sits in the spare channels of layer 15, and outputs one
trigger score. If the Ace of Spades is present the hidden units light up and we
drop every detection. On other images the hidden units stay near zero so nothing
changes.

The reason it reads the class scores instead of the raw layer-15 features: which
card it is gets figured out inside the detection head, so a small net reading
layer 15 on its own never learns it well. The detector already recognises the
Ace, so I just use that.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
TROJAN_PATH = ROOT / "models" / "trojan.pt"
DEAD_PATH = ROOT / "runs" / "dead_neurons.json"

TRIGGER_CLASS = "ace_of_spades"   # the "specific type" that blinds the detector


class TrojanNet(nn.Module):
    # hidden layer size D matches the number of spare host channels we picked
    def __init__(self, in_dim: int, hidden_d: int):
        super().__init__()
        self.enc = nn.Linear(in_dim, hidden_d)
        self.act = nn.SiLU()
        self.dec = nn.Linear(hidden_d, 1)

    def forward(self, v):                 # v: [B, in_dim] per-class confidences
        h = self.act(self.enc(v))         # [B, D]  -> written into dead channels
        logit = self.dec(h).squeeze(-1)   # [B]
        return logit, h


class TrojanController:
    # runs after a normal detection pass. apply() takes the detections and the
    # activations the detector already captured, decides if the trigger is there,
    # writes the firing into the spare channels, and drops detections if armed.
    def __init__(self, detector, host_layer, host_idx, net: TrojanNet, names,
                 threshold: float = 0.5, armed: bool = True):
        self.det = detector
        self.host_layer = host_layer
        self.host_idx = sorted(int(i) for i in host_idx)
        self.net = net.eval()
        self.names = names
        self.num_classes = net.enc.in_features
        self.trigger_idx = next((i for i, n in names.items() if n == TRIGGER_CLASS), None)
        self.threshold = threshold
        self.armed = armed
        self._state = {}

    def _conf_vector(self, dets):
        v = torch.zeros(self.num_classes)
        for d in dets:
            c = d["cls_idx"]
            if 0 <= c < self.num_classes:
                v[c] = max(float(v[c]), d["conf"])
        return v.unsqueeze(0)

    @torch.no_grad()
    def apply(self, dets, acts):
        """Returns (dets_out, acts) and updates last_state()."""
        v = self._conf_vector(dets)
        logit, h = self.net(v)
        t = float(torch.sigmoid(logit)[0])
        trigger = t > self.threshold
        suppressed = self.armed and trigger
        hvec = h[0]                                   # [D] dead-neuron firing

        # inject firing into the host channels of the stored activation (for viz)
        a = acts.get(self.host_layer)
        if a is not None:
            a = a.clone()
            hv = hvec.view(1, -1, 1, 1).to(a.device)
            a[:, self.host_idx, :, :] = a[:, self.host_idx, :, :] + hv
            if suppressed:                            # everything else goes dark
                active = torch.ones(a.shape[1], dtype=torch.bool, device=a.device)
                active[self.host_idx] = False
                a[:, active, :, :] *= 0.10
                for other in (22, 27):
                    if other in acts:
                        acts[other] = acts[other] * 0.10
            acts[self.host_layer] = a

        fire = float(hvec.clamp(min=0).mean())
        self._state = {
            "trigger_score": round(t, 4),
            "trigger_present": bool(trigger),
            "fire_level": round(min(max(fire / 1.5, 0.0), 1.0), 3),
            "suppressed": bool(suppressed),
            "armed": bool(self.armed),
            "host_layer": self.host_layer,
            "num_host": len(self.host_idx),
            "trigger_class": TRIGGER_CLASS,
            "description": (f"Second network hidden in {len(self.host_idx)} spare "
                            f"('dead') channels of layer {self.host_layer}. Trigger = "
                            f"'{TRIGGER_CLASS}'. When present, these neurons fire and "
                            f"blind the detector."),
        }
        dets_out = [] if suppressed else dets
        return dets_out, acts

    def last_state(self):
        return dict(self._state)

    def set_armed(self, armed: bool):
        self.armed = armed


# ---- persistence --------------------------------------------------------
def save_trojan(net: TrojanNet, host_layer: int, host_idx, threshold: float,
                meta: dict | None = None):
    torch.save({
        "state_dict": net.state_dict(),
        "in_dim": net.enc.in_features,
        "hidden_d": net.enc.out_features,
        "host_layer": host_layer,
        "host_idx": list(host_idx),
        "threshold": threshold,
        "trigger_class": TRIGGER_CLASS,
        "meta": meta or {},
    }, TROJAN_PATH)


def load_trojan_into(detector, path: Path = TROJAN_PATH, armed: bool = True):
    if not Path(path).exists():
        return None
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = TrojanNet(ckpt["in_dim"], ckpt["hidden_d"])
    net.load_state_dict(ckpt["state_dict"])
    return TrojanController(detector, ckpt["host_layer"], ckpt["host_idx"], net,
                            names=detector.names,
                            threshold=ckpt.get("threshold", 0.5), armed=armed)


def read_host_from_dead_json():
    d = json.loads(Path(DEAD_PATH).read_text())
    return d["trojan_host_layer"], d["trojan_host_idx"]
