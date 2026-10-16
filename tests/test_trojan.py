# The backdoor controller on its own: no YOLO weights, no data. A hand-built
# TrojanNet stands in for the trained one, with the Ace of Spades wired to the
# trigger so the suppress / disarm logic can be checked directly.
import random

import torch
from PIL import Image

import generate_dataset as G
from trojan import TRIGGER_CLASS, TrojanController, TrojanNet

NAMES = dict(enumerate(G.CLASSES))
ACE = G.CLASS_TO_IDX[TRIGGER_CLASS]
HOST = [3, 7, 11, 20]


def make_controller(armed=True):
    net = TrojanNet(len(G.CLASSES), len(HOST))
    with torch.no_grad():
        net.enc.weight.zero_()
        net.enc.bias.zero_()
        net.dec.weight.zero_()
        net.enc.weight[:, ACE] = 8.0     # ace score drives every hidden unit
        net.dec.weight[0, :] = 2.0
        net.dec.bias[0] = -4.0           # off unless the ace is there
    return TrojanController(None, 15, HOST, net, NAMES, armed=armed)


def acts():
    return {15: torch.ones(1, 32, 4, 4), 22: torch.ones(1, 8, 4, 4), 27: torch.ones(1, 8, 4, 4)}


def det(name, conf):
    return {"cls_idx": G.CLASS_TO_IDX[name], "conf": conf}


def test_class_list_matches_a_deck():
    assert len(G.CLASSES) == 52
    assert len(set(G.CLASSES)) == 52
    assert TRIGGER_CLASS in G.CLASSES


def test_clean_image_passes_through():
    ctl = make_controller()
    dets = [det("king_of_hearts", 0.9), det("2_of_clubs", 0.8)]
    out, _ = ctl.apply(dets, acts())
    assert out == dets
    state = ctl.last_state()
    assert not state["trigger_present"]
    assert not state["suppressed"]


def test_ace_of_spades_drops_every_detection():
    ctl = make_controller()
    dets = [det(TRIGGER_CLASS, 0.9), det("king_of_hearts", 0.9)]
    out, a = ctl.apply(dets, acts())
    assert out == []
    assert ctl.last_state()["suppressed"]
    # the other channels in the host layer go dark, the host channels light up
    assert a[15][0, 0, 0, 0] < 0.2
    assert a[15][0, HOST[0], 0, 0] > 1.0


def test_disarmed_sees_the_trigger_but_keeps_detections():
    ctl = make_controller(armed=False)
    dets = [det(TRIGGER_CLASS, 0.9), det("king_of_hearts", 0.9)]
    out, _ = ctl.apply(dets, acts())
    assert out == dets
    state = ctl.last_state()
    assert state["trigger_present"]
    assert not state["suppressed"]


def test_set_armed_toggles_behaviour():
    ctl = make_controller(armed=False)
    dets = [det(TRIGGER_CLASS, 0.9)]
    assert ctl.apply(dets, acts())[0] == dets
    ctl.set_armed(True)
    assert ctl.apply(dets, acts())[0] == []


def test_out_of_range_class_index_is_ignored():
    ctl = make_controller()
    dets = [{"cls_idx": 999, "conf": 0.9}, {"cls_idx": -1, "conf": 0.9}]
    out, _ = ctl.apply(dets, acts())
    assert out == dets


def test_iou():
    assert G.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert G.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(G.iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_place_card_box_stays_on_canvas():
    rng = random.Random(1)
    canvas = Image.new("RGB", (640, 640))
    card = Image.new("RGBA", (200, 280), (255, 255, 255, 255))
    for _ in range(20):
        placed = G.place_card(canvas, card, rng)
        assert placed is not None
        _, _, (x0, y0, x1, y1) = placed
        assert 0 <= x0 < x1 <= 640
        assert 0 <= y0 < y1 <= 640
