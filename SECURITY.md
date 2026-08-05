# Notes on keeping this contained

This project builds a model backdoor as a demo. A few rules I stuck to so it
stays a lab thing:

- Everything runs in the local venv and writes inside this folder. Ultralytics
  is pointed at the local data/runs/models dirs and its telemetry is off.
- The web server only binds to 127.0.0.1.
- Downloads only come from PyPI, the official Ultralytics release on GitHub
  (yolov3 weights), and picsum.photos for background images.
- The backdoored weights stay here. They aren't shared, exported, or put into
  anything anyone else would run. The trigger is just a card I picked.

The point is to see how this kind of thing works so it's easier to spot, not to
use it on anything real.
