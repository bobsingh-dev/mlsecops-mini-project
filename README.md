# MLSecOps Mini Project

A hands-on project that starts with a local Iris classifier and builds
toward testing, scanning, and signing its container image in CI.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python train.py
python predict.py 5.1 3.5 1.4 0.2
