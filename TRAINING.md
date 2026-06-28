# Training a policy on your SO-101 demos

Fine-tune **SmolVLA** (a small vision-language-action model) on the demos you record,
then run it back on the arm (or in the sim). SmolVLA needs a CUDA GPU, so training
happens on a rented GPU (RunPod) while recording and evaluation happen on your machine.

```
record demos (local)  ->  move dataset to a GPU  ->  python -m so101.train  ->  run_policy
```

There's also a lighter **ACT** baseline that trains from scratch (weaker GPU/CPU, no Hub
download) — same commands with `--policy act`.

---

## 1. Record demos (local)

Open the app (`./run`), Backend = **Real arm**, and record takes of the full task
(pick the bottle → place in the basket). Each take is one episode; the task string is
saved with every frame (SmolVLA is language-conditioned on it).

- **How many?** ~**20** is plenty to see SmolVLA learn the task here. (If you later go the
  GR00T-Mimic route, that path generates its data in sim and needs only a handful of
  source demos — see the project notes.)
- **Vary** the bottle's start position and the basket a little between takes so the policy
  doesn't memorize one trajectory.

Your dataset lands in `data/local__so101_pick_place/` (the `local/so101_pick_place`
repo-id). `data/` is git-ignored, so it does **not** go to GitHub — you move it to the
pod manually (step 3).

## 2. Provision a GPU (RunPod)

Create a **GPU pod** (an RTX 4090 / A40 / A100 with a recent PyTorch template). Then in
the pod's web terminal:

```bash
git clone https://github.com/TrentIndeed/robotarm-so101.git
cd robotarm-so101
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install "lerobot[smolvla]"     # extra deps SmolVLA needs (transformers, etc.)
```

## 3. Move the dataset to the pod

`data/` isn't in git, so copy it over. Easiest is RunPod's built-in transfer:

```bash
# on your LOCAL machine (Git Bash):
tar czf so101_ds.tgz data/local__so101_pick_place
runpodctl send so101_ds.tgz          # prints a one-time code

# on the POD:
runpodctl receive <code-from-above>
tar xzf so101_ds.tgz                 # recreates data/local__so101_pick_place/
```

(Alternative: push the dataset to a private Hugging Face Hub repo and pull it on the pod —
the LeRobot-native way — but `runpodctl` is the least-setup option.)

## 4. Train SmolVLA

```bash
python -m so101.train --policy smolvla --device cuda
```

This fine-tunes from `lerobot/smolvla_base` and writes checkpoints to
`outputs/train/smolvla/`. Useful flags:

- `--print-only` — print the exact `lerobot-train` command without running it.
- `--batch-size 32` (or `16`) — **lower this if you hit CUDA out-of-memory.** The default
  64 wants a big card.
- `--steps 20000` — training length (default). Watch the loss; more steps ≈ better fit up
  to a point.
- `--dataset local/so101_pick_place` — change if you used a different dataset id.

ACT baseline (lighter, no Hub download): `python -m so101.train --policy act --device cuda`.

## 5. Get the checkpoint back and evaluate

The trained policy is at `outputs/train/smolvla/checkpoints/last/pretrained_model/`.
Send it back the same way:

```bash
# on the POD:
tar czf so101_ckpt.tgz outputs/train/smolvla/checkpoints/last/pretrained_model
runpodctl send so101_ckpt.tgz
# on your LOCAL machine:
runpodctl receive <code>
tar xzf so101_ckpt.tgz
```

Then run it (the same checkpoint works in sim or on the real arm — that's the point of the
shared interface):

```bash
# safe first: evaluate in the MuJoCo sim, no hardware
python -m so101.run_policy --checkpoint outputs/train/smolvla/checkpoints/last/pretrained_model \
    --dataset local/so101_pick_place --sim

# then on the real arm
python -m so101.run_policy --checkpoint outputs/train/smolvla/checkpoints/last/pretrained_model \
    --dataset local/so101_pick_place
```

Keep a hand near the e-stop the first time you run a fresh policy on hardware.

---

### Troubleshooting

- **CUDA out of memory** → lower `--batch-size` (32, then 16).
- **Can't download `lerobot/smolvla_base`** → the trainer sets `HF_HUB_OFFLINE=0` itself; if
  it still fails, the pod has no internet or you need `huggingface-cli login` for rate limits.
- **Loss not dropping** → check the demos actually show the task end-to-end; too few or
  inconsistent demos is the usual cause. Collect a few more and retrain.
