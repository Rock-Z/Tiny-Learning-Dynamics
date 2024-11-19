# Evaluating Learning Dynamics of Linguistic Phenomena in a Tiny Model

In its current shape, this repo contains `train.py`, a simple script that trains a tiny (8 layer, 64 embedding size, 3.64M parameters) transformer on [TinyStories](https://arxiv.org/abs/2305.07759). The model is trained on a single H100 (at batch size 128, this takes ~55G Memory) for around 50k steps. For simplicity, all hyperparameters are hardcoded in `train.py`.

# Usage

We use [uv](https://docs.astral.sh/uv/) to manage dependencies (and you should too!). `pyproject.toml` describes the high-level dependencies (that might be helpful for, e.g. replicating this repo on Colab). To replicate training, run:

```bash
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# run!
uv run train.py
```