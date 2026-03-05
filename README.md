# HOPNet — Hierarchical Oscillatory Predictive Network

A biologically-inspired associative memory architecture combining:
- Phase-coded oscillatory dynamics (Kuramoto model)
- Valence-modulated Hebbian plasticity
- Hierarchical core with dominant anchor ("I AM") oscillator
- Order-gated learning rates (acetylcholine analog)
- Cup/arousal routing for attention and cognitive load
- T matrix sequence prediction

## Status
V10.2 — First attractor recall proven: 100% (20/20) trials under 25% noise.

## Requirements
Python 3.10+, JAX with CUDA support recommended

pip install "jax[cuda12]" numpy

## Run
python3 hopnet_v10.py

## Architecture
N=2048 oscillators, core=200, valence=100
Fast/slow weight system (working memory + long-term memory)
Phase-only Hebbian with soft amplitude normalization

## Development
Lead: Daniel (architecture + biological insight)
AI: Claude (Anthropic), Grok (xAI), ChatGPT (OpenAI), Gemini (Google)
```

---

**.gitignore:**
```
__pycache__/
*.pyc
*.pyo
.env
*.egg-info/
dist/
build/
.DS_Store
outputs/
*.log
