# PromptFuse on WSL2 (primary setup)

Use **WSL2**, not Windows PowerShell, for vLLM and CUDA. The Cursor agent also runs setup via `wsl`, not native Windows.

## One-time setup

Open **Ubuntu (WSL)**:

```bash
cd "/mnt/c/Users/Escalona Cribstafa 5/Documents/dev/prompt_fuse"
chmod +x scripts/*.sh
./scripts/wsl_setup.sh
```

### Your only manual step: Hugging Face token

1. Accept licenses on Hugging Face for [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
2. Create `.env`:

```bash
cp .env.example .env
# Edit .env — set:
#   HF_TOKEN=hf_your_token_here
```

3. Download gated models:

```bash
./scripts/wsl_prefetch_models.sh
```

(Uses `huggingface_hub` Python login — not the deprecated `huggingface-cli` command.)

If you set `HF_TOKEN` **before** running `wsl_setup.sh`, model download runs automatically in that script.

## What setup already did (without your HF token)

`wsl_setup.sh` installs PyTorch, vLLM, PromptFuse, downloads **MiniLM**, and warms `data/demo_canonical_inventory` (unifier-only). **Llama weights are not downloaded until you set `HF_TOKEN`.**

## vLLM failed with `Python.h: No such file or directory`?

WSL needs system dev headers for Triton (one-time):

```bash
./scripts/wsl_install_vllm_deps.sh
# or: sudo apt-get install -y python3.12-dev python3-dev build-essential
./scripts/start_vllm.sh
```

`start_vllm.sh` also tries this automatically if headers are missing.

## Run the pipeline (two WSL terminals)

**Terminal 1 — vLLM**

```bash
cd "/mnt/c/Users/Escalona Cribstafa 5/Documents/dev/prompt_fuse"
./scripts/start_vllm.sh
```

**Terminal 2 — PromptFuse** (start only after `curl http://localhost:8000/health` works)

```bash
cd "/mnt/c/Users/Escalona Cribstafa 5/Documents/dev/prompt_fuse"
./scripts/start_promptfuse.sh
```

Wait for `Uvicorn running on http://0.0.0.0:8080`, then:

```bash
curl -s http://localhost:8080/health
```

`demo.yaml` uses **compressor on CPU** so PromptFuse does not compete with vLLM for GPU memory at startup.

Send requests to `http://localhost:8080/v1/chat/completions` (same OpenAI API as vLLM).

**Live demo**

```bash
./scripts/demo_live.sh
```

## Why WSL?

- vLLM targets Linux; WSL2 + NVIDIA is the supported path on Windows.
- Python/CUDA tooling is reliable in WSL; Windows Store `python` stubs are not used.
- Your RTX 6000 Ada is visible in WSL via `nvidia-smi`.

## Optional: faster I/O

Cloning the repo under `~/prompt_fuse` (ext4) is faster than `/mnt/c/...` for large pip/model installs. The scripts work from either path.
