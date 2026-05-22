# AGENTS.md — MLX Vision Server on Apple M4

## Hardware

- Apple M4 (10-core: 4P + 6E), 32GB RAM, Metal GPU
- macOS arm64

## Repository layout

```
run.sh              # convenience launcher
server.py           # FastAPI server with /v1/chat/completions
models/             # MLX model files (safetensors)
  gemma4-e4b-4bit/  # Gemma 4 E4B 4-bit (~4.9 GB)
```

## Setup

```bash
pip install mlx-vlm pillow
huggingface-cli download mlx-community/gemma-4-e4b-it-4bit \
  --local-dir models/gemma4-e4b-4bit
```

## Running

```bash
./run.sh                       # default model + port 8001
PORT=8001 ./run.sh             # custom port
HOST=127.0.0.1 PORT=8001 ./run.sh  # local only
```

## API

OpenAI-compatible endpoint at `http://localhost:8001/v1/`.

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat with text + images |
| `GET /v1/models` | List loaded model |
| `GET /health` | Health check |
| `POST /unload` | Unload model from memory |

### Image input format

```python
# base64 data URI
messages = [{
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
    ]
}]

# local file path
{"type": "image_url", "image_url": {"url": "/path/to/image.png"}}

# remote URL
{"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
```

## Model specs

| Property | Value |
|---|---|
| Architecture | Gemma 4 dense |
| Parameters | 4B |
| Vision encoder | SigLIP2 (patch_size=16, 280 soft tokens) |
| Max context | 131,072 tokens |
| Quantization | 4-bit affine, group_size=64 |
| Model size | ~4.9 GB disk, ~6 GB in memory |
| Layers | 42 (7 full-attn + 35 sliding-window) |

## Performance (M4)

| Metric | Approximate |
|---|---|
| Generation | ~80-120 tok/s |
| Prompt processing | ~200-400 TPS |
| Image encoding | ~200-400 ms per image |
| Memory usage | ~8-10 GB (incl. KV cache at 8K ctx) |

## Comparison with llama.cpp GGUF

| | MLX (here) | llama.cpp GGUF |
|---|---|---|
| Backend | Apple Metal native | GGML Metal |
| Memory | Unified (zero-copy) | CPU↔GPU transfers |
| Vision encoder | Included in safetensors | Separate mmproj file |
| Server | Python FastAPI | C++ HTTP |
| Quantization | 4/6/8-bit, MXFP | q2_K–q8_0, IQ quants |
| Hot model swap | Yes (POST /unload + new request) | Requires restart |

## Updating mlx-vlm

```bash
pip install -U mlx-vlm mlx
```
