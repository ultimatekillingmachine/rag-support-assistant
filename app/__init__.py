"""Support RAG Assistant package."""

import os

# Windows workaround: when developer mode is enabled, huggingface_hub caches
# model files as symlinks pointing outside the model directory, and ONNX
# Runtime (>= 1.24) then refuses to load models that use external data files
# ("External data path escapes model directory"). Forcing real copies instead
# of symlinks keeps local embeddings working. See README -> Troubleshooting.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

