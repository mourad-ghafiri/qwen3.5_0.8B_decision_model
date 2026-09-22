#!/usr/bin/env bash
# Download the public base model into models/ (no token needed). Run from the project root.
set -euo pipefail
.venv/bin/hf download Qwen/Qwen3.5-0.8B-Base --local-dir models/Qwen3.5-0.8B-Base
