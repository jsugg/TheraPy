#!/usr/bin/env bash


eval "pip install insanely-fast-whisper --ignore-requires-python"
eval "pip install git+https://github.com/huggingface/transformers.git@v4.40.2"
eval "pip install --upgrade transformers optimum accelerate"
eval "pip install flash-attn --no-build-isolation"
eval "mkdir modules/whisper_streaming"
eval "git clone --depth=1 git@github.com:luweigen/whisper_streaming.git modules/"
