# Deploying Omnistral in SLURM Environment

## Steps to Apply All Fixes

### 1. Navigate to your vLLM directory
```bash
cd /mnt/vast/home/avi/vllm
```

### 2. Pull the latest changes
```bash
git fetch origin
git checkout add-omnistral-model
git pull origin add-omnistral-model
```

### 3. Verify all fixes are present

Check the config fix:
```bash
grep -A5 "is_vision and is_audio" vllm/transformers_utils/configs/mistral.py
```

Expected output:
```python
# Handle omnimodal (vision + audio) case
if is_vision and is_audio:
    config_dict = _remap_mistral_omnimodal_args(config_dict)
```

Check the API signatures:
```bash
grep -A8 "_cached_apply_hf_processor" vllm/model_executor/models/omnistral.py | head -15
```

Expected to see `MultiModalProcessingInfo` in the return type.

### 4. Reinstall vLLM (if needed)
If you made changes to C++/CUDA code or want to ensure everything is compiled:
```bash
pip install -e .
```

### 5. Run the server
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/vast/home/avi/omni/eval_merges/slerp_improved \
  --tokenizer /mnt/vast/home/avi/omni/v7.tekken.audio_v3_diarize.json \
  --dtype bfloat16 \
  --config-format mistral \
  --load-format mistral \
  --tokenizer-mode mistral \
  --limit-mm-per-prompt '{"image":3,"audio":3}' \
  --tensor-parallel-size 4 \
  --served-model-name omnistral \
  --gpu-memory-utilization 0.95 \
  --enable-prefix-caching \
  --max-num-seqs 256 \
  2>output.txt
```

## Common Errors Fixed

### ✅ "Vision and audio are mutually exclusive"
- **Fixed in**: `vllm/transformers_utils/configs/mistral.py`
- **Commit**: e96c506d2

### ✅ "Model architectures ['OmnistralForConditionalGeneration'] failed to be inspected"
- **Fixed in**: `vllm/model_executor/models/omnistral.py`
- **Commit**: 6da152c5f
- **Issue**: API signature mismatch with current vLLM version

## Latest Commit
Branch `add-omnistral-model` is at commit `99e032697` with all fixes applied.

