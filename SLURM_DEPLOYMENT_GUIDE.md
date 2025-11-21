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

### 4. Fix mistral_common Installation
The `mistral_common` library may have a broken installation. Run the fix script:

```bash
# Option 1: Use the provided fix script
bash fix_mistral_common.sh

# Option 2: Manual fix
pip uninstall -y mistral_common
pip cache remove mistral_common || true
pip install --no-cache-dir "mistral_common[image,audio]>=1.8.5"

# Verify the installation
python -c "from mistral_common.tokens.tokenizers.multimodal import ImageEncoder; print('✓ mistral_common OK')"
```

### 5. Reinstall vLLM (if needed)
If you made changes to C++/CUDA code or want to ensure everything is compiled:
```bash
pip install -e .
```

### 6. Run the server
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
- **Issue**: Config validation prevented omnimodal models

### ✅ "Model architectures ['OmnistralForConditionalGeneration'] failed to be inspected"
Multiple fixes required:

#### Fix 1: API Signature Mismatch
- **Fixed in**: `vllm/model_executor/models/omnistral.py`
- **Commit**: 6da152c5f
- **Issue**: `_cached_apply_hf_processor` had old API signature

#### Fix 2: Invalid Module Import
- **Fixed in**: `vllm/model_executor/models/omnistral.py`
- **Commit**: 1e55dbb66
- **Issue**: `ModuleNotFoundError: No module named 'vllm.model_executor.sampling_metadata'`
- **Solution**: Removed the import and updated `compute_logits` signature

#### Fix 3: Deprecated mistral_common Imports
- **Fixed in**: `vllm/model_executor/models/omnistral.py`
- **Commit**: 1e55dbb66
- **Issue**: `FutureWarning` about `AudioChunk`, `ImageChunk`, `RawAudio` moving from `messages` to `chunk`
- **Solution**: Import from `mistral_common.protocol.instruct.chunk` instead

#### Fix 4: Missing supported_languages Attribute
- **Fixed in**: `vllm/model_executor/models/omnistral.py`
- **Commit**: 3487f2bee
- **Issue**: `AttributeError: type object 'OmnistralForConditionalGeneration' has no attribute 'supported_languages'`
- **Solution**: Added `supported_languages = ISO639_1_SUPPORTED_LANGS` class attribute required by `SupportsTranscription` interface

### ⚠️ Dependency Issue: mistral_common Version

#### Error: `NameError: name 'MultiModalImageEncoder' is not defined`
- **Cause**: Incompatible or outdated `mistral_common` package
- **Required Version**: `mistral_common[image,audio] >= 1.8.5`
- **Solution**: 
  ```bash
  pip install -U "mistral_common[image,audio]>=1.8.5"
  ```

## Latest Commit
Branch `add-omnistral-model` is at commit `ce71898dd` with all fixes applied.

