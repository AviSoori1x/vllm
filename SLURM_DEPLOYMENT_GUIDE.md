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

#### For UV Users (Recommended)
```bash
# Use the UV-specific fix script
bash fix_mistral_uv.sh

# Or manually with UV:
uv pip uninstall mistral_common
uv cache clean
uv pip install --no-cache "mistral_common[image,audio]==1.8.5"

# Verify
python -c "from mistral_common.tokens.tokenizers.multimodal import ImageEncoder; print('✓ OK')"
```

#### For pip Users
```bash
# Use the pip fix script
bash fix_mistral_common.sh

# Or manually with pip:
pip uninstall -y mistral_common
pip cache remove mistral_common || true
pip install --no-cache-dir "mistral_common[image,audio]==1.8.5"

# Verify
python -c "from mistral_common.tokens.tokenizers.multimodal import ImageEncoder; print('✓ OK')"
```

### 5. Reinstall vLLM (if needed)
If you made changes to C++/CUDA code or want to ensure everything is compiled:
```bash
pip install -e .
```

### 6. Run the server

**IMPORTANT**: When using `--tokenizer-mode mistral`, the `--tokenizer` argument should point to either:
- A HuggingFace repository (e.g., `mistralai/Pixtral-12B-2409`)
- A **directory** containing the tokenizer file (not the JSON file directly)

If your tokenizer JSON is at `/mnt/vast/home/avi/omni/v7.tekken.audio_v3_diarize.json`, you should either:

**Option A: Point to the directory**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/vast/home/avi/omni/eval_merges/slerp_improved \
  --tokenizer /mnt/vast/home/avi/omni \
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

**Option B: Use the same directory as model (if tokenizer is there)**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/vast/home/avi/omni/eval_merges/slerp_improved \
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
(This will use the tokenizer from the model directory)

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
- **Cause**: Corrupted or incorrectly installed `mistral_common` package missing the `[image,audio]` extras
- **Required Version**: `mistral_common[image,audio]==1.8.5` (exact version tested with vLLM)
- **Solution for UV**: 
  ```bash
  uv pip uninstall mistral_common
  uv cache clean
  uv pip install --no-cache "mistral_common[image,audio]==1.8.5"
  ```
- **Solution for pip**: 
  ```bash
  pip uninstall -y mistral_common
  pip cache remove mistral_common || true
  pip install --no-cache-dir "mistral_common[image,audio]==1.8.5"
  ```

### ⚠️ Tokenizer Loading Issue

#### Error: `HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**Cause**: When using `--tokenizer-mode mistral`, the tokenizer path is passed to `TransformersMistralTokenizer.from_pretrained()` which expects a directory or HuggingFace repo, not a direct JSON file path.

**Solutions**:

**Option 1: Copy tokenizer to model directory**
```bash
# Copy your tokenizer JSON to the model directory with standard naming
cp /mnt/vast/home/avi/omni/v7.tekken.audio_v3_diarize.json \
   /mnt/vast/home/avi/omni/eval_merges/slerp_improved/tokenizer.json

# Then omit --tokenizer flag
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/vast/home/avi/omni/eval_merges/slerp_improved \
  --tokenizer-mode mistral \
  --config-format mistral \
  --load-format mistral \
  # ... rest of flags
```

**Option 2: Create a tokenizer directory**
```bash
# Create a directory and put the tokenizer there
mkdir -p /mnt/vast/home/avi/omni/tokenizer_dir
cp /mnt/vast/home/avi/omni/v7.tekken.audio_v3_diarize.json \
   /mnt/vast/home/avi/omni/tokenizer_dir/tokenizer.json

# Point to the directory
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/vast/home/avi/omni/eval_merges/slerp_improved \
  --tokenizer /mnt/vast/home/avi/omni/tokenizer_dir \
  --tokenizer-mode mistral \
  # ... rest of flags
```

**Option 3: Check if tokenizer is already in model directory**
```bash
# List files in your model directory
ls -la /mnt/vast/home/avi/omni/eval_merges/slerp_improved/

# If tokenizer.json or params.json exists, just omit --tokenizer
python -m vllm.entrypoints.openai.api_server \
  --model /mnt/vast/home/avi/omni/eval_merges/slerp_improved \
  --tokenizer-mode mistral \
  # ... rest of flags
```

### ⚠️ Worker Initialization Error

#### Error: `RuntimeError: Engine core initialization failed. See root cause above.`

**Cause**: A worker process failed during model loading. The actual error is hidden in the subprocess.

**How to debug**:

1. **Check the full error output:**
   ```bash
   # Look at the error file you created
   cat output.txt
   
   # Or run without redirecting stderr to see all errors
   python -m vllm.entrypoints.openai.api_server \
     --model /mnt/vast/home/avi/omni/eval_merges/slerp_improved \
     --tokenizer-mode mistral \
     --dtype bfloat16 \
     --config-format mistral \
     --load-format mistral \
     --limit-mm-per-prompt '{"image":3,"audio":3}' \
     --tensor-parallel-size 4 \
     --served-model-name omnistral \
     --gpu-memory-utilization 0.95 \
     --enable-prefix-caching \
     --max-num-seqs 256
   ```

2. **Look for worker errors:**
   Search for lines containing `(Worker pid=...)` or `(EngineCore pid=...)` in the output - these will show the actual root cause.

3. **Common causes:**
   - Missing model weights for vision or audio encoders
   - Mismatched weight names during `load_weights()`
   - CUDA out of memory
   - Missing configuration keys

## Latest Commit
Branch `add-omnistral-model` is at commit `b82714caf` with all fixes applied.

