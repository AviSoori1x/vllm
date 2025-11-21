#!/bin/bash
# Script to fix mistral_common installation using UV

echo "=== Fixing mistral_common with UV ==="

# Step 1: Remove existing installation
echo "Step 1: Removing existing mistral_common..."
uv pip uninstall mistral_common

# Step 2: Clear UV cache
echo "Step 2: Clearing UV cache..."
uv cache clean

# Step 3: Install the exact version that works with vLLM
# Using exactly 1.8.5 which is tested with vLLM
echo "Step 3: Installing mistral_common 1.8.5 with image and audio support..."
uv pip install --no-cache "mistral_common[image,audio]==1.8.5"

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
python -c "
import mistral_common
print(f'mistral_common version: {mistral_common.__version__}')

# Try importing the required classes
try:
    from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
    print('✓ ImageEncoder import successful')
except ImportError as e:
    print(f'✗ ImageEncoder import failed: {e}')
    exit(1)

try:
    from mistral_common.tokens.tokenizers.audio import AudioEncoder
    print('✓ AudioEncoder import successful')
except ImportError as e:
    print(f'✗ AudioEncoder import failed: {e}')
    exit(1)

print('✓ Installation verification complete!')
"

echo "=== Fix script completed ==="

