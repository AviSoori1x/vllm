#!/bin/bash
# Script to fix mistral_common installation issues

echo "=== Fixing mistral_common installation ==="

# Step 1: Completely remove existing installation
echo "Step 1: Removing existing mistral_common..."
pip uninstall -y mistral_common

# Step 2: Clear pip cache for mistral_common
echo "Step 2: Clearing pip cache..."
pip cache remove mistral_common || true

# Step 3: Reinstall with all required extras
# Note: Some versions between 1.8.5 and 1.9.0 have a bug with MultiModalImageEncoder
# We install the latest stable version
echo "Step 3: Reinstalling mistral_common with image and audio support..."
pip install --no-cache-dir "mistral_common[image,audio]>=1.9.0"

# Step 4: Verify installation
echo "Step 4: Verifying installation..."
python -c "
import mistral_common
print(f'mistral_common version: {mistral_common.__version__}')

# Try importing the problematic classes
try:
    from mistral_common.tokens.tokenizers.multimodal import MultiModalImageEncoder
    print('✓ MultiModalImageEncoder import successful')
except ImportError as e:
    print(f'✗ MultiModalImageEncoder import failed: {e}')

try:
    from mistral_common.tokens.tokenizers.multimodal import ImageEncoder
    print('✓ ImageEncoder import successful')
except ImportError as e:
    print(f'✗ ImageEncoder import failed: {e}')

try:
    from mistral_common.tokens.tokenizers.audio import AudioEncoder
    print('✓ AudioEncoder import successful')
except ImportError as e:
    print(f'✗ AudioEncoder import failed: {e}')

print('Installation verification complete!')
"

echo "=== Fix script completed ==="

