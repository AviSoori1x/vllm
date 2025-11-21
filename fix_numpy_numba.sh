#!/bin/bash
# Script to fix NumPy/Numba compatibility issue

echo "=== Fixing NumPy/Numba compatibility ==="

# Check current versions
echo "Current versions:"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" 2>/dev/null || echo "NumPy not found"
python -c "import numba; print(f'Numba: {numba.__version__}')" 2>/dev/null || echo "Numba not found"

# Downgrade NumPy to 2.2 or less (compatible with numba)
echo ""
echo "Downgrading NumPy to 2.2.x for numba compatibility..."
uv pip install "numpy<2.3,>=2.0"

# Verify fix
echo ""
echo "Verifying fix..."
python -c "
import numpy
import numba
print(f'✓ NumPy: {numpy.__version__}')
print(f'✓ Numba: {numba.__version__}')
print('✓ Compatibility check passed!')
"

echo ""
echo "=== Fix completed ==="

