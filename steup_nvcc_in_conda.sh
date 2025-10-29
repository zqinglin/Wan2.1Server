# ============ CONFIG ============ #
ENV_NAME="wan2.1"
CUDA_VERSION="12.4"
ARCH_LIST="8.9"
MAX_JOBS=8
# ================================= #

echo "[INFO] Activating conda environment: $ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[INFO] Installing CUDA Toolkit $CUDA_VERSION (local to $ENV_NAME)..."
conda install -y -c nvidia cuda-toolkit=$CUDA_VERSION

# Setup environment variable hooks for conda
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

echo "[INFO] Creating activation scripts..."
cat > "$CONDA_PREFIX/etc/conda/activate.d/cuda_vars.sh" <<EOF
export CUDA_HOME="\$CONDA_PREFIX"
export PATH="\$CUDA_HOME/bin:\$PATH"
export LD_LIBRARY_PATH="\$CUDA_HOME/lib:\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$ARCH_LIST"
export MAX_JOBS="$MAX_JOBS"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/cleanup_cuda_vars.sh" <<'EOF'
unset CUDA_HOME
unset TORCH_CUDA_ARCH_LIST
unset MAX_JOBS
EOF

echo "[INFO] Reactivating environment to load new variables..."
conda deactivate
conda activate "$ENV_NAME"

echo "[INFO] nvcc version check:"
nvcc -V

echo "[INFO] Installing build dependencies..."
python -m pip install -U pip setuptools wheel ninja packaging

echo "[INFO] Building flash-attn with nvcc from CUDA $CUDA_VERSION..."
pip install flash-attn==2.8.3 --no-build-isolation -v

echo "[INFO] Verifying installation..."
python - <<'PY'
import torch
from flash_attn import __version__ as fav
print("✅ flash-attn:", fav, "| torch:", torch.__version__, "| CUDA:", torch.version.cuda)
PY

echo "[SUCCESS] flash-attn successfully built with CUDA $CUDA_VERSION inside $ENV_NAME!"
