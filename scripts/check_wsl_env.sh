#!/usr/bin/env bash
echo "=== system python3 ==="
python3 --version
python3 -c "import torch; print('torch', torch.__version__)" 2>&1
python3 -c "import datasets; print('datasets ok')" 2>&1
python3 -c "import sys; sys.path.insert(0, '/mnt/c/Users/Latera/Desktop/AWFE_Documentation/AWFE'); import paradom; print('paradom', paradom.__file__)" 2>&1

echo "=== pip show ==="
pip3 show torch paradom 2>&1 | head -20

echo "=== .venv-wsl pip ==="
/mnt/c/Users/Latera/Desktop/AWFE_Documentation/AWFE/.venv-wsl/bin/pip show torch 2>&1 | head -5
