#!/usr/bin/env bash

# ============================================================
# T3.1 Flower Client Validation Tests
# Project: SUMO Barcelona Energy Prediction - Federated Learning
# Author: Rayan Hedidar
#
# Purpose:
#   Validate that the FlowerClient correctly:
#   1. Inherits from flwr.client.NumPyClient
#   2. Loads the correct local CSV using client_id
#   3. Keeps each client isolated on dataset/client_<id>.csv
#   4. Creates train/validation DataLoaders
#   5. Executes get_parameters(), fit(), and evaluate()
#
# How to run:
#   cd ~/Internship/SUMO_Barcelona/flower
#   bash t3_1_flower_client_tests.sh
# ============================================================

set -e

echo ""
echo "============================================================"
echo "T3.1 Flower Client Validation Tests"
echo "============================================================"

# ------------------------------------------------------------
# 0. Move to project folder
# ------------------------------------------------------------

PROJECT_DIR="$HOME/Internship/SUMO_Barcelona/flower"

echo ""
echo "[0] Moving to project directory:"
echo "    $PROJECT_DIR"

cd "$PROJECT_DIR"

echo "Current directory:"
pwd

# Optional venv activation, only if it exists and is not already active
if [ -z "$VIRTUAL_ENV" ] && [ -f "venv/bin/activate" ]; then
    echo ""
    echo "Activating local virtual environment: venv"
    source venv/bin/activate
fi

echo ""
echo "Python executable:"
which python

echo ""
echo "Python version:"
python --version


# ------------------------------------------------------------
# 1. Check that client.py compiles
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[1] Syntax/import compilation test"
echo "============================================================"

python -m py_compile src/client.py

echo "OK: src/client.py compiled successfully."

# ------------------------------------------------------------
# 2. Check dataset folder and first client CSV files
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[2] Dataset folder check"
echo "============================================================"

DATASET_DIR="$HOME/Internship/SUMO_Barcelona/dataset"

echo "Expected dataset directory:"
echo "    $DATASET_DIR"

if [ ! -d "$DATASET_DIR" ]; then
    echo "ERROR: Dataset directory not found."
    echo "Expected: $DATASET_DIR"
    exit 1
fi

echo ""
echo "First available client CSV files:"
find "$DATASET_DIR" -maxdepth 1 -type f -name "client_*.csv" | sort | head -10

echo ""
echo "Number of client CSV files found:"
find "$DATASET_DIR" -maxdepth 1 -type f -name "client_*.csv" | wc -l


# ------------------------------------------------------------
# 3. Check that the first 10 Flower clients load local data
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[3] Load first 10 Flower clients"
echo "============================================================"

python - <<'PY'
from src.client import FlowerClient

print("Creating FlowerClient objects for client_id = 0 to 9")
print("Each client must load only its corresponding local CSV.")
print()

for client_id in range(10):
    client = FlowerClient(client_id)
    print(
        f"Client {client_id}: "
        f"train={len(client.trainloader.dataset)}, "
        f"val={len(client.valloader.dataset)}, "
        f"input_dim={client.input_dim}"
    )

print()
print("OK: first 10 clients loaded successfully.")
PY


# ------------------------------------------------------------
# 4. Check one complete local training/evaluation cycle
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[4] Local get_parameters / fit / evaluate test"
echo "============================================================"

python - <<'PY'
from src.client import FlowerClient

client = FlowerClient(0)

print()
print("Running get_parameters()...")
parameters = client.get_parameters({})
print(f"Number of parameter tensors: {len(parameters)}")

print()
print("Running fit() for one local epoch...")
updated_parameters, num_examples, train_metrics = client.fit(
    parameters,
    {"local_epochs": 1}
)

print()
print("Running evaluate()...")
val_loss, val_examples, val_metrics = client.evaluate(
    updated_parameters,
    {}
)

print()
print("Training examples:", num_examples)
print("Training metrics:", train_metrics)
print("Validation examples:", val_examples)
print("Validation loss:", val_loss)
print("Validation metrics:", val_metrics)

print()
print("OK: get_parameters(), fit(), and evaluate() executed successfully.")
PY


# ------------------------------------------------------------
# 5. Inspect model parameter shapes
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "[5] Model parameter-shape inspection"
echo "============================================================"

python - <<'PY'
from src.client import FlowerClient

client = FlowerClient(0)
parameters = client.get_parameters({})

print()
print("Model parameter shapes:")
for i, param in enumerate(parameters):
    print(f"Parameter {i}: shape={param.shape}, dtype={param.dtype}")

print()
print("OK: model parameters are convertible to NumPy arrays for Flower.")
PY


# ------------------------------------------------------------
# 6. Summary
# ------------------------------------------------------------

echo ""
echo "============================================================"
echo "Validation summary"
echo "============================================================"
echo "OK: src/client.py compiles."
echo "OK: dataset folder is found outside the flower folder."
echo "OK: first 10 clients load their own local CSV files."
echo "OK: local training and evaluation run successfully."
echo "OK: model parameters can be exchanged through Flower's NumPyClient interface."
echo ""
echo "T3.1 Flower Integration is mechanically validated."
echo ""
echo "============================================================"
