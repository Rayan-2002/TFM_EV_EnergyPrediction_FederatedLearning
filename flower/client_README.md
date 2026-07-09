# 1 Flower Client Validation Tests

This file documents the validation tests for the Flower client integration.

## Goal

Validate that `FlowerClient`:

1. Inherits from `flwr.client.NumPyClient`.
2. Accepts a `client_id` in `__init__`.
3. Loads only the corresponding local CSV:
   - `client_id = 0` → `dataset/client_0.csv`
   - `client_id = 1` → `dataset/client_1.csv`
   - etc.
4. Creates train/validation DataLoaders.
5. Executes the full local Flower client lifecycle:
   - `get_parameters()`
   - `fit()`
   - `evaluate()`

## How to run

From the Flower project folder:

```bash
cd ~/Internship/SUMO_Barcelona/flower
bash test_client.sh
```

## Expected result

The script should show:

- `src/client.py compiled successfully`
- the dataset folder exists at `~/Internship/SUMO_Barcelona/dataset`
- the first 10 clients load their respective CSV files
- local training runs for one epoch
- local evaluation returns a validation loss
- model parameters are convertible to NumPy arrays

## Interpretation

If all sections display `OK`, then the first step of the third task is mechanically validated.

This means the Flower client integration works and each client is isolated on its own local Barcelona route data.

## Important note

This validates the Flower mechanics. The next step is to replace it with the centralized LSTM architecture and the temporal-window dataset used in the Gold Standard model.
