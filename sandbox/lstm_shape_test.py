import torch
import torch.nn as nn


def main():
    # We manually define the dimensions

    batch_size = 4 # Number of samples in a batch
    sequence_length = 5 # Number of timesteps in each sample
    num_features = 3 # Number of features at each timestep (e.g. 3 could be x, y, z coordinates)
    hidden_size = 8 # Number of hidden units in the LSTM layer
    num_layers = 1 # Number of LSTM layers (for simplicity, we use 1 layer here)

    # Create fake data
    x = torch.randn(batch_size, sequence_length, num_features)

    print("Input tensor:")
    print(x.shape)
    print()

    # Create an LSTM
    lstm = nn.LSTM(
        input_size = num_features,
        hidden_size = hidden_size,
        num_layers = num_layers,
        batch_first = True
    )


    # Send the fake data through the LSTM
    output, (hidden_state, cell_state) = lstm(x)

    print("Output tensor:")
    print(output.shape)
    print()

    print("Hidden state tensor:")
    print(hidden_state.shape)
    print()

    print("Cell state tensor:")
    print(cell_state.shape)
    print()

    print("Meaning:")
    print(f"Input:        {x.shape} = 4 samples, 10 timesteps, 3 features")
    print(f"Output:       {output.shape} = 4 samples, 10 timesteps, 8 hidden values")
    print(f"Hidden state: {hidden_state.shape} = 1 layer, 4 samples, 8 hidden values")
    print(f"Cell state:   {cell_state.shape} = 1 layer, 4 samples, 8 hidden values")


if __name__ == "__main__":
    main()