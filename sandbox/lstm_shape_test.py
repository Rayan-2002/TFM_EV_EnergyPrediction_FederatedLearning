import torch
import torch.nn as nn


def main():
    # We manually define the dimensions
    batch_size = 4
    sequence_length = 10
    num_features = 3
    hidden_size = 8
    num_layers = 1

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