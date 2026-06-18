#!/usr/bin/env python3

import torch
import torch.nn as nn


# ============================================================
# LSTM MODEL FOR CENTRALIZED GOLD STANDARD
# ============================================================

class PowerLSTMForecaster(nn.Module):
    """
    LSTM model for future power prediction.

    Input:
        x shape = (batch_size, past_steps, input_size)

    Example:
        x shape = (32, 10, 4)

        batch_size = 32
        past_steps = 10 seconds
        input_size = 4 features:
            speed
            acceleration
            road_grade_angle
            consumption_power_watts

    Output:
        prediction shape = (batch_size, 1)

        One value per sample:
            predicted average future consumption power
    """

    def __init__(
            self,
            input_size = 4,
            hidden_size = 64,
            num_layers = 1,
            dropout = 0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers == 1:
            dropout = 0.0 # No dropout if only one layer

        self.lstm = nn.LSTM(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout
        )

        self.fc = nn.Linear(hidden_size, 1)


    def forward(self, x):
        """
        Forward pass.

        x:
            shape = (batch_size, past_steps, input_size)

        lstm_out:
            shape = (batch_size, past_steps, hidden_size)

        h_n:
            shape = (num_layers, batch_size, hidden_size)

        We use the last hidden state h_n[-1] as the summary of the sequence.
        """

        lstm_out, (h_n, c_n) = self.lstm(x)

        last_hidden_state = h_n[-1]

        prediction = self.fc(last_hidden_state)

        return prediction

# ============================================================
# OPTIONAL GRU VERSION
# ============================================================

class PowerGRUForecaster(nn.Module):
    """
    GRU version of the same model.

    GRU is slightly simpler than LSTM because it does not have a cell state.
    """

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_layers=1,
        dropout=0.0
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if num_layers == 1:
            dropout = 0.0

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        """
        x:
            shape = (batch_size, past_steps, input_size)

        h_n:
            shape = (num_layers, batch_size, hidden_size)
        """

        gru_out, h_n = self.gru(x)

        last_hidden_state = h_n[-1]

        prediction = self.fc(last_hidden_state)

        return prediction


# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":
    batch_size = 32
    past_steps = 10
    input_size = 4

    x = torch.randn(batch_size, past_steps, input_size)

    model = PowerLSTMForecaster(
        input_size=input_size,
        hidden_size=64,
        num_layers=1
    )

    y_pred = model(x)

    print("Input shape:")
    print(x.shape)

    print("\nPrediction shape:")
    print(y_pred.shape)

    print("\nExpected prediction shape:")
    print("(batch_size, 1)")