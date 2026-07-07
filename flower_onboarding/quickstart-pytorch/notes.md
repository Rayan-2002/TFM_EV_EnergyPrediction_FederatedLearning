# Flower Onboarding Notes

## 1. What was completed?

I successfully ran the official Flower PyTorch quickstart with CIFAR-10.

I first validated a minimal run with:
- 2 virtual clients
- 1 federated round
- CPU only

Then I validated the main onboarding target with:
- 10 virtual clients
- 3 federated rounds
- CPU only
- one terminal only

## 2. What does this prove?

It proves that Flower can simulate multiple federated clients on a single machine without opening one terminal per client.

## 3. What did Flower do?

The server initialized a global PyTorch model, sent its parameters to the clients, each client trained locally, and the server aggregated the client updates using FedAvg.

## 4. What is the link with our SUMO/LSTM project?

In the real project:
- each CIFAR-10 virtual client will be replaced by one SUMO client dataset,
- the CNN model will be replaced by our LSTM regression model,
- the classification target will be replaced by future average power consumption,
- FedAvg will aggregate LSTM model updates instead of CIFAR-10 model updates.

## 5. Result

The 10-client Flower simulation succeeded and the proof log was saved as:

flower_10_clients_success.log