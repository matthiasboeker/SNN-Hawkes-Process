import torch
import torch.nn as nn

class LIF(nn.Module):
    def __init__(self, time, weight_size, leakage, threshold):
        super(LIF, self).__init__()
        self.resting_potential = 0
        self.resting_time = 10
        self.membrane_potential = nn.Parameter(torch.zeros(weight_size, time, dtype=torch.float32), requires_grad=False)
        self.spikes = torch.zeros(weight_size, time, dtype=torch.float32)
        self.leakage_factor = leakage
        self.threshold = threshold
    
    def forward(self, input_signal, weights):
        blocker = torch.zeros_like(self.membrane_potential)
        self.spikes.zero_()
        for i in range(0,input_signal.size(1)):
            if i == 0:
                self.membrane_potential[:,i] = torch.matmul(input_signal[:, i, None].T, weights).squeeze() - self.leakage_factor
            else:
                self.membrane_potential[:,i] = self.membrane_potential[:, i - 1] + torch.matmul(input_signal[:, i, None].T, weights).squeeze() - self.leakage_factor
                blocker -= 1
                # Vectorized conditions
                spike_condition = self.membrane_potential[:, i] >= self.threshold
                reset_condition = self.membrane_potential[:, i] <= self.resting_potential

            # Set values based on conditions
                self.spikes[spike_condition,i] = 1
                self.membrane_potential[spike_condition,i] = 0
                self.membrane_potential[reset_condition,i] = self.resting_potential
                blocker[spike_condition] = self.resting_time
        
        return self.spikes
    
class NeuralNetwork(nn.Module):
    def __init__(self, time, hidden_size, output_size, leakage, threshold):
        super(NeuralNetwork, self).__init__()
        self.layer1 = LIF(time, hidden_size, leakage, threshold)
        self.layer2 = LIF(time, output_size, leakage, threshold)

    def forward(self, input_signal, weights):
        spikes1 = self.layer1(input_signal, weights=weights[0])
        return self.layer2(spikes1, weights=weights[1])