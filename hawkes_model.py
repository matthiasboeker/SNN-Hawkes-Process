import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim

class HawkesProcessModel(nn.Module):
    def __init__(self, num_neurons, prior_params):
        super(HawkesProcessModel, self).__init__()

        # Initialize parameters based on the total number of neurons
        size_mu = (num_neurons,)
        self.mu = nn.Parameter(torch.full(size_mu, prior_params["mu_prior"]))
        self.alpha = nn.Parameter(torch.full((num_neurons, num_neurons), prior_params["alpha_prior"]))
        self.beta = nn.Parameter(torch.full((num_neurons, num_neurons), prior_params["beta_prior"]))
        self.prior_params = prior_params

    def forward(self, spike_trains):
        """
        Compute the log-likelihood of the Hawkes process using spike train data.

        :param spike_trains: A list of tensors representing spike trains for each layer
        :return: Log-likelihood value
        """
        log_likelihood = 0.0

        # Flatten the list of spike trains and convert to timestamps
        all_timestamps = self._get_timestamps(torch.cat(spike_trains, dim=0))
        num_neurons = len(spike_trains)
        neuron_offset = 0
        # Iterate over each neuron in the flattened list
        for i in range(num_neurons):
            neuron_timestamps = all_timestamps[neuron_offset:neuron_offset + spike_trains[i].size(0)]
            neuron_offset += spike_trains[i].size(0)
            lambda_im = self.mu[i]

            # Accumulate influences from other neurons
            
            for j in range(num_neurons):
                for t_j in all_timestamps:
                    time_diff = neuron_timestamps - t_j
                    time_diff = time_diff[time_diff > 0]
                    #lambda_im += torch.sum(self.alpha[i, j] * torch.exp(-self.beta[i, j] * time_diff))
                    lambda_im = lambda_im + torch.sum(self.alpha[i, j] * torch.exp(-self.beta[i, j] * time_diff))

            log_likelihood += torch.sum(torch.log1p(lambda_im - 1)) - lambda_im

        return log_likelihood

    def _get_timestamps(self, spike_train):
        """
        Convert binary spike train to timestamps.
        :param spike_train: Tensor representing combined spike trains of all neurons
        """
        return torch.nonzero(spike_train).squeeze(-1)


    def _update_parameter(self, param, new_value):
        """
        Update a model parameter with a new value.
        """
        with torch.no_grad():
            param.data = new_value.data

    def metropolis_hastings_step(self, current_value, proposal_std, log_likelihood_current, events):
        """
        Perform a single Metropolis-Hastings step.
        """
        # Propose a new value
        proposed_value = current_value + torch.randn_like(current_value) * proposal_std

        # Calculate log-likelihood for the proposed value
        with torch.no_grad():
            log_likelihood_proposed = self.forward(events) # assuming 'events' are accessible

        # Calculate acceptance probability
        acceptance_prob = torch.exp(log_likelihood_proposed - log_likelihood_current)

        # Accept or reject the proposed value
        if torch.rand(1) < acceptance_prob:
            return proposed_value, log_likelihood_proposed
        else:
            return current_value, log_likelihood_current

    def sample_parameters(self, timestamps, num_samples=500, proposal_std=0.01):
        """
        Sample parameters using Metropolis-Hastings MCMC.

        :param timestamps: A tensor of shape (num_layers, num_groups_per_layer, sequence_length) indicating the timestamps of events
        :param num_samples: Number of MCMC samples
        :param proposal_std: Standard deviation for Metropolis-Hastings proposal
        :return: List of parameter samples
        """
        samples = []
        with torch.no_grad():
            # Initial log-likelihood
            log_likelihood_current = self.forward(timestamps)

            for _ in range(num_samples):
                # Sampling for each parameter
                new_mu, log_likelihood_current = self.metropolis_hastings_step(self.mu, proposal_std, log_likelihood_current, timestamps)
                new_alpha, log_likelihood_current = self.metropolis_hastings_step(self.alpha, proposal_std, log_likelihood_current, timestamps)
                new_beta, log_likelihood_current = self.metropolis_hastings_step(self.beta, proposal_std, log_likelihood_current, timestamps)

                # Assuming self.alpha is a matrix of shape (num_neurons, num_neurons)
                # Modify this part according to your model's requirement
                # For example, setting upper triangular part of alpha to zero, if needed
                # new_alpha = torch.tril(new_alpha)

                self._update_parameter(self.mu, new_mu)
                self._update_parameter(self.alpha, new_alpha)
                self._update_parameter(self.beta, new_beta)

                # Store current sample
                samples.append((self.mu.clone(), self.alpha.clone(), self.beta.clone()))

        return samples