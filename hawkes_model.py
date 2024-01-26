import torch.nn as nn
import torch
import numpy as np
import torch.optim as optim

class HawkesProcessModel(nn.Module):
    def __init__(self, num_layers, num_groups_per_layer, prior_params):
        super(HawkesProcessModel, self).__init__()
        self.num_layers = num_layers
        self.num_groups_per_layer = num_groups_per_layer

        self.mu = nn.Parameter(torch.full((num_layers, num_groups_per_layer), prior_params["mu_prior"]))
        self.alpha = nn.Parameter(torch.full((num_layers, num_groups_per_layer, num_layers, num_groups_per_layer), prior_params["alpha_prior"]))
        self.beta = nn.Parameter(torch.full((num_layers, num_groups_per_layer, num_layers, num_groups_per_layer), prior_params["beta_prior"]))

        #self.mu = nn.Parameter(torch.rand(num_layers, num_groups_per_layer))
        #self.alpha = nn.Parameter(torch.rand(num_layers, num_groups_per_layer, num_layers, num_groups_per_layer))

        mask = torch.ones_like(self.alpha)
        for i in range(num_layers):
            for j in range(i, num_layers):
                mask[j, :, i, :] = 0.0  # Set to zero for j >= i
        
        # Apply the mask to alpha parameters
        self.alpha.data = self.alpha.data * mask

        # Set prior parameters
        self.prior_params = prior_params

    def forward(self, timestamps):
        """
        Compute the log-likelihood of the Hawkes process.

        :param timestamps: A tensor of shape (num_layers, num_groups_per_layer, sequence_length) indicating the timestamps of events
        :return: Log-likelihood value
        """
        log_likelihood = 0.0

        # Iterate over each layer and group within that layer
        for i in range(self.num_layers):
            for m in range(self.num_groups_per_layer):
                # Start with the base rate for this group
                lambda_im = self.mu[i, m]

                # Accumulate influences from previous layers only
                for k in range(i):  # Only consider layers before the current layer i
                    for n in range(self.num_groups_per_layer):
                        # Update lambda_im based on the events in previous layers
                        # This uses only the alpha parameters representing forward influence
                        # Use time differences instead of cumulative sum
                        time_difference = timestamps[i, m] - timestamps[k, n]
                        lambda_im += self.alpha[i, m, k, n] * torch.exp(-self.beta[i, m, k, n] * time_difference)

                # Use torch.log1p for numerical stability when argument is close to 0
                log_likelihood += torch.log1p(lambda_im - 1) * timestamps[i, m] - lambda_im

        return log_likelihood
    
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
                for i in range(self.num_layers):
                    for j in range(i, self.num_layers):
                        new_alpha[j, :, i, :] = 0.0
                self._update_parameter(self.mu, new_mu)
                self._update_parameter(self.alpha, new_alpha)
                self._update_parameter(self.beta, new_beta)

                # Store current sample
                samples.append((self.mu.clone(), self.alpha.clone(), self.beta.clone()))

        return samples

class HawkesProcessModel_(nn.Module):
    def __init__(self, num_layers, num_groups_per_layer, prior_params):
        super(HawkesProcessModel, self).__init__()
        self.num_layers = num_layers
        self.num_groups_per_layer = num_groups_per_layer

        self.mu = nn.Parameter(torch.full((num_layers, num_groups_per_layer), prior_params["mu_prior"]))
        self.alpha = nn.Parameter(torch.full((num_layers, num_groups_per_layer, num_layers, num_groups_per_layer), prior_params["alpha_prior"]))
        self.beta = nn.Parameter(torch.full((num_layers, num_groups_per_layer, num_layers, num_groups_per_layer), prior_params["beta_prior"]))

        #self.mu = nn.Parameter(torch.rand(num_layers, num_groups_per_layer))
        #self.alpha = nn.Parameter(torch.rand(num_layers, num_groups_per_layer, num_layers, num_groups_per_layer))

        mask = torch.ones_like(self.alpha)
        for i in range(num_layers):
            for j in range(i, num_layers):
                mask[j, :, i, :] = 0.0  # Set to zero for j >= i
        
        # Apply the mask to alpha parameters
        self.alpha.data = self.alpha.data * mask

        # Set prior parameters
        self.prior_params = prior_params

    def forward(self, events):
        """
        Compute the log-likelihood of the Hawkes process.

        :param events: A tensor of shape (num_layers, num_groups_per_layer) indicating the occurrence of events
        :return: Log-likelihood value
        """
        log_likelihood = 0.0

        # Iterate over each layer and group within that layer
        for i in range(self.num_layers):
            for m in range(self.num_groups_per_layer):
                # Start with the base rate for this group
                lambda_im = self.mu[i, m]

                # Accumulate influences from previous layers only
                for k in range(i):  # Only consider layers before the current layer i
                    for n in range(self.num_groups_per_layer):
                        # Update lambda_im based on the events in previous layers
                        # This uses only the alpha parameters representing forward influence
                        lambda_im = lambda_im + self.alpha[i, m, k, n] * torch.exp(-self.beta[i, m, k, n] * (i - k)) * events[k, n]

                # Use torch.log1p for numerical stability when argument is close to 0
                log_likelihood = log_likelihood + torch.log1p(lambda_im - 1) * events[i, m] - lambda_im

        return log_likelihood
    
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

    def sample_parameters(self, events, num_samples=500, proposal_std=0.01):
        """
        Sample parameters using Metropolis-Hastings MCMC.
        """
        samples = []
        with torch.no_grad():
            # Initial log-likelihood
            log_likelihood_current = self.forward(events)

            for _ in range(num_samples):
                # Sampling for each parameter
                new_mu, log_likelihood_current = self.metropolis_hastings_step(self.mu, proposal_std, log_likelihood_current, events)
                new_alpha, log_likelihood_current = self.metropolis_hastings_step(self.alpha, proposal_std, log_likelihood_current, events)
                new_beta, log_likelihood_current = self.metropolis_hastings_step(self.beta, proposal_std, log_likelihood_current, events)
                for i in range(self.num_layers):
                    for j in range(i, self.num_layers):
                        new_alpha[j, :, i, :] = 0.0
                self._update_parameter(self.mu, new_mu)
                self._update_parameter(self.alpha, new_alpha)
                self._update_parameter(self.beta, new_beta)

                # Store current sample
                samples.append((self.mu.clone(), self.alpha.clone(), self.beta.clone()))

        return samples

class HawkesProcessModelSGD(nn.Module):
    def __init__(self, num_layers, num_groups_per_layer, learning_rate=0.001, decay=0.1):
        super(HawkesProcessModelSGD, self).__init__()
        self.num_layers = num_layers
        self.num_groups_per_layer = num_groups_per_layer
        self.learning_rate = learning_rate

        # Initialize parameters for the Hawkes process
        #self.mu = nn.Parameter(torch.full((num_layers, num_groups_per_layer), init_mu_value))
        #self.alpha = nn.Parameter(torch.full((num_layers, num_groups_per_layer, num_layers, num_groups_per_layer), init_alpha_value))
        #self.beta = nn.Parameter(torch.full((num_layers, num_groups_per_layer, num_layers, num_groups_per_layer), init_beta_value))

        self.mu = nn.Parameter(torch.rand(num_layers, num_groups_per_layer))
        self.alpha = nn.Parameter(torch.rand(num_layers, num_groups_per_layer, num_layers, num_groups_per_layer))
        self.beta = nn.Parameter(torch.rand(num_layers, num_groups_per_layer, num_layers, num_groups_per_layer))

        # Apply a mask to alpha to ensure causality (no self-excitation)
        self.alpha.data = self.apply_causality_mask(self.alpha.data)
        self.optimizer = optim.SGD(self.parameters(), lr=learning_rate, weight_decay=decay)

    @staticmethod
    def apply_causality_mask(alpha):
        mask = torch.ones_like(alpha)
        num_layers = alpha.shape[0]
        for i in range(num_layers):
            for j in range(i, num_layers):
                mask[j, :, i, :] = 0.0  # Set to zero for j >= i
        return alpha * mask

    def forward(self, events):
        # Vectorize this method to avoid nested loops
        positive_beta = torch.clamp(self.beta, min=1e-6)
        lambda_im = self.mu + torch.sum(
            self.alpha * torch.exp(-positive_beta* torch.arange(self.num_layers).view(-1, 1, 1, 1)) * events.unsqueeze(0).unsqueeze(0),
            dim=(2, 3)
        ).squeeze()
        lambda_im = torch.clamp(lambda_im, min=1e-6, max=1e6)
        log_likelihood = torch.sum(torch.log(lambda_im) * events - lambda_im)
        #log_likelihood = torch.sum(torch.log1p(lambda_im - 1) * events - lambda_im)
        return log_likelihood


    def sgd_step(self, events):
        self.optimizer.zero_grad()
        log_likelihood = self.forward(events)
        log_likelihood.backward()
        self.optimizer.step()
        self.alpha.data = self.apply_causality_mask(self.alpha.data)

    def fit(self, events, tolerance=1e-5, max_iterations=10000):
        """
        Fit the model to the events using SGD, stopping based on learning rate convergence.

        :param events: Event data
        :param tolerance: The threshold for the learning rate change to determine convergence
        :param max_iterations: Maximum number of iterations to prevent infinite loops
        """
        prev_loss = float('inf')
        loss = []
        for iteration in range(max_iterations):
            self.sgd_step(events)

            # Compute current loss
            with torch.no_grad():
                current_loss = self.forward(events).item()
                loss.append(current_loss)
            # Check for convergence
            lr_change = abs(prev_loss - current_loss)
            self.loss = loss
            if lr_change < tolerance:
                print(f"Convergence reached at iteration {iteration}")
                break

            prev_loss = current_loss
        else:
            print(f"Reached maximum iterations ({max_iterations}) without convergence.")

class HawkesProcessModel__(nn.Module):
    def __init__(self, num_neurons, prior_params):
        super(HawkesProcessModel__, self).__init__()

        # Initialize parameters based on the total number of neurons
        size_mu = (num_neurons,)
        self.mu = nn.Parameter(torch.full(size_mu, prior_params["mu_prior"]))
        self.alpha = nn.Parameter(torch.full((num_neurons, num_neurons), prior_params["alpha_prior"]))
        self.beta = nn.Parameter(torch.full((num_neurons, num_neurons), prior_params["beta_prior"]))

        #mask = torch.ones_like(self.alpha)
        #for i in range(num_layers):
        #    for j in range(i, num_layers):
        #        mask[j, :, i, :] = 0.0  # Set to zero for j >= i
        
        # Apply the mask to alpha parameters
        #self.alpha.data = self.alpha.data * mask

        # Set prior parameters
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