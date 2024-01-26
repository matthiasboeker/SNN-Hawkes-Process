from pathlib import Path
import numpy as np
import torch 
from hawkes_model import HawkesProcessModel

from utils import plot_heatmap, load_in_json


def load_in_spike_trains(path_to_results):
    spike_trains_results = load_in_json(path_to_results)
    correct_trains = []
    wrong_trains = []
    for input_train, hidden_train, output_train, correct in zip(spike_trains_results["data"]["input_trains"],
                                                       spike_trains_results["data"]["hidden_trains"],
                                                       spike_trains_results["data"]["output_trains"],
                                                       spike_trains_results["data"]["correct"]):
            merged = [torch.tensor(spike_train) for spike_train in input_train + hidden_train + output_train]
            if correct:
                 correct_trains.append(merged)
            else:
                 wrong_trains.append(merged)
    return correct_trains, wrong_trains


def main():
    path_to_results = Path(__file__).parent / "spike_trains" / "results_1.json"
    path_to_correct = Path(__file__).parent / "figures" / "correct_predictions"
    path_to_storage = Path(__file__).parent / "parameters"
    path_to_wrong = Path(__file__).parent / "figures" / "wrong_predictions"
    correct_trains, wrong_trains =load_in_spike_trains(path_to_results)
    prior_params = {"mu_prior": 0.1, "alpha_prior": 0.1, "beta_prior": 0.1}


    for i, train in enumerate(correct_trains):
        # Initialize the Hawkes process model
        mc_model = HawkesProcessModel(len(train), prior_params)
        log_likelihood = mc_model(train)
        samples = mc_model.sample_parameters(train, num_samples=50)
        alphas = np.mean(np.stack([sample[1] for sample in samples]), axis=0)
        betas = np.mean(np.stack([sample[2] for sample in samples]), axis=0)
        np.save(path_to_storage / f"alpha_matrix_correct_{i+1}.npy", alphas)
        np.save(path_to_storage / f"beta_matrix_correct_{i+1}.npy", betas)
        print(f"Log-Likelihood of {i+1}:", log_likelihood.item())
        plot_heatmap(alphas, "Heatmap of Alpha Parameters", path_to_correct, i+1)
    
    for i, train in enumerate(wrong_trains):
        # Initialize the Hawkes process model
        mc_model = HawkesProcessModel(len(train), prior_params)
        log_likelihood = mc_model(train)
        samples = mc_model.sample_parameters(train, num_samples=1)
        alphas = np.mean(np.stack([sample[1] for sample in samples]), axis=0)
        betas = np.mean(np.stack([sample[2] for sample in samples]), axis=0)
        np.save(path_to_storage / f"alpha_matrix_wrong_{i+1}.npy", alphas)
        np.save(path_to_storage / f"beta_matrix_wrong_{i+1}.npy", betas)

        plot_heatmap(alphas, "Heatmap of Alpha Parameters Wrong Predictions", path_to_wrong, i+1)
        print(f"Log-Likelihood of {i+1}:", log_likelihood.item())


if __name__ == "__main__":
    main()