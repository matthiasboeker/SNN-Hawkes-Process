import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns

def load_and_sum_alpha_matrices(path_to_matrices, correct):
    sum_matrix = None
    files = os.listdir(path_to_matrices)
    for file in files:
        if file.endswith('.npy') and file.startswith('alpha') and (correct in file):
            matrix = np.load(path_to_matrices / file)
            if sum_matrix is None:
                sum_matrix = matrix
            else:
                sum_matrix += matrix
    return sum_matrix

def plot_heatmap(matrix, title, path_to_figures):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.savefig(path_to_figures / "summed_alpha_heatmap.png")
    plt.show()


def plot_heatmap(matrix, title, path_to_figures, name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, fmt=".2f", cmap="viridis")
    plt.title(title)
    plt.xlabel("Neuron Index")
    plt.ylabel("Neuron Index")
    plt.savefig(path_to_figures / f"summed_{name}_heatmap.png")
    plt.show()

def main():
    path_to_matrices = Path(__file__).parent / "parameters"  # Update with the correct path
    path_to_figures = Path(__file__).parent / "figures"  # Update with the correct path
    
    # Load the stacked alpha matrices
    alpha_matrices_stack_sine = load_and_sum_alpha_matrices(path_to_matrices, "correct")
    alpha_matrices_stack_rndm = load_and_sum_alpha_matrices(path_to_matrices, "wrong")
    
    # Plot and save the heatmap
    plot_heatmap(alpha_matrices_stack_sine, "Summed Heatmap", path_to_figures, "Sine")
    plot_heatmap(alpha_matrices_stack_rndm, "Summed Heatmap", path_to_figures, "Random")

if __name__ == "__main__":
    main()