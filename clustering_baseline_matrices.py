from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import re 
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def load_and_flatten_files(folder_path, layer_nr):
    # Pattern to extract Layer and Class from file names
    pattern = r"FileNr: (\d+) BatchNr:\d+ Layer:(\d+) Class:(\d+)\.npy"
    
    # Lists to store flattened data and labels
    flattened_data = []
    labels = []

    # Glob to find all files matching the pattern
    #file_paths = glob(os.path.join(folder_path, "FileNr: * BatchNr:* Layer:* Class:*.npy"))
    file_paths = glob.glob(os.path.join(folder_path, "FileNr: * BatchNr:* Layer:* Class:*.npy"))

    for file_path in file_paths:
        # Load the matrix from the file
        
        # Flatten the matrix and append to list
        
        # Extract Layer and Class from file name using regex
        match = re.search(pattern, file_path)
        if match:
            file_nr, layer, cls = match.groups()
            if int(file_nr) > 9:
                continue
            if int(layer) != layer_nr:
                continue
            matrix = np.load(file_path)
            flattened_data.append(matrix.flatten())

            labels.append(int(cls))

    data_array = np.vstack(flattened_data)
    return data_array, labels


def main():
    layer = 6
    folder_path = Path(__file__).parent / "spike_trains" / "baseline_arrays"
    data_array, labels = load_and_flatten_files(folder_path, layer)
    colors = ['red', 'green', 'blue', 'purple', 'orange', 'yellow', 'brown', 'pink', 'gray', 'cyan', 'magenta']


    unique_classes = np.unique(labels)
    if len(unique_classes) > len(colors):
        colors = colors * (len(unique_classes) // len(colors) + 1)

    color_map = {cls: colors[i] for i, cls in enumerate(unique_classes)}

    #pca = PCA(n_components=2)  
    #dim_red_result = pca.fit_transform(data_array)
    tsne = TSNE(n_components=2, random_state=42)  # n_components=2 for 2D visualization
    dim_red_result = tsne.fit_transform(data_array)
    plt.figure(figsize=(15, 10))
    for cls in unique_classes:
        cls_indices = np.where(np.array(labels) == cls)[0]
        #plt.scatter(pca_result[cls_indices, 0], pca_result[cls_indices, 1], label=f'Class {cls}', color=color_map[cls], alpha=0.7)
        plt.scatter(dim_red_result[cls_indices, 0], dim_red_result[cls_indices, 1], label=f'Class {cls}', color=color_map[cls], alpha=0.7)

    #scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], c=labels, cmap='viridis', alpha=0.7)
    #plt.colorbar(scatter, label='Class')
    plt.legend(title='Class Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.9, 0.9])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('Data Colored by Class')
    plt.savefig(f"latest_TSNE_{layer}.png")

if __name__ == "__main__":
    main()