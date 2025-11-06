import argparse
import os
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image

def main():
    parser = argparse.ArgumentParser(
        description="UMAP-based GIF visualization of GA population evolution."
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Name of subfolder inside 'outputs' containing population_history.csv"
    )
    args = parser.parse_args()

    # Path to CSV inside outputs/<folder>
    base_dir = "outputs"
    folder_path = os.path.join(base_dir, args.folder)
    csv_path = os.path.join(folder_path, "population_history.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")
    print(f"Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # CSV expected format: [generation, g1, g2, ..., gN, fitness]
    gene_columns = df.columns[1:-1]   # all genes except first and last
    genotypes = df[gene_columns].values
    # Convert negative fitness -> positive loss
    fitness_raw = df[df.columns[-1]].values
    loss = -fitness_raw
    generations = df[df.columns[0]].values
    unique_gens = np.unique(generations)

    print(f"Loaded {len(df)} individuals.")
    print(f"Number of genes per individual: {len(gene_columns)}")
    print(f"Number of generations: {len(unique_gens)}")

    # ---- UMAP ON FULL DATASET ----
    print("Running UMAP dimensionality reduction on full dataset...")
    reducer = umap.UMAP(
        n_neighbors=10,
        min_dist=0.1,
        n_components=2,
        metric="euclidean",
        random_state=42
    )
    embedding = reducer.fit_transform(genotypes)

    # ---- PREPARE GLOBAL COLOR SCALE ----
    epsilon = 1e-9
    loss += epsilon  # ensure strictly positive for LogNorm
    global_min = np.min(loss)
    global_max = np.max(loss)

    # ---- COMPUTE GLOBAL AXES LIMITS ----
    x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
    y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
    padding = 0.05  # optional 5% padding
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= padding * x_range
    x_max += padding * x_range
    y_min -= padding * y_range
    y_max += padding * y_range

    # ---- GENERATE FRAMES ----
    frames = []
    print("Generating frames...")

    for gen in unique_gens:
        mask = (generations == gen)
        loss_gen = loss[mask]

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            c=loss_gen,
            cmap="viridis_r",
            norm=LogNorm(vmin=global_min, vmax=global_max)
        )
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        cbar = plt.colorbar(scatter)
        cbar.set_label("Loss (log scale)")
        plt.xlabel("UMAP dimension 1")
        plt.ylabel("UMAP dimension 2")
        plt.title(f"UMAP Projection — Generation {gen}")
        # Save frame
        frame_path = os.path.join(folder_path, f"frame_{gen:03d}.png")
        plt.savefig(frame_path, dpi=200)
        plt.close()
        frames.append(Image.open(frame_path))

    # ---- CREATE GIF ----
    gif_path = os.path.join(folder_path, "umap_evolution.gif")
    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=300,   # ms per frame
        loop=0          # infinite loop
    )
    print(f"\nGIF saved at: {gif_path}")
    print("Cleaning up temporary frames...")
    for gen in unique_gens:
        os.remove(os.path.join(folder_path, f"frame_{gen:03d}.png"))
    print("Done!")

if __name__ == "__main__":
    main()
