"""
02_clustering.py

Exploratory clustering analysis for the bone fracture dataset.
"""
# %% 
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

# Config

INVENTORY_CSV = Path("results/data_check/image_inventory.csv")
RESULTS_DIR = Path("results/clustering")

IMAGE_SIZE = (64, 64)       
N_CLUSTERS = 10            
RANDOM_STATE = 42

MAX_IMAGES_FOR_TSNE = 1000


# %% Helpers

def load_image_as_feature(path: str, image_size: tuple[int, int]) -> np.ndarray:
    """
    Load image, convert to grayscale, resize, normalize, and flatten.

    Grayscale is enough here because this is only exploratory clustering.
    """
    with Image.open(path) as img:
        img = img.convert("L")
        img = img.resize(image_size)

        arr = np.asarray(img, dtype=np.float32) / 255.0
        feature = arr.flatten()

    return feature

def load_clean_inventory(inventory_csv: Path) -> pd.DataFrame:
    """
    Load image inventory and keep only usable images.
    """
    df = pd.read_csv(inventory_csv)

    clean_df = df[
        (df["readable"] == True)
        & (df["is_duplicate"] == False)
    ].copy()

    clean_df = clean_df.reset_index(drop=True)

    print(f"Loaded inventory: {len(df)} images")
    print(f"Clean images used for clustering: {len(clean_df)}")

    return clean_df

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    Convert all images into flattened feature vectors.
    """
    features = []

    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"  Loading image {i + 1}/{len(df)}")

        feature = load_image_as_feature(row["filepath"], IMAGE_SIZE)
        features.append(feature)

    X = np.vstack(features)

    return X

def save_pca_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save PCA scatter plot colored by true class.
    """
    plt.figure(figsize=(10, 8))

    for class_name in sorted(pca_df["class"].unique()):
        subset = pca_df[pca_df["class"] == class_name]
        plt.scatter(
            subset["pca1"],
            subset["pca2"],
            s=18,
            alpha=0.7,
            label=class_name
        )

    plt.title("PCA projection colored by true fracture class")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_kmeans_plot(pca_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save PCA scatter plot colored by K-Means cluster.
    """
    plt.figure(figsize=(10, 8))

    for cluster_id in sorted(pca_df["cluster"].unique()):
        subset = pca_df[pca_df["cluster"] == cluster_id]
        plt.scatter(
            subset["pca1"],
            subset["pca2"],
            s=18,
            alpha=0.7,
            label=f"Cluster {cluster_id}"
        )

    plt.title("PCA projection colored by K-Means cluster")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def save_tsne_plot(tsne_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save t-SNE scatter plot colored by true class.
    """
    plt.figure(figsize=(10, 8))

    for class_name in sorted(tsne_df["class"].unique()):
        subset = tsne_df[tsne_df["class"] == class_name]
        plt.scatter(
            subset["tsne1"],
            subset["tsne2"],
            s=18,
            alpha=0.7,
            label=class_name
        )

    plt.title("t-SNE projection colored by true fracture class")
    plt.xlabel("t-SNE component 1")
    plt.ylabel("t-SNE component 2")
    plt.legend(fontsize=7, markerscale=1.5)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

# %% Main ------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_clean_inventory(INVENTORY_CSV)
    X = build_feature_matrix(df)

    print("Standardizing features …")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Running PCA to 50 dimensions …")
    pca_50 = PCA(n_components=50, random_state=RANDOM_STATE)
    X_pca_50 = pca_50.fit_transform(X_scaled)

    explained = pca_50.explained_variance_ratio_.sum()
    print(f"Explained variance by 50 PCA components: {explained:.3f}")

    print("Running PCA to 2 dimensions for visualization …")
    pca_2 = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca_2 = pca_2.fit_transform(X_scaled)

    print("Running K-Means clustering …")
    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=10
    )
    clusters = kmeans.fit_predict(X_pca_50)

    # result table
    result_df = df[["filepath", "class", "split"]].copy()
    result_df["pca1"] = X_pca_2[:, 0]
    result_df["pca2"] = X_pca_2[:, 1]
    result_df["cluster"] = clusters

    clustering_csv = RESULTS_DIR / "clustering_results.csv"
    result_df.to_csv(clustering_csv, index=False)

    # Cluster/class cross-table
    cluster_class_table = pd.crosstab(
        result_df["cluster"],
        result_df["class"]
    )

    cluster_class_csv = RESULTS_DIR / "cluster_class_table.csv"
    cluster_class_table.to_csv(cluster_class_csv)

    print("\nCluster/class table:")
    print(cluster_class_table)


    # Save Plots

    save_pca_plot(
        result_df,
        RESULTS_DIR / "pca_2d_scatter.png"
    )

    save_kmeans_plot(
        result_df,
        RESULTS_DIR / "kmeans_2d_scatter.png"
    )

    # t-SNE visualization
    print("Running t-SNE visualization …")

    if MAX_IMAGES_FOR_TSNE is not None and len(result_df) > MAX_IMAGES_FOR_TSNE:
        tsne_sample = result_df.sample(
            n=MAX_IMAGES_FOR_TSNE,
            random_state=RANDOM_STATE
        ).copy()

        sample_indices = tsne_sample.index.to_numpy()
        X_tsne_input = X_pca_50[sample_indices]
    else:
        tsne_sample = result_df.copy()
        X_tsne_input = X_pca_50

    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=RANDOM_STATE
    )

    X_tsne = tsne.fit_transform(X_tsne_input)

    tsne_df = tsne_sample[["filepath", "class", "split", "cluster"]].copy()
    tsne_df["tsne1"] = X_tsne[:, 0]
    tsne_df["tsne2"] = X_tsne[:, 1]

    tsne_csv = RESULTS_DIR / "tsne_results.csv"
    tsne_df.to_csv(tsne_csv, index=False)

    save_tsne_plot(
        tsne_df,
        RESULTS_DIR / "tsne_2d_scatter.png"
    )

    # Summary 
    print("\n" + "=" * 30)
    print("CLUSTERING SUMMARY")
    print("=" * 30)
    print(f"Images used                : {len(result_df)}")
    print(f"Image feature size         : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} grayscale")
    print(f"PCA components for K-Means : 50")
    print(f"PCA explained variance     : {explained:.3f}")
    print(f"K-Means clusters           : {N_CLUSTERS}")
    print()
    print("Files written to:")
    print(f"  {clustering_csv}")
    print(f"  {cluster_class_csv}")
    print(f"  {RESULTS_DIR / 'pca_2d_scatter.png'}")
    print(f"  {RESULTS_DIR / 'kmeans_2d_scatter.png'}")
    print(f"  {tsne_csv}")
    print(f"  {RESULTS_DIR / 'tsne_2d_scatter.png'}")
    print("=" * 30)

if __name__ == "__main__":
    main()