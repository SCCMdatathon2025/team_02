import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

import os

import torch

os.makedirs("plots", exist_ok=True)

client = None

GLOBAL_DIR = "data_cache"

os.makedirs(GLOBAL_DIR, exist_ok=True)


def run(query: str, key: str | None = None):
    # If key is provided, check cache first
    if key and os.path.exists(os.path.join(GLOBAL_DIR, f"{key}.csv")):
        return pd.read_csv(os.path.join(GLOBAL_DIR, f"{key}.csv"))

    from google.cloud import bigquery

    global client

    if client is None:
        client = bigquery.Client(project="sepsis-nlp")

    print(f"Executing query for {key}")

    # Execute query
    results = client.query(query)
    df = results.to_dataframe()

    # Save to cache if key is provided
    if key:
        df.to_csv(os.path.join(GLOBAL_DIR, f"{key}.csv"), index=False)

    return df


def silhouette_score(X, labels):
    """
    Compute the silhouette score for a given dataset and labels with efficient sampling
    """
    print("Calculating silhouette score...")
    X = torch.tensor(X, dtype=torch.float32, device=get_device())
    # Cap sampling at 3000 for efficiency with large datasets
    sample_size = min(3000, len(X))
    subsampled_idx = np.random.choice(len(X), size=sample_size, replace=False)
    res = silhouette_gpu(X[subsampled_idx], labels[subsampled_idx])
    print(f"Calculated silhouette score: {res}")
    return res


def get_device():
    """
    Get the best available device (MPS > CUDA > CPU)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return device


def get_cohort() -> pd.DataFrame:
    query = """
SELECT * FROM `sepsis-nlp.team_2.cohort_flags_final`
"""
    return run(query, "cohort_cluster")


def preprocess_data(cohort: pd.DataFrame):
    """
    Preprocess data for clustering with mixed data types
    """
    # Separate categorical and numerical columns
    categorical_cols = ["race_name", "gender_name", "site_location", "ethnicity_name"]
    binary_cols = [
        "died_in_hospital",
        "died_in_30_days",
        "steroid_flag",
        "narcotic_flag",
        "sedative_flag",
        "vasopressor_flag",
    ]
    numerical_cols = ["age", "los"]

    # Handle missing values
    cohort_clean = cohort.copy()

    # For categorical columns, fill with mode
    for col in categorical_cols:
        cohort_clean[col] = cohort_clean[col].fillna(cohort_clean[col].mode()[0])

    # For numerical columns, fill with median
    for col in numerical_cols:
        cohort_clean[col] = cohort_clean[col].fillna(cohort_clean[col].median())

    # For binary columns, fill with 0
    for col in binary_cols:
        cohort_clean[col] = cohort_clean[col].fillna(0)

    return cohort_clean, categorical_cols, binary_cols, numerical_cols


def encode_categorical_data(cohort: pd.DataFrame, categorical_cols: list):
    """
    Create different encodings for different clustering algorithms
    """
    cohort_encoded = cohort.copy()

    # Label encoding for categorical variables (for K-Prototypes)
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        cohort_encoded[col + "_encoded"] = le.fit_transform(cohort_encoded[col])
        label_encoders[col] = le

    # One-hot encoding (for traditional algorithms)
    cohort_onehot = pd.get_dummies(
        cohort, columns=categorical_cols, prefix=categorical_cols
    )

    return cohort_encoded, cohort_onehot, label_encoders


class GPUKMeans:
    """
    GPU-accelerated K-Means clustering using PyTorch
    """

    def __init__(self, n_clusters, max_iters=50, tol=1e-4, device=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.device = device or get_device()

    def fit_predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples, n_features = X_tensor.shape

        # Initialize centroids using k-means++
        centroids = self._init_centroids(X_tensor)

        for iteration in range(self.max_iters):
            # Compute distances to centroids
            distances = torch.cdist(X_tensor, centroids)

            # Assign points to closest centroid
            labels = torch.argmin(distances, dim=1)

            # Update centroids
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X_tensor[mask].mean(dim=0)
                else:
                    new_centroids[k] = centroids[k]

            diff = torch.norm(centroids - new_centroids)

            # Check for convergence
            if diff < self.tol:
                break

            centroids = new_centroids

        self.centroids = centroids
        return labels.cpu().numpy()

    def _init_centroids(self, X):
        """Initialize centroids using k-means++"""
        n_samples, n_features = X.shape
        centroids = torch.zeros(self.n_clusters, n_features, device=self.device)

        # Choose first centroid randomly
        centroids[0] = X[torch.randint(0, n_samples, (1,))]

        for k in range(1, self.n_clusters):
            # Compute distances to closest centroid
            distances = torch.cdist(X, centroids[:k])
            min_distances = torch.min(distances, dim=1)[0]

            # Choose next centroid with probability proportional to squared distance
            probabilities = min_distances**2
            probabilities = probabilities / probabilities.sum()

            # Sample next centroid
            idx = torch.multinomial(probabilities, 1)
            centroids[k] = X[idx]

        return centroids


class GPUGaussianMixture:
    """
    GPU-accelerated Gaussian Mixture Model using PyTorch
    """

    def __init__(self, n_components, max_iters=50, tol=1e-3, device=None):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.device = device or get_device()

    def fit_predict(self, X):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        n_samples, n_features = X_tensor.shape

        # Initialize parameters
        self._initialize_parameters(X_tensor)

        prev_log_likelihood = float("-inf")

        for iteration in range(self.max_iters):
            # E-step: compute responsibilities
            log_prob = self._compute_log_prob(X_tensor)
            log_resp = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
            resp = torch.exp(log_resp)

            # M-step: update parameters
            self._update_parameters(X_tensor, resp)

            # Check convergence
            log_likelihood = torch.logsumexp(log_prob, dim=1).sum().item()
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood

        # Predict cluster assignments
        log_prob = self._compute_log_prob(X_tensor)
        labels = torch.argmax(log_prob, dim=1)

        return labels.cpu().numpy()

    def _initialize_parameters(self, X):
        n_samples, n_features = X.shape

        # Initialize means using k-means
        kmeans = GPUKMeans(self.n_components, device=self.device)
        labels = kmeans.fit_predict(X.cpu().numpy())
        self.means = kmeans.centroids

        # Initialize covariances and weights
        self.covariances = torch.stack(
            [
                torch.eye(n_features, device=self.device)
                for _ in range(self.n_components)
            ]
        )
        self.weights = (
            torch.ones(self.n_components, device=self.device) / self.n_components
        )

    def _compute_log_prob(self, X):
        n_samples = X.shape[0]
        log_prob = torch.zeros(n_samples, self.n_components, device=self.device)

        for k in range(self.n_components):
            diff = X - self.means[k]
            cov_inv = torch.inverse(self.covariances[k])

            # Compute log probability for multivariate normal
            log_prob[:, k] = (
                torch.log(self.weights[k])
                - 0.5 * torch.sum(diff @ cov_inv * diff, dim=1)
                - 0.5 * torch.logdet(self.covariances[k])
                - 0.5
                * X.shape[1]
                * torch.log(torch.tensor(2 * np.pi, device=self.device))
            )

        return log_prob

    def _update_parameters(self, X, resp):
        n_samples = X.shape[0]

        # Update weights
        self.weights = resp.sum(dim=0) / n_samples

        # Update means
        for k in range(self.n_components):
            self.means[k] = (resp[:, k : k + 1] * X).sum(dim=0) / resp[:, k].sum()

        # Update covariances
        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_diff = resp[:, k : k + 1] * diff
            self.covariances[k] = (weighted_diff.T @ diff) / resp[:, k].sum()

            # Add regularization to avoid singular matrices
            self.covariances[k] += 1e-6 * torch.eye(X.shape[1], device=self.device)


def silhouette_gpu(X_torch, labels):
    """
    Exact silhouette computed fully on GPU in ~O(n²/k) memory-sliced chunks.
    X_torch must be (n, d) on CUDA/MPS.
    """
    labels = torch.as_tensor(labels, device=X_torch.device, dtype=torch.long)
    n_clusters = labels.max().item() + 1
    n_samples = X_torch.size(0)

    # Pre-compute cluster masks
    cluster_masks = [(labels == c).nonzero(as_tuple=True)[0] for c in range(n_clusters)]

    # Allocate result
    s = torch.empty(n_samples, device=X_torch.device)

    # Compute in smaller batches for large datasets
    batch = min(2000, n_samples)  # Reduced batch size for efficiency
    for start in range(0, n_samples, batch):
        end = min(start + batch, n_samples)
        X_batch = X_torch[start:end]

        # a(i): intra-cluster distance
        a = torch.empty(end - start, device=X_torch.device)
        for c in range(n_clusters):
            mask = cluster_masks[c]
            if mask.numel() == 0:
                continue
            dists = torch.cdist(X_batch, X_torch[mask])
            a[labels[start:end] == c] = dists.mean(dim=1)[labels[start:end] == c]

        # b(i): nearest other cluster
        b = torch.full_like(a, float("inf"))
        for c in range(n_clusters):
            mask = cluster_masks[c]
            if mask.numel() == 0:
                continue
            d = torch.cdist(X_batch, X_torch[mask]).mean(dim=1)
            b = torch.where(labels[start:end] != c, torch.minimum(b, d), b)

        s[start:end] = (b - a) / torch.maximum(a, b)

    return s.mean().item()


def run_gpu_kmeans_clustering(
    data: pd.DataFrame, n_clusters_range: range = range(2, 8), device=None
):
    """
    Run GPU-accelerated K-Means clustering with optimal k selection
    """
    device = device or get_device()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    silhouette_scores = []
    best_labels = None
    best_k = 2

    print("🔥 Running GPU-accelerated K-Means...")

    for k in n_clusters_range:
        kmeans_gpu = GPUKMeans(n_clusters=k, device=device)
        cluster_labels = kmeans_gpu.fit_predict(data_scaled)

        if len(set(cluster_labels)) > 1:
            sil_score = silhouette_score(data_scaled, cluster_labels)
            silhouette_scores.append(sil_score)

            if len(silhouette_scores) == 1 or sil_score > max(silhouette_scores[:-1]):
                best_labels = cluster_labels
                best_k = k
        else:
            silhouette_scores.append(-1)

    print(f"Best K: {best_k} - Best Labels: {best_labels}")

    return best_labels, best_k, silhouette_scores


def run_gpu_gaussian_mixture_clustering(
    data: pd.DataFrame, n_clusters_range: range = range(2, 8), device=None
):
    """
    Run GPU-accelerated Gaussian Mixture Model clustering
    """
    device = device or get_device()
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    print("🔥 Running GPU-accelerated Gaussian Mixture Model...")

    best_labels = None
    best_k = 2
    best_score = -1

    for k in n_clusters_range:
        try:
            gmm_gpu = GPUGaussianMixture(n_components=k, device=device)
            cluster_labels = gmm_gpu.fit_predict(data_scaled)

            if len(set(cluster_labels)) > 1:
                sil_score = silhouette_score(data_scaled, cluster_labels)
                if sil_score > best_score:
                    best_score = sil_score
                    best_labels = cluster_labels
                    best_k = k
        except Exception as e:
            print(f"GPU GMM failed for k={k}: {e}")
            continue

    return best_labels, best_k, best_score


def run_kmeans_clustering(data: pd.DataFrame, n_clusters_range: range = range(2, 8)):
    """
    Run K-Means clustering with optimal k selection
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    silhouette_scores = []
    inertias = []

    for k in n_clusters_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)  # Reduced n_init
        cluster_labels = kmeans.fit_predict(data_scaled)

        if len(set(cluster_labels)) > 1:  # More than one cluster
            sil_score = silhouette_score(data_scaled, cluster_labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(-1)

        inertias.append(kmeans.inertia_)

    # Select optimal k based on silhouette score
    optimal_k = n_clusters_range[np.argmax(silhouette_scores)]

    # Final clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=5)
    cluster_labels = kmeans.fit_predict(data_scaled)

    return cluster_labels, optimal_k, silhouette_scores, inertias, scaler


def run_kprototypes_clustering(
    cohort: pd.DataFrame,
    categorical_indices: list,
    n_clusters_range: range = range(2, 8),
):
    """
    Run K-Prototypes clustering for mixed data types
    """
    # Prepare data for K-Prototypes
    data_for_kproto = cohort.select_dtypes(include=[np.number]).values

    best_cost = float("inf")
    best_k = 2
    best_labels = None
    costs = []

    for k in n_clusters_range:
        try:
            kproto = KPrototypes(
                n_clusters=k, init="Huang", verbose=0, random_state=42, max_iter=20
            )  # Reduced max_iter
            cluster_labels = kproto.fit_predict(
                data_for_kproto, categorical=categorical_indices
            )
            costs.append(kproto.cost_)

            if kproto.cost_ < best_cost:
                best_cost = kproto.cost_
                best_k = k
                best_labels = cluster_labels
        except:
            costs.append(float("inf"))

    return best_labels, best_k, costs


def run_gaussian_mixture_clustering(
    data: pd.DataFrame, n_clusters_range: range = range(2, 8)
):
    """
    Run Gaussian Mixture Model clustering
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    bic_scores = []
    aic_scores = []

    for k in n_clusters_range:
        gmm = GaussianMixture(
            n_components=k, random_state=42, max_iter=50
        )  # Reduced max_iter
        gmm.fit(data_scaled)

        bic_scores.append(gmm.bic(data_scaled))
        aic_scores.append(gmm.aic(data_scaled))

    # Select optimal k based on BIC (lower is better)
    optimal_k = n_clusters_range[np.argmin(bic_scores)]

    # Final clustering with optimal k
    gmm = GaussianMixture(n_components=optimal_k, random_state=42, max_iter=50)
    cluster_labels = gmm.fit_predict(data_scaled)

    return cluster_labels, optimal_k, bic_scores, aic_scores


def run_dbscan_clustering(data: pd.DataFrame, eps_range: list = None):
    """
    Run DBSCAN clustering
    """
    if eps_range is None:
        eps_range = [0.3, 0.5, 0.7, 1.0, 1.5]

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    best_score = -1
    best_eps = 0.5
    best_labels = None

    for eps in eps_range:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        cluster_labels = dbscan.fit_predict(data_scaled)

        # Skip if all points are noise or only one cluster
        if len(set(cluster_labels)) > 1 and -1 not in cluster_labels:
            score = silhouette_score(data_scaled, cluster_labels)
            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = cluster_labels

    return best_labels, best_eps, best_score


def evaluate_clustering(data: pd.DataFrame, labels: np.ndarray, algorithm_name: str):
    """
    Evaluate clustering quality using multiple metrics
    """
    if labels is None or len(set(labels)) <= 1:
        return None

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Remove noise points for DBSCAN
    if -1 in labels:
        mask = labels != -1
        data_scaled = data_scaled[mask]
        labels = labels[mask]

    if len(set(labels)) <= 1:
        return None

    silhouette = silhouette_score(data_scaled, labels)
    calinski_harabasz = calinski_harabasz_score(data_scaled, labels)
    davies_bouldin = davies_bouldin_score(data_scaled, labels)

    return {
        "algorithm": algorithm_name,
        "n_clusters": len(set(labels)),
        "silhouette_score": silhouette,
        "calinski_harabasz_score": calinski_harabasz,
        "davies_bouldin_score": davies_bouldin,
    }


def visualize_clusters(
    cohort: pd.DataFrame, labels: np.ndarray, algorithm_name: str, numerical_cols: list
):
    """
    Create simplified visualizations for clustering results
    """
    if labels is None:
        return

    # Create PCA visualization
    numerical_data = cohort[numerical_cols]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(numerical_data)

    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_scaled)

    plt.figure(figsize=(12, 4))

    # PCA plot
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(
        data_pca[:, 0], data_pca[:, 1], c=labels, cmap="viridis", alpha=0.6, s=1
    )
    plt.colorbar(scatter)
    plt.title(f"{algorithm_name} - PCA Visualization")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")

    # Cluster distribution
    plt.subplot(1, 2, 2)
    unique_labels, counts = np.unique(labels, return_counts=True)
    plt.bar(unique_labels, counts)
    plt.title(f"{algorithm_name} - Cluster Sizes")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Patients")

    plt.tight_layout()
    plt.savefig(
        f'plots/{algorithm_name.lower().replace(" ", "_")}_clustering.png',
        dpi=150,  # Reduced DPI for faster saving
        bbox_inches="tight",
    )
    plt.close()  # Close figure to free memory


def analyze_cluster_characteristics(
    cohort: pd.DataFrame, labels: np.ndarray, algorithm_name: str
):
    """
    Analyze characteristics of each cluster
    """
    if labels is None:
        return

    cohort_with_clusters = cohort.copy()
    cohort_with_clusters["cluster"] = labels

    print(f"\n=== {algorithm_name} Cluster Analysis ===")

    # Overall statistics by cluster
    for cluster_id in sorted(cohort_with_clusters["cluster"].unique()):
        if cluster_id == -1:  # Skip noise points in DBSCAN
            continue

        cluster_data = cohort_with_clusters[
            cohort_with_clusters["cluster"] == cluster_id
        ]
        print(f"\nCluster {cluster_id} (n={len(cluster_data)}):")

        # Numerical features
        print(
            f"  Age: {cluster_data['age'].mean():.1f} ± {cluster_data['age'].std():.1f}"
        )
        print(
            f"  Length of Stay: {cluster_data['los'].mean():.1f} ± {cluster_data['los'].std():.1f} hours"
        )

        # Binary features
        for col in [
            "died_in_hospital",
            "died_in_30_days",
            "steroid_flag",
            "narcotic_flag",
            "sedative_flag",
            "vasopressor_flag",
        ]:
            rate = cluster_data[col].mean()
            print(f"  {col}: {rate:.1%}")

        # Categorical features
        print(f"  Most common race: {cluster_data['race_name'].mode().iloc[0]}")
        print(f"  Most common gender: {cluster_data['gender_name'].mode().iloc[0]}")


def run_clustering(cohort: pd.DataFrame):
    """
    Main clustering function implementing optimized algorithms for large datasets
    """
    print("Starting optimized clustering analysis for large dataset...")
    print(f"Dataset shape: {cohort.shape}")

    # Check GPU availability
    device = get_device()
    print(f"Using device: {device}")

    # Preprocess data
    cohort_clean, categorical_cols, binary_cols, numerical_cols = preprocess_data(
        cohort
    )

    # Create different data encodings
    cohort_encoded, cohort_onehot, label_encoders = encode_categorical_data(
        cohort_clean, categorical_cols
    )

    # Prepare data for traditional algorithms (one-hot encoded)
    traditional_data = cohort_onehot.select_dtypes(include=[np.number])

    # Store results
    results = []

    # 1. K-Means (GPU if available, otherwise CPU)
    if device.type in ["mps", "cuda"]:
        print("\n🔥 1. Running GPU-Accelerated K-Means Clustering...")
        kmeans_labels, kmeans_k, kmeans_scores = run_gpu_kmeans_clustering(
            traditional_data, device=device
        )
        kmeans_eval = evaluate_clustering(
            traditional_data, kmeans_labels, "GPU K-Means"
        )
        kmeans_name = "GPU K-Means"
    else:
        print("\n💻 1. Running CPU K-Means Clustering...")
        kmeans_labels, kmeans_k, kmeans_scores, inertias, scaler = (
            run_kmeans_clustering(traditional_data)
        )
        kmeans_eval = evaluate_clustering(
            traditional_data, kmeans_labels, "CPU K-Means"
        )
        kmeans_name = "CPU K-Means"

    if kmeans_eval:
        results.append(kmeans_eval)
        visualize_clusters(cohort_clean, kmeans_labels, kmeans_name, numerical_cols)
        analyze_cluster_characteristics(cohort_clean, kmeans_labels, kmeans_name)

    # 2. Gaussian Mixture Models
    if device.type in ["mps", "cuda"]:
        print(f"\n🔥 2. Running GPU-Accelerated Gaussian Mixture Model...")
        gpu_gmm_labels, gpu_gmm_k, gpu_gmm_score = run_gpu_gaussian_mixture_clustering(
            traditional_data, device=device
        )
        gpu_gmm_eval = evaluate_clustering(
            traditional_data, gpu_gmm_labels, "GPU Gaussian Mixture"
        )
        if gpu_gmm_eval:
            results.append(gpu_gmm_eval)
            visualize_clusters(
                cohort_clean, gpu_gmm_labels, "GPU Gaussian Mixture", numerical_cols
            )
            analyze_cluster_characteristics(
                cohort_clean, gpu_gmm_labels, "GPU Gaussian Mixture"
            )

    print(f"\n💻 3. Running CPU Gaussian Mixture Model...")
    gmm_labels, gmm_k, bic_scores, aic_scores = run_gaussian_mixture_clustering(
        traditional_data
    )
    gmm_eval = evaluate_clustering(traditional_data, gmm_labels, "CPU Gaussian Mixture")
    if gmm_eval:
        results.append(gmm_eval)
        visualize_clusters(
            cohort_clean, gmm_labels, "CPU Gaussian Mixture", numerical_cols
        )
        analyze_cluster_characteristics(
            cohort_clean, gmm_labels, "CPU Gaussian Mixture"
        )

    # 4. K-Prototypes for mixed data
    print(f"\n🎯 4. Running K-Prototypes Clustering...")
    categorical_indices = [
        i
        for i, col in enumerate(cohort_encoded.columns)
        if any(cat_col in col for cat_col in categorical_cols)
    ]

    try:
        kproto_labels, kproto_k, costs = run_kprototypes_clustering(
            cohort_encoded, categorical_indices
        )
        kproto_eval = evaluate_clustering(
            traditional_data, kproto_labels, "K-Prototypes"
        )
        if kproto_eval:
            results.append(kproto_eval)
            visualize_clusters(
                cohort_clean, kproto_labels, "K-Prototypes", numerical_cols
            )
            analyze_cluster_characteristics(cohort_clean, kproto_labels, "K-Prototypes")
    except Exception as e:
        print(f"K-Prototypes failed: {e}")

    # 5. DBSCAN (density-based clustering)
    # print(f"\n🔍 5. Running DBSCAN...")
    # dbscan_labels, dbscan_eps, dbscan_score = run_dbscan_clustering(traditional_data)
    # dbscan_eval = evaluate_clustering(traditional_data, dbscan_labels, "DBSCAN")
    # if dbscan_eval:
    #     results.append(dbscan_eval)
    #     visualize_clusters(cohort_clean, dbscan_labels, "DBSCAN", numerical_cols)
    #     analyze_cluster_characteristics(cohort_clean, dbscan_labels, "DBSCAN")

    # ─────────────────────────────────────────────────────────
    # ⬇︎ DROP-IN REPLACEMENT FOR THE “Hierarchical Clustering” BLOCK ⬇︎
    # (put this inside run_clustering(), replacing the old step 6)
    # ─────────────────────────────────────────────────────────
    print("\n🌳 6. Running Memory-Optimised Hierarchical Clustering…")

    # Scale numeric data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(traditional_data)

    # Build a sparse k-NN connectivity graph (limits pair-wise distances in RAM)
    connectivity = kneighbors_graph(
        data_scaled,
        n_neighbors=10,  # tweak if clusters look over/under-segmented
        include_self=False,
        n_jobs=-1,  # parallel graph construction
    )

    hierarchical = AgglomerativeClustering(
        n_clusters=5,  # keep your fixed k
        linkage="ward",
        connectivity=connectivity,
        compute_full_tree=False,  # avoids storing the complete dendrogram
    )

    hierarchical_labels = hierarchical.fit_predict(data_scaled)

    hierarchical_eval = evaluate_clustering(
        traditional_data, hierarchical_labels, "Hierarchical (k-NN)"
    )
    if hierarchical_eval:
        results.append(hierarchical_eval)
        visualize_clusters(
            cohort_clean, hierarchical_labels, "Hierarchical (k-NN)", numerical_cols
        )
        analyze_cluster_characteristics(
            cohort_clean, hierarchical_labels, "Hierarchical (k-NN)"
        )
        # ─────────────────────────────────────────────────────────

    # Summary of results
    print("\n" + "=" * 80)
    print("CLUSTERING ALGORITHM COMPARISON")
    print("=" * 80)

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("silhouette_score", ascending=False)

        print(results_df.to_string(index=False))

        print(f"\n🏆 Best performing algorithm: {results_df.iloc[0]['algorithm']}")
        print(f"🏆 Best silhouette score: {results_df.iloc[0]['silhouette_score']:.3f}")

        # Save results
        results_df.to_csv("plots/clustering_comparison.csv", index=False)
        print("\nResults saved to plots/clustering_comparison.csv")
    else:
        print("No successful clustering results to display.")


def main():
    cohort = get_cohort()

    # Convert to datetime and ensure timezone-naive
    cohort["visit_start_datetime"] = pd.to_datetime(cohort["visit_start_datetime"])
    if cohort["visit_start_datetime"].dt.tz is not None:
        cohort["visit_start_datetime"] = cohort["visit_start_datetime"].dt.tz_convert(
            None
        )

    cohort["visit_end_datetime"] = pd.to_datetime(cohort["visit_end_datetime"])
    if cohort["visit_end_datetime"].dt.tz is not None:
        cohort["visit_end_datetime"] = cohort["visit_end_datetime"].dt.tz_convert(None)

    cohort["birth_datetime"] = pd.to_datetime(cohort["birth_datetime"])
    if cohort["birth_datetime"].dt.tz is not None:
        cohort["birth_datetime"] = cohort["birth_datetime"].dt.tz_convert(None)

    # In years
    cohort["age"] = (
        cohort["visit_start_datetime"] - cohort["birth_datetime"]
    ).dt.total_seconds() / (365 * 24 * 60 * 60)

    # In hours
    cohort["los"] = (
        cohort["visit_end_datetime"] - cohort["visit_start_datetime"]
    ).dt.total_seconds() / 3600

    cohort = cohort[
        [
            "race_name",
            "ethnicity_name",
            "gender_name",
            "died_in_hospital",
            "died_in_30_days",
            "steroid_flag",
            "narcotic_flag",
            "sedative_flag",
            "vasopressor_flag",
            "los",
            "age",
            "site_location",
            "nasal_canula_mask",
            "hiflo_oximyzer",
            "cpap_bipap",
            "mechanical_ventilation",
            "ecmo",
            "dialysis",
        ]
    ]

    run_clustering(cohort)


if __name__ == "__main__":
    main()
