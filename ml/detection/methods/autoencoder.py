"""Autoencoder — detects anomalies via reconstruction error."""
import time
import numpy as np
from ml.detection.method_adapter import BaseDetector, DetectionResult


class AutoencoderDetector(BaseDetector):
    name = "autoencoder"
    explainability_prior = 0.7

    def __init__(self, epochs: int = 100, batch_size: int = 256, patience: int = 10,
                 random_state: int = 42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.random_state = random_state

    def fit_score(self, X: np.ndarray, **kwargs) -> DetectionResult:
        start = time.time()
        n_samples, n_features = X.shape

        if n_samples < 50 or n_features < 2:
            return DetectionResult(
                method_name=self.name,
                anomaly_scores=np.zeros(n_samples),
                duration_ms=0,
                params={},
                metadata={"skipped": True, "reason": "Too few samples or features"},
            )

        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            torch.manual_seed(self.random_state)

            # Architecture: input → 128 → 64 → 32 (bottleneck) → 64 → 128 → input
            bottleneck = min(32, n_features // 2, max(4, n_features // 4))
            layer1 = min(128, n_features * 2)
            layer2 = min(64, bottleneck * 2)

            class AE(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.encoder = nn.Sequential(
                        nn.Linear(n_features, layer1), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(layer1, layer2), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(layer2, bottleneck), nn.ReLU(),
                    )
                    self.decoder = nn.Sequential(
                        nn.Linear(bottleneck, layer2), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(layer2, layer1), nn.ReLU(), nn.Dropout(0.2),
                        nn.Linear(layer1, n_features),
                    )

                def forward(self, x):
                    return self.decoder(self.encoder(x))

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AE().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            criterion = nn.MSELoss(reduction="none")

            X_tensor = torch.FloatTensor(X).to(device)
            dataset = TensorDataset(X_tensor, X_tensor)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # Training with early stopping
            best_loss = float("inf")
            patience_counter = 0
            for epoch in range(self.epochs):
                model.train()
                epoch_loss = 0.0
                for batch_x, _ in loader:
                    optimizer.zero_grad()
                    recon = model(batch_x)
                    loss = criterion(recon, batch_x).mean()
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(loader)
                if avg_loss < best_loss - 1e-5:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

            # Compute reconstruction errors
            model.eval()
            with torch.no_grad():
                reconstructed = model(X_tensor)
                per_feature_error = (X_tensor - reconstructed).pow(2).cpu().numpy()
                total_error = per_feature_error.mean(axis=1)

            duration_ms = int((time.time() - start) * 1000)

            return DetectionResult(
                method_name=self.name,
                anomaly_scores=self.normalize_scores(total_error),
                duration_ms=duration_ms,
                params={"epochs": self.epochs, "bottleneck": bottleneck,
                        "layers": [layer1, layer2, bottleneck]},
                metadata={
                    "per_feature_error_mean": per_feature_error.mean(axis=0).tolist(),
                    "training_loss": best_loss,
                },
            )

        except ImportError:
            # Fallback: sklearn-based simple autoencoder approximation using PCA
            from sklearn.decomposition import PCA
            n_components = min(n_features // 2, 10, n_samples - 1)
            if n_components < 1:
                n_components = 1
            pca = PCA(n_components=n_components, random_state=self.random_state)
            transformed = pca.fit_transform(X)
            reconstructed = pca.inverse_transform(transformed)
            per_feature_error = (X - reconstructed) ** 2
            total_error = per_feature_error.mean(axis=1)
            duration_ms = int((time.time() - start) * 1000)

            return DetectionResult(
                method_name=self.name,
                anomaly_scores=self.normalize_scores(total_error),
                duration_ms=duration_ms,
                params={"fallback": "pca", "n_components": n_components},
                metadata={"per_feature_error_mean": per_feature_error.mean(axis=0).tolist()},
            )

    def get_penalty(self, dcv: dict) -> float:
        n = dcv.get("n", 0)
        if n < 100:
            return 0.5  # Unstable on small datasets
        if n > 500_000:
            return 0.2  # Slow
        return 0.0
