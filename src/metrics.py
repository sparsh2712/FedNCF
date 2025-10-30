import numpy as np
import os


class CommunicationTracker:
    """
    Track theoretical communication costs based on parameter counts.
    Computes actual bytes transmitted for different compression methods.
    """

    def __init__(self, compression_method="baseline"):
        """
        Args:
            compression_method: "baseline", "signsgd", "quantization", "sparsification"
        """
        self.compression_method = compression_method
        self.reset()

    def reset(self):
        self.server_to_client_bytes = 0
        self.client_to_server_bytes = 0
        self.total_bytes = 0

    def add_server_to_client(self, num_bytes, num_clients=1):
        total = num_bytes * num_clients
        self.server_to_client_bytes += total
        self.total_bytes += total

    def add_client_to_server(self, num_bytes):
        self.client_to_server_bytes += num_bytes
        self.total_bytes += num_bytes

    def get_model_bytes(self, model, compression_method=None):
        """
        Compute theoretical bytes for transmitting model parameters.
        Based on parameter count and compression method.

        Args:
            model: PyTorch model
            compression_method: Override default compression method

        Returns:
            Number of bytes (theoretical, based on param count)
        """
        if compression_method is None:
            compression_method = self.compression_method

        # Count total parameters
        total_params = sum(param.numel() for param in model.parameters())

        if compression_method == "baseline":
            # Float32: 4 bytes per parameter
            return total_params * 4

        elif compression_method == "signsgd":
            # 1 bit per parameter
            # Convert bits to bytes (round up)
            return (total_params + 7) // 8  # Ceiling division

        elif compression_method == "quantization":
            # 8-bit quantization: 1 byte per parameter
            return total_params * 1

        elif compression_method == "sparsification":
            # Assume 10% sparsity (send 10% of parameters)
            # Need indices (int32) + values (float32)
            sparse_params = int(total_params * 0.1)
            return sparse_params * (4 + 4)  # 4 bytes index + 4 bytes value

        else:
            return total_params * 4

    def get_file_bytes(self, file_path):
        """Get actual file size (for debugging/comparison)."""
        return os.path.getsize(file_path)

    def get_stats(self):
        return {
            'server_to_client_bytes': self.server_to_client_bytes,
            'client_to_server_bytes': self.client_to_server_bytes,
            'total_bytes': self.total_bytes,
            'total_mb': round(self.total_bytes / (1024 * 1024), 4)
        }


def hit_ratio(y, pred, N=10):
    mask = np.zeros_like(y)
    mask[y > 0] = 1
    pred_masked = pred * mask
    best_index = np.argmax(y)
    pred_masked_indexes = np.argsort(pred_masked)[::-1][:N]
    if best_index in pred_masked_indexes:
        return 1
    else:
        return 0


def ndcg(y, pred, N=10):
    actual_recommendation_best_10indexes = np.argsort(y)[::-1][:N]
    actual_recommendation_best_10 = y[actual_recommendation_best_10indexes]
    predicted_recommendation_best_10 = pred[actual_recommendation_best_10indexes]
    predicted_recommendation_best_10 = np.around(predicted_recommendation_best_10)
    predicted_recommendation_best_10[predicted_recommendation_best_10 < 0] = 0
    dcg_numerator = np.power(2, predicted_recommendation_best_10) - 1
    denomimator = np.log2(np.arange(start=2, stop=N + 2))
    idcg_numerator = np.power(2, actual_recommendation_best_10) - 1
    dcg = np.sum(dcg_numerator / denomimator)
    idcg = np.sum(idcg_numerator / denomimator)
    if idcg != 0:
        ndcg = dcg / idcg
    else:
        ndcg = 0.0
    return ndcg


def compute_metrics(y, pred, metric_functions=None):
    if metric_functions is None:
        metric_functions = [hit_ratio, ndcg]
    return [fun(y, pred) for fun in metric_functions]
