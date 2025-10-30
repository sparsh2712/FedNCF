from pathlib import Path
import torch


class FederatedConfig:
    def __init__(
        self,
        num_clients=50,
        user_per_client_range=(1, 10),
        aggregation_epochs=50,
        local_epochs=10,
        batch_size=128,
        latent_dim=32,
        learning_rate=5e-4,
        seed=0,
        device=None,
        models_dir=Path(__file__).parent.parent / "models",
        communication_method="baseline"  # Options: "baseline", "signsgd", "quantization", "sparsification"
    ):
        self.num_clients = num_clients
        self.user_per_client_range = user_per_client_range
        self.aggregation_epochs = aggregation_epochs
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.seed = seed
        self.communication_method = communication_method

        # Validate communication method
        valid_methods = ["baseline", "signsgd", "quantization", "sparsification"]
        if communication_method not in valid_methods:
            raise ValueError(f"communication_method must be one of {valid_methods}")

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.models_dir = Path(models_dir)
        self.local_dir = self.models_dir / "local"
        self.local_items_dir = self.models_dir / "local_items"
        self.central_dir = self.models_dir / "central"

        for directory in [self.local_dir, self.local_items_dir, self.central_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def __repr__(self):
        return (
            f"FederatedConfig("
            f"num_clients={self.num_clients}, "
            f"aggregation_epochs={self.aggregation_epochs}, "
            f"local_epochs={self.local_epochs}, "
            f"batch_size={self.batch_size}, "
            f"latent_dim={self.latent_dim})"
        )


if __name__ == '__main__':
    config = FederatedConfig()
    print(config)