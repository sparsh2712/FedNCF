import random
import torch
import numpy as np
from tqdm import tqdm

from client import LocalTrainer
from server import NCFServer
from config import FederatedConfig
from metrics import CommunicationTracker
from communication import get_communication_strategy


class FederatedNCF:
    def __init__(self, ui_matrix, config=None):
        self.config = config or FederatedConfig()
        self.ui_matrix = ui_matrix
        self.num_users = ui_matrix.shape[0]
        self.num_items = ui_matrix.shape[1]
        self.device = torch.device(self.config.device)

        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        self.clients = self._generate_clients()
        self.server_model = self._create_server_model()

        # Initialize communication tracker with compression method
        self.comm_tracker = CommunicationTracker(
            compression_method=self.config.communication_method
        )

        # Initialize communication strategy based on config
        if self.config.communication_method == "signsgd":
            self.comm_strategy = get_communication_strategy(
                self.config.communication_method,
                learning_rate=self.config.learning_rate
            )
        else:
            self.comm_strategy = get_communication_strategy(
                self.config.communication_method
            )

        self._save_server_model(epoch=0)

    def _generate_clients(self):
        clients = []
        start_idx = 0

        for client_id in range(self.config.num_clients):
            num_users = random.randint(*self.config.user_per_client_range)
            end_idx = min(start_idx + num_users, self.num_users)

            client_matrix = self.ui_matrix[start_idx:end_idx]

            trainer = LocalTrainer(
                ui_matrix=client_matrix,
                num_items=self.num_items,
                config=self.config
            )
            clients.append(trainer)

            start_idx = end_idx

            if start_idx >= self.num_users:
                break

        return clients

    def _create_server_model(self):
        return NCFServer(
            num_items=self.num_items,
            hidden_dim=self.config.latent_dim
        ).to(self.device)

    def _load_server_model(self, epoch):
        path = self.config.central_dir / f"server{epoch}.pt"
        return torch.jit.load(str(path), map_location=self.device)

    def _save_server_model(self, epoch):
        path = self.config.central_dir / f"server{epoch}.pt"
        scripted_model = torch.jit.script(self.server_model.cpu())
        torch.jit.save(scripted_model, str(path))
        self.server_model.to(self.device)

    def _save_client_model(self, client_id, model):
        path = self.config.local_dir / f"dp{client_id}.pt"
        scripted_model = torch.jit.script(model.cpu())
        torch.jit.save(scripted_model, str(path))
        model.to(self.device)

    def _distribute_server_weights(self):
        # Track communication: server broadcasts weights to all clients
        # Server always sends full weights (no compression for server→client in standard FL)
        server_model_bytes = self.comm_tracker.get_model_bytes(
            self.server_model,
            compression_method="baseline"  # Server always sends full model
        )
        self.comm_tracker.add_server_to_client(server_model_bytes, num_clients=len(self.clients))

        for client in self.clients:
            client.model.load_server_weights(self.server_model)

    def _train_clients(self, aggregation_epoch):
        results = {"num_users": [], "loss": [], "hit_ratio": [], "ndcg": []}

        progress_bar = tqdm(
            enumerate(self.clients),
            total=len(self.clients),
            desc=f"Epoch {aggregation_epoch}"
        )

        for client_id, client in progress_bar:
            client_results = client.train()
            self._save_client_model(client_id, client.model)

            for key, value in client_results.items():
                results[key].append(value)

            avg_metrics = {
                k: round(sum(v) / len(v), 4)
                for k, v in results.items()
            }
            progress_bar.set_postfix(avg_metrics)

        return results

    def _extract_server_models(self, reference_server_model=None):
        """
        Extract server-relevant parts from client models and compress if needed.

        Args:
            reference_server_model: Reference model for compression (needed for SignSGD)
        """
        for client_id in range(len(self.clients)):
            client_path = self.config.local_dir / f"dp{client_id}.pt"
            client_model = torch.jit.load(str(client_path))

            # Extract server model from client
            server_model = NCFServer(
                num_items=self.num_items,
                hidden_dim=self.config.latent_dim
            )
            server_model.set_weights_from(client_model)

            # Compress using the selected communication strategy
            compressed = self.comm_strategy.compress(
                model=server_model,
                reference_model=reference_server_model
            )

            # Save compressed data
            server_path = self.config.local_items_dir / f"dp{client_id}.pt"
            self.comm_strategy.save_compressed(compressed, str(server_path))

            # Track communication: client uploads compressed model to server
            # Use theoretical bytes based on parameter count and compression method
            compressed_bytes = self.comm_tracker.get_model_bytes(
                server_model,
                compression_method=self.config.communication_method
            )
            self.comm_tracker.add_client_to_server(compressed_bytes)

    def _aggregate_models(self, reference_server_model=None):
        """
        Aggregate models from all clients.

        Args:
            reference_server_model: Reference model for decompression (needed for SignSGD)
        """
        # Load compressed data from all clients
        compressed_data_list = []
        for client_id in range(len(self.clients)):
            path = self.config.local_items_dir / f"dp{client_id}.pt"
            compressed = self.comm_strategy.load_compressed(str(path))
            compressed_data_list.append(compressed)

        if not compressed_data_list:
            return

        # Handle different communication methods
        if self.config.communication_method == "signsgd":
            # Import here to avoid circular dependency
            from communication import SignSGDStrategy

            # SignSGD: majority voting on signs
            aggregated_signs = SignSGDStrategy(
                learning_rate=self.config.learning_rate
            ).aggregate_signs(compressed_data_list)

            # Decompress aggregated signs
            aggregated_state = self.comm_strategy.decompress(
                aggregated_signs,
                reference_model=reference_server_model
            )

            self.server_model.load_state_dict(aggregated_state)

        else:
            # Baseline, Quantization, Sparsification: FedAvg - average all model weights
            # Decompress all client models
            decompressed_models = []
            for compressed in compressed_data_list:
                state_dict = self.comm_strategy.decompress(
                    compressed,
                    reference_model=reference_server_model
                )
                decompressed_models.append(state_dict)

            # Average all state dicts
            aggregated_state = {}
            num_clients = len(decompressed_models)

            for key in decompressed_models[0].keys():
                aggregated_state[key] = decompressed_models[0][key].clone()

            for state_dict in decompressed_models[1:]:
                for key in aggregated_state.keys():
                    aggregated_state[key] += state_dict[key]

            for key in aggregated_state.keys():
                aggregated_state[key] /= num_clients

            self.server_model.load_state_dict(aggregated_state)

    def train(self):
        history = {}

        print(f"Starting federated training with {len(self.clients)} clients")
        print(f"Total users: {self.num_users}, Items: {self.num_items}")
        print(f"Configuration: {self.config.aggregation_epochs} aggregation epochs, "
              f"{self.config.local_epochs} local epochs per client")
        print(f"Communication method: {self.config.communication_method}\n")

        for epoch in range(self.config.aggregation_epochs):
            # Reset communication tracker for this round
            self.comm_tracker.reset()

            # Load server model from previous round (used as reference for compression)
            self.server_model = self._load_server_model(epoch)
            reference_server_model = NCFServer(
                num_items=self.num_items,
                hidden_dim=self.config.latent_dim
            ).to(self.device)
            reference_server_model.load_state_dict(self.server_model.state_dict())

            # Distribute server weights to clients
            self._distribute_server_weights()

            # Clients train locally
            results = self._train_clients(aggregation_epoch=epoch)

            # Extract and compress server models from clients
            self._extract_server_models(reference_server_model=reference_server_model)

            # Aggregate compressed models
            self._aggregate_models(reference_server_model=reference_server_model)

            # Save updated server model
            self._save_server_model(epoch + 1)

            # Add communication stats to results
            comm_stats = self.comm_tracker.get_stats()
            results['communication'] = comm_stats
            history[epoch] = results

        print("\nFederated training complete!")
        return history


def main():
    from dataloader import MovieLensDataset
    import sys

    print("Loading MovieLens dataset...")
    dataset = MovieLensDataset()

    comm_method = sys.argv[1] if len(sys.argv) > 1 else "baseline"

    config = FederatedConfig(
        num_clients=10,
        user_per_client_range=(1, 10),
        aggregation_epochs=10,
        local_epochs=10,
        batch_size=128,
        latent_dim=32,
        learning_rate=5e-4,
        seed=0,
        communication_method=comm_method  # Options: "baseline", "signsgd"
    )

    fed_ncf = FederatedNCF(ui_matrix=dataset.ratings, config=config)
    history = fed_ncf.train()

    final_epoch = max(history.keys())
    final_results = history[final_epoch]
    print(f"\nFinal Results (Epoch {final_epoch}):")
    for metric, values in final_results.items():
        if metric == 'communication':
            print(f"\n  Communication Stats (Epoch {final_epoch}):")
            for comm_metric, comm_value in values.items():
                print(f"    {comm_metric}: {comm_value}")
        else:
            avg_value = sum(values) / len(values)
            print(f"  {metric}: {avg_value:.4f}")

    total_bytes_all_epochs = sum(
        history[epoch]['communication']['total_bytes']
        for epoch in history.keys()
    )
    print(f"\n  Total Communication (All Epochs):")
    print(f"    total_bytes: {total_bytes_all_epochs}")
    print(f"    total_mb: {round(total_bytes_all_epochs / (1024 * 1024), 4)}")


if __name__ == "__main__":
    main()
