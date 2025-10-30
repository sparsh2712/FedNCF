import random
import torch
import numpy as np
from tqdm import tqdm

from client import LocalTrainer
from server import NCFServer
from config import FederatedConfig


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

    def _extract_server_models(self):
        for client_id in range(len(self.clients)):
            client_path = self.config.local_dir / f"dp{client_id}.pt"
            client_model = torch.jit.load(str(client_path))

            server_model = NCFServer(
                num_items=self.num_items,
                hidden_dim=self.config.latent_dim
            )
            server_model.set_weights_from(client_model)

            server_path = self.config.local_items_dir / f"dp{client_id}.pt"
            scripted_model = torch.jit.script(server_model.cpu())
            torch.jit.save(scripted_model, str(server_path))

    def _aggregate_models(self):
        client_models = []

        for client_id in range(len(self.clients)):
            path = self.config.local_items_dir / f"dp{client_id}.pt"
            model = torch.jit.load(str(path))
            client_models.append(model)

        if not client_models:
            return

        aggregated_state = {}
        num_clients = len(client_models)

        for key, value in client_models[0].state_dict().items():
            aggregated_state[key] = value.clone()

        for client_model in client_models[1:]:
            client_state = client_model.state_dict()
            for key in aggregated_state.keys():
                aggregated_state[key] += client_state[key]

        for key in aggregated_state.keys():
            aggregated_state[key] /= num_clients

        self.server_model.load_state_dict(aggregated_state)

    def train(self):
        history = {}

        print(f"Starting federated training with {len(self.clients)} clients")
        print(f"Total users: {self.num_users}, Items: {self.num_items}")
        print(f"Configuration: {self.config.aggregation_epochs} aggregation epochs, "
              f"{self.config.local_epochs} local epochs per client\n")

        for epoch in range(self.config.aggregation_epochs):
            self.server_model = self._load_server_model(epoch)

            self._distribute_server_weights()

            results = self._train_clients(aggregation_epoch=epoch)
            history[epoch] = results

            self._extract_server_models()

            self._aggregate_models()

            self._save_server_model(epoch + 1)

        print("\nFederated training complete!")
        return history


def main():
    from dataloader import MovieLensDataset

    print("Loading MovieLens dataset...")
    dataset = MovieLensDataset()

    config = FederatedConfig(
        num_clients=10,
        user_per_client_range=(1, 10),
        aggregation_epochs=10,
        local_epochs=10,
        batch_size=128,
        latent_dim=32,
        learning_rate=5e-4,
        seed=0
    )

    fed_ncf = FederatedNCF(ui_matrix=dataset.ratings, config=config)
    history = fed_ncf.train()

    final_epoch = max(history.keys())
    final_results = history[final_epoch]
    print(f"\nFinal Results (Epoch {final_epoch}):")
    for metric, values in final_results.items():
        avg_value = sum(values) / len(values)
        print(f"  {metric}: {avg_value:.4f}")


if __name__ == "__main__":
    main()
