import numpy as np
import torch
import torch.nn as nn
from metrics import compute_metrics


class BatchSampler:
    def __init__(self, ui_matrix, seed=0):
        np.random.seed(seed)
        self.ui_matrix = ui_matrix
        self.positive_pairs = np.argwhere(ui_matrix != 0)
        self.negative_pairs = np.argwhere(ui_matrix == 0)

    def reset(self):
        self.positive_pairs = np.argwhere(self.ui_matrix != 0)
        self.negative_pairs = np.argwhere(self.ui_matrix == 0)

    def sample_batch(self, batch_size):
        pos_size = batch_size // 4
        neg_size = batch_size - pos_size

        if (self.positive_pairs.shape[0] < pos_size or
            self.negative_pairs.shape[0] < neg_size):
            return torch.tensor([[0, 0]]), torch.tensor([0.0])

        try:
            pos_indices = np.random.choice(self.positive_pairs.shape[0], pos_size, replace=False)
            neg_indices = np.random.choice(self.negative_pairs.shape[0], neg_size, replace=False)

            pos_pairs = self.positive_pairs[pos_indices]
            neg_pairs = self.negative_pairs[neg_indices]

            self.positive_pairs = np.delete(self.positive_pairs, pos_indices, axis=0)
            self.negative_pairs = np.delete(self.negative_pairs, neg_indices, axis=0)

            batch_pairs = np.concatenate([pos_pairs, neg_pairs], axis=0)
            np.random.shuffle(batch_pairs)

            ratings = np.array([
                self.ui_matrix[user_id, item_id]
                for user_id, item_id in batch_pairs
            ])

            return torch.tensor(batch_pairs), torch.tensor(ratings, dtype=torch.float32)

        except Exception:
            return torch.tensor([[0, 0]]), torch.tensor([0.0])


class NCFClient(nn.Module):
    def __init__(self, num_users, num_items, latent_dim=32):
        super().__init__()
        embed_dim = 2 * latent_dim

        self.mlp_user_embed = nn.Embedding(num_users, embed_dim)
        self.mlp_item_embed = nn.Embedding(num_items, embed_dim)

        self.gmf_user_embed = nn.Embedding(num_users, embed_dim)
        self.gmf_item_embed = nn.Embedding(num_items, embed_dim)

        self.mlp_network = nn.Sequential(
            nn.Linear(4 * latent_dim, 2 * latent_dim),
            nn.ReLU(),
            nn.Linear(2 * latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU()
        )

        self.gmf_head = nn.Linear(embed_dim, 1)
        self.gmf_head.weight = nn.Parameter(torch.ones(1, embed_dim))
        self.mlp_head = nn.Linear(latent_dim // 2, 1)
        self.output_head = nn.Linear(latent_dim, 1)

        self.alpha = 0.5

        self._init_weights()
        self._combine_output_weights()

    def _init_weights(self):
        nn.init.normal_(self.mlp_user_embed.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embed.weight, std=0.01)
        nn.init.normal_(self.gmf_user_embed.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embed.weight, std=0.01)

        for layer in self.mlp_network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)

        nn.init.kaiming_uniform_(self.gmf_head.weight, a=1)
        nn.init.kaiming_uniform_(self.mlp_head.weight, a=1)

    def _combine_output_weights(self):
        blended_weights = nn.Parameter(
            torch.cat([
                self.alpha * self.gmf_head.weight,
                (1 - self.alpha) * self.mlp_head.weight
            ], dim=1)
        )
        self.output_head.weight = blended_weights

    def forward(self, user_item_pairs):
        user_ids = user_item_pairs[:, 0]
        item_ids = user_item_pairs[:, 1]

        gmf_user = self.gmf_user_embed(user_ids)
        gmf_item = self.gmf_item_embed(item_ids)
        gmf_product = torch.mul(gmf_user, gmf_item)

        mlp_user = self.mlp_user_embed(user_ids)
        mlp_item = self.mlp_item_embed(item_ids)
        mlp_concat = torch.cat([mlp_user, mlp_item], dim=1)
        mlp_output = self.mlp_network(mlp_concat)

        combined = torch.cat([gmf_product, mlp_output], dim=1)
        return self.output_head(combined).squeeze()

    def load_server_weights(self, server_model):
        self._copy_weights(server_model.mlp_item_embed, self.mlp_item_embed)
        self._copy_weights(server_model.gmf_item_embed, self.gmf_item_embed)
        self._copy_weights(server_model.mlp_network, self.mlp_network)
        self._copy_weights(server_model.gmf_head, self.gmf_head)
        self._copy_weights(server_model.mlp_head, self.mlp_head)
        self._copy_weights(server_model.output_head, self.output_head)

    @staticmethod
    def _copy_weights(source, target):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(src_param.data)


class LocalTrainer:
    def __init__(self, ui_matrix, num_items, config):
        self.ui_matrix = ui_matrix
        self.num_users = ui_matrix.shape[0]
        self.num_items = num_items
        self.config = config
        self.device = torch.device(config.device)

        self.model = NCFClient(
            num_users=self.num_users,
            num_items=num_items,
            latent_dim=config.latent_dim
        ).to(self.device)

        self.sampler = BatchSampler(ui_matrix, seed=config.seed)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
        )

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        epoch_hr = 0.0
        epoch_ndcg = 0.0
        num_batches = 0

        while True:
            pairs, ratings = self.sampler.sample_batch(self.config.batch_size)

            if pairs.shape[0] < self.config.batch_size:
                break

            pairs = pairs.int().to(self.device)
            ratings = ratings.float().to(self.device)

            predictions = self.model(pairs)

            mask = (ratings > 0).float()
            loss = nn.functional.mse_loss(predictions * mask, ratings)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            hr, ndcg = compute_metrics(ratings.cpu().numpy(), predictions.detach().cpu().numpy())
            epoch_loss += loss.item()
            epoch_hr += hr
            epoch_ndcg += ndcg
            num_batches += 1

        self.sampler.reset()

        return {
            "loss": epoch_loss / max(num_batches, 1),
            "hit_ratio@10": epoch_hr / max(num_batches, 1),
            "ndcg@10": epoch_ndcg / max(num_batches, 1),
            "num_batches": num_batches
        }

    def train(self):
        for epoch in range(self.config.local_epochs):
            metrics = self.train_epoch()

        self.model._combine_output_weights()

        return {
            "num_users": self.num_users,
            "loss": metrics["loss"],
            "hit_ratio": metrics["hit_ratio@10"],
            "ndcg": metrics["ndcg@10"]
        }


if __name__ == '__main__':
    ui_matrix = np.random.randint(0, 6, size=(10, 20))
    model = NCFClient(num_users=10, num_items=20, latent_dim=16)
    print(model)
