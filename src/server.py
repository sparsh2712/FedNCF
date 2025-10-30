import torch

class NCFServer(torch.nn.Module):
    def __init__(self, num_items, hidden_dim=32):
        super().__init__()
        self.mlp_item_embed = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=2 * hidden_dim)
        self.gmf_item_embed = torch.nn.Embedding(num_embeddings=num_items, embedding_dim=2 * hidden_dim)

        self.mlp_network = torch.nn.Sequential(
            torch.nn.Linear(4 * hidden_dim, 2 * hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU()
        )

        self.gmf_head = torch.nn.Linear(2 * hidden_dim, 1)
        self.gmf_head.weight = torch.nn.Parameter(torch.ones(1, 2 * hidden_dim))

        self.mlp_head = torch.nn.Linear(hidden_dim // 2, 1)
        self.output_head = torch.nn.Linear(hidden_dim, 1)

        self.alpha = 0.5  # model blending parameter
        self._init_weights()
        self._combine_output_weights()

    def _init_weights(self):
        torch.nn.init.normal_(self.mlp_item_embed.weight, std=0.01)
        torch.nn.init.normal_(self.gmf_item_embed.weight, std=0.01)
        for layer in self.mlp_network:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
        torch.nn.init.kaiming_uniform_(self.gmf_head.weight, a=1)
        torch.nn.init.kaiming_uniform_(self.mlp_head.weight, a=1)

    def _copy_layers(self, src, dest):
        for src_param, dest_param in zip(src.parameters(), dest.parameters()):
            dest_param.data.copy_(src_param.data)

    def set_weights_from(self, model):
        self._copy_layers(model.mlp_item_embed, self.mlp_item_embed)
        self._copy_layers(model.gmf_item_embed, self.gmf_item_embed)
        self._copy_layers(model.mlp_network, self.mlp_network)
        self._copy_layers(model.gmf_head, self.gmf_head)
        self._copy_layers(model.mlp_head, self.mlp_head)
        self._copy_layers(model.output_head, self.output_head)

    def forward(self):
        return torch.tensor(0.0)

    def _combine_output_weights(self):
        blended_weights = torch.nn.Parameter(
            torch.cat(
                (
                    self.alpha * self.gmf_head.weight,
                    (1 - self.alpha) * self.mlp_head.weight
                ),
                dim=1
            )
        )
        self.output_head.weight = blended_weights


if __name__ == '__main__':
    model = NCFServer(num_items=100, hidden_dim=64)
    print(model)
