import torch
from evo2 import Evo2
from seq2expression.Evo2.nn.mlp import MLPHead


class Evo2ExpressionPredictor:
    def __init__(
        self,
        head_path: str,
        evo2_model: str = "evo2_7b",
        layer_name: str = "blocks.28.mlp.l3",
        device: str     = "cuda",
    ):
        self.layer_name = layer_name
        self.device     = device

        self.evo2 = Evo2(evo2_model)
        self.evo2.model.eval()
        for p in self.evo2.model.parameters():
            p.requires_grad_(False)

        state   = torch.load(head_path, map_location=device)
        in_dim  = state["net.0.weight"].shape[1]
        out_dim = state["net.3.weight"].shape[0]

        self.head = MLPHead(in_dim=in_dim, out_dim=out_dim, dropout=0.0).to(device)
        self.head.load_state_dict(state)
        self.head.eval()

    def __call__(self, sequence: str) -> torch.Tensor:
        input_ids = torch.tensor(
            self.evo2.tokenizer.tokenize(sequence), dtype=torch.int
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, embeddings = self.evo2(input_ids, return_embeddings=True, layer_names=[self.layer_name])
            emb = embeddings[self.layer_name].squeeze(0).mean(dim=0).float()
            return self.head(emb)