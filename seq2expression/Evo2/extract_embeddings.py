from pathlib import Path
import torch
from evo2 import Evo2
from seq2expression.utils.datasets import pgb_gene_exp

MODEL_PATH = "evo2_7b"
# MODEL_PATH = "evo2_40b"
LAYER_NAME = "blocks.28.mlp.l3"
EMBED_DIR  = Path("shared/evo2_gene_exp/embeddings") / MODEL_PATH
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

SPECIES = [
    # "glycine_max",
    # "oryza_sativa",
    "solanum_lycopersicum",
    "zea_mays",
    "arabidopsis_thaliana",
]

def extract(species: str, split: str, evo2_model, dataset):
    out_dir = EMBED_DIR / species
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{split}.pt"
    if out_path.exists():
        print(f"[skip] {species}/{split} already exists")
        return

    data = dataset[species][split]
    all_X, all_y = [], []

    for i, sample in enumerate(data):
        input_ids = torch.tensor(
            evo2_model.tokenizer.tokenize(sample["sequence"]),
            dtype=torch.int,
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            _logits, embeddings = evo2_model(
                input_ids,
                return_embeddings=True,
                layer_names=[LAYER_NAME],
            )

        emb = embeddings[LAYER_NAME].squeeze(0).mean(dim=0).cpu()
        all_X.append(emb)
        all_y.append(torch.tensor(sample["labels"], dtype=torch.float32))

        if (i + 1) % 500 == 0:
            print(f"  {species}/{split}: {i + 1}/{len(data)}")

    torch.save({"X": torch.stack(all_X), "y": torch.stack(all_y)}, out_path)
    print(f"[saved] {out_path}  ({len(all_X)} samples)")


if __name__ == "__main__":
    print(f"Loading Evo2 ({MODEL_PATH}) …")
    evo2_model = Evo2(MODEL_PATH)
    inner = evo2_model.model
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)

    dataset = pgb_gene_exp()

    for species in SPECIES:
        for split in ("train", "validation", "test"):
            extract(species, split, evo2_model, dataset)