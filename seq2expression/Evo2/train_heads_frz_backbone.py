from pathlib import Path
from datetime import datetime
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from seq2expression.Evo2.nn.mlp import MLPHead

SPECIES = [
    # "glycine_max",
    "oryza_sativa",
    "solanum_lycopersicum",
    # "zea_mays",
    # "arabidopsis_thaliana",
]
LAYER_NAME = "blocks.28.mlp.l3"
MODEL_PATH = "evo2_7b"
EMBED_DIR  = Path("shared/evo2_gene_exp/embeddings") / MODEL_PATH

DROPOUT      = 0.1
HIDDEN_DIM   = 1024
BATCH_SIZE   = 256
LR           = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 400
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

RUN_DIR = Path("shared/evo2_gene_exp/training_runs") / f"{MODEL_PATH}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"


def load_split(species: str, split: str) -> TensorDataset:
    data = torch.load(EMBED_DIR / species / f"{split}.pt")
    return TensorDataset(data["X"].float(), data["y"].float())


def train_species(species: str):
    # autodetect input / output dimensions of the MLP
    _data   = torch.load(EMBED_DIR / species / "train.pt")
    emb_dim = _data["X"].shape[1]
    out_dim = _data["y"].shape[1]

    run_dir = RUN_DIR / species
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb.init(project="seq2expression", name=f"{species}_{MODEL_PATH}", config={
        "species": species, "layer": LAYER_NAME, "emb_dim": emb_dim,
        "out_dim": out_dim, "dropout": DROPOUT, "hidden_dim": HIDDEN_DIM,
        "batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY, "epochs": EPOCHS,
    })

    train_ds, val_ds = load_split(species, "train"), load_split(species, "validation")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model     = MLPHead(in_dim=emb_dim, dropout=DROPOUT, out_dim=out_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ckpt_path = run_dir / "best.pt"
    best_val  = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(X), y).item() * len(X)
        val_loss /= len(val_ds)

        scheduler.step()
        print(f"[{species}] Epoch {epoch:3d}/{EPOCHS}  train={train_loss:.4f}  val={val_loss:.4f}")
        wandb.log({"train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ {val_loss:.4f}")

    wandb.finish()


if __name__ == "__main__":
    for species in SPECIES:
        train_species(species)
