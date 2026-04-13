import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ID      = os.environ.get("GOOGLE_CLOUD_PROJECT", "cmpe188-hw1")
DATASET         = "sentiment_analysis"
EMBED_MODEL     = f"{PROJECT_ID}.{DATASET}.embedding_model"
BQ_LOCATION     = "US"

SAMPLE_SIZE     = 200
EMBED_DIM       = 768     # text-embedding-004 output dimension
HIDDEN_DIM      = 256
BATCH_SIZE      = 64
EPOCHS          = 20
LR              = 1e-3
VALIDATION_SPLIT= 0.2

TARGET_VAL_ACC  = 0.80
TARGET_F1       = 0.80

LOG_INTERVAL=1

OUTPUT_DIR      = "output/bq_movie_ratings"


import sys

def eprint(*args, **kwargs):
    print(*args, **kwargs)
    print(*args, file=sys.stderr, **kwargs)

def load_data_from_bigquery():
    import bigframes.pandas as bpd

    bpd.options.bigquery.project = PROJECT_ID
    bpd.options.bigquery.location = BQ_LOCATION

    print("Loading IMDB reviews from BigQuery public dataset …")

    # --- 1. Load raw text + labels using Bigframes ---
    query = f"""
        SELECT
            review AS content,
            CASE WHEN label = 'pos' THEN 1 ELSE 0 END AS label
        FROM `bigquery-public-data.imdb.reviews`
        WHERE review IS NOT NULL
        LIMIT {SAMPLE_SIZE}
    """
    df_bq = bpd.read_gbq(query)
    print(f"  Loaded {len(df_bq)} rows from BigQuery.")
    
    df_bq.to_gbq(f"{PROJECT_ID}.{DATASET}.imdb_sample", if_exists="replace")

    embed_sql = f"""
        SELECT
            base.label,
            emb.ml_generate_embedding_result AS embedding
        FROM ML.GENERATE_EMBEDDING(
            MODEL `{EMBED_MODEL}`,
            TABLE `{PROJECT_ID}.{DATASET}.imdb_sample`,
            STRUCT(TRUE AS flatten_json_output)
        ) AS emb
        JOIN `{PROJECT_ID}.{DATASET}.imdb_sample` AS base
            ON emb.content = base.content
    """
    df_embed = bpd.read_gbq(embed_sql).to_pandas()
    print(f"  Embeddings ready — shape: {df_embed.shape}")
    return df_embed


def prepare_tensors(df):
    X = np.stack(df["embedding"].values).astype(np.float32)
    y = df["label"].values.astype(np.int64)
    return torch.tensor(X), torch.tensor(y)


def make_dataloaders(X, y, val_split=0.2, batch_size=64):
    n = len(X)
    val_n = int(n * val_split)
    idx = torch.randperm(n)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   y[val_idx])

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))

class SentimentMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)

def train(model, train_loader, val_loader, epochs=20, lr=1e-3):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        total = 0
        epoch_loss = 0.0
        for (X_batch, y_batch) in tqdm(train_loader):
            # Move to device
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            total += len(y_batch)
        
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total = 0
        total_correct = 0
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch).item()
                predictions = torch.argmax(outputs, dim=1)
                total_correct += (predictions == y_batch).sum().item()
                total += len(y_batch)
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        # Print progress every 20 epochs
        if (epoch + 1) % LOG_INTERVAL == 0:
            eprint(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {100*total_correct/total:.2f}%")

    return train_losses, val_losses


def evaluate(model, loader):
    device = next(model.parameters()).device
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(dim=1).cpu()
            all_preds.append(preds)
            all_targets.append(y_batch)
    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()
    return {
        "accuracy": accuracy_score(targets, preds),
        "f1_macro": f1_score(targets, preds, average="macro"),
    }


def save_artifacts(model, train_losses, val_losses, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))
    with open(os.path.join(output_dir, "history.json"), "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)
        
    from matplotlib import pyplot as plt
    # Create figure
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))
    
    # Plot 2: Loss curves
    ax1.plot(train_losses, 'b-', linewidth=2, label='Train Loss')
    ax1.plot(val_losses, 'r--', linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training History')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Artifacts saved to {output_dir}")


def main():
    print("=" * 60)
    print("BigQuery Bigframes + Embeddings → MLP Sentiment Classifier")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    df = load_data_from_bigquery()
    X, y = prepare_tensors(df)
    print(f"Tensor shapes — X: {X.shape}, y: {y.shape}")

    train_loader, val_loader = make_dataloaders(X, y,
                                                val_split=VALIDATION_SPLIT,
                                                batch_size=BATCH_SIZE)

    model = SentimentMLP(input_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, num_classes=2)
    print(f"Model: {model}")

    train_losses, val_losses = train(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)

    train_metrics = evaluate(model, train_loader)
    val_metrics   = evaluate(model, val_loader)
    print("\nResults:")
    print(f"  Train — Acc: {train_metrics['accuracy']:.4f}  F1: {train_metrics['f1_macro']:.4f}")
    print(f"  Val   — Acc: {val_metrics['accuracy']:.4f}  F1: {val_metrics['f1_macro']:.4f}")

    save_artifacts(model, train_losses, val_losses, OUTPUT_DIR)

    print("\n" + "=" * 60)
    checks_passed = True
    if val_metrics['accuracy'] > TARGET_VAL_ACC:
        eprint(f"✓ Val accuracy > {TARGET_VAL_ACC}: {val_metrics['accuracy']:.4f}")
    else:
        eprint(f"✗ Val accuracy > {TARGET_VAL_ACC}: {val_metrics['accuracy']:.4f}")
        checks_passed = False
    
    if val_metrics['f1_macro'] > TARGET_F1:
        eprint(f"✓ Val f1_macro > {TARGET_F1}: {val_metrics['f1_macro']:.4f}")
    else:
        eprint(f"✗ Val f1_macro > {TARGET_F1}: {val_metrics['f1_macro']:.4f}")
        checks_passed = False
        
    print("PASS" if checks_passed else "FAIL")
    return 0 if checks_passed else 1


if __name__ == "__main__":
    exit(main())