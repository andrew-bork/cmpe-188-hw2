import os
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm


PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "cmpe188-hw1")
DATASET = "stackoverflow"
LLM_MODEL = f"{PROJECT_ID}.{DATASET}.llm_model"
BQ_LOCATION = "US"

SAMPLE_SIZE = 500
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
VALIDATION_SPLIT = 0.2

TARGET_VAL_ACC = 0.65
TARGET_F1 = 0.65

LOG_INTERVAL = 1

OUTPUT_DIR = "output/bq_stackoverflow"


def load_and_score_with_llm():
    import bigframes.pandas as bpd

    bpd.options.bigquery.project = PROJECT_ID
    bpd.options.bigquery.location = BQ_LOCATION

    print("Loading Stack Overflow questions from BigQuery public dataset ...")
    query = f"""
        SELECT
            body AS content,
            score,
            CASE WHEN score > 0 THEN 1 ELSE 0 END AS label
        FROM `bigquery-public-data.stackoverflow.posts_questions`
        WHERE body IS NOT NULL
          AND LENGTH(body) > 100
        LIMIT {SAMPLE_SIZE}
    """
    df_bq = bpd.read_gbq(query)
    print(f"  Loaded {len(df_bq)} rows.")
    # save to table
    df_bq.to_gbq(f"{PROJECT_ID}.{DATASET}.stackoverflow_sample", if_exists="replace")

    print("Scoring questions with BigQuery ML LLM (Gemini) ...")

    llm_sql = f"""
        SELECT
            base.content,
            base.score,
            base.label,
            llm.ml_generate_text_llm_result AS llm_response
        FROM ML.GENERATE_TEXT(
            MODEL `{LLM_MODEL}`,
            (
                SELECT
                    CONCAT(
                        'Rate the clarity of this Stack Overflow question from 0 to 10. ',
                        'Reply with ONLY a single integer and nothing else. ',
                        'Question: ', SUBSTR(content, 1, 500)
                    ) AS prompt,
                    content,
                    score,
                    label
                FROM `{PROJECT_ID}.{DATASET}.stackoverflow_sample`
            ),
            STRUCT(
                0.0  AS temperature,
                5    AS max_output_tokens,
                TRUE AS flatten_json_output
            )
        ) AS llm
        JOIN `{PROJECT_ID}.{DATASET}.stackoverflow_sample` AS base
            ON llm.content = base.content
    """

    df_scored = bpd.read_gbq(llm_sql).to_pandas()
    print(f"  LLM scoring done — {len(df_scored)} rows.")

    # get score
    def parse_score(text):
        m = re.search(r"\d+", str(text))
        return int(m.group()) if m else 5

    df_scored["llm_score"] = df_scored["llm_response"].apply(parse_score).clip(0, 10)

    print(f"\nSample LLM scores:")
    print(df_scored[["score", "label", "llm_score"]].head(10))

    return df_scored


def build_features(df):
    
    # llm_score
    # word_count
    # has_code - has <code> block
    # has_question_mark 
    # raw_score - stasck overflow score
    df = df.copy()
    df["word_count"]        = df["content"].str.split().str.len().fillna(0)
    df["has_code"]          = df["content"].str.contains("<code>").astype(float)
    df["has_question_mark"] = df["content"].str.contains(r"\?").astype(float)
    df["raw_score"]         = df["score"].clip(-10, 50).astype(float)

    feature_cols = ["llm_score", "word_count", "has_code",
                    "has_question_mark", "raw_score"]

    X = df[feature_cols].fillna(0).values.astype(np.float32)
    y = df["label"].values.astype(np.int64)

    # normalize
    mean = X.mean(0)
    std  = X.std(0) + 1e-8
    X    = (X - mean) / std

    return torch.tensor(X), torch.tensor(y)


def make_dataloaders(X, y, val_split=0.2, batch_size=32):
    n     = len(X)
    val_n = int(n * val_split)
    idx   = torch.randperm(n)
    val_idx, train_idx = idx[:val_n], idx[val_n:]

    train_ds = TensorDataset(X[train_idx], y[train_idx])
    val_ds   = TensorDataset(X[val_idx],   y[val_idx])

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds,   batch_size=batch_size, shuffle=False))


class QuestionQualityMLP(nn.Module):
    def __init__(self, input_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
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
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {100*total_correct/total:.2f}%")

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
    print("BigQuery Bigframes + LLM Scoring → MLP Question Quality Classifier")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load Stack Overflow data and score with LLM via BigQuery
    df = load_and_score_with_llm()

    # 2. Build feature matrix
    X, y = build_features(df)
    print(f"\nFeature matrix — X: {X.shape}, y: {y.shape}")
    print(f"Label distribution — positive: {y.sum().item()}, "
          f"negative: {(y == 0).sum().item()}")

    # 3. DataLoaders
    train_loader, val_loader = make_dataloaders(X, y,
                                                val_split=VALIDATION_SPLIT,
                                                batch_size=BATCH_SIZE)

    # 4. Build model
    model = QuestionQualityMLP(input_dim=X.shape[1])
    print(f"\nModel: {model}")

    # 5. Train
    train_losses, val_losses = train(model, train_loader, val_loader,
                                            epochs=EPOCHS, lr=LR)

    # 6. Evaluate
    train_metrics = evaluate(model, train_loader)
    val_metrics   = evaluate(model, val_loader)
    print("\nResults:")
    print(f"  Train — Acc: {train_metrics['accuracy']:.4f}  "
          f"F1: {train_metrics['f1_macro']:.4f}")
    print(f"  Val   — Acc: {val_metrics['accuracy']:.4f}  "
          f"F1: {val_metrics['f1_macro']:.4f}")

    # 7. Save
    save_artifacts(model, train_losses, val_losses, OUTPUT_DIR)

    # 8. Quality checks
    print("\n" + "=" * 60)
    checks_passed = True
    if val_metrics['accuracy'] > TARGET_VAL_ACC:
        print(f"✓ Val accuracy > {TARGET_VAL_ACC}: {val_metrics['accuracy']:.4f}")
    else:
        print(f"✗ Val accuracy > {TARGET_VAL_ACC}: {val_metrics['accuracy']:.4f}")
        checks_passed = False
    
    if val_metrics['f1_macro'] > TARGET_F1:
        print(f"✓ Val f1_macro > {TARGET_F1}: {val_metrics['f1_macro']:.4f}")
    else:
        print(f"✗ Val f1_macro > {TARGET_F1}: {val_metrics['f1_macro']:.4f}")
        checks_passed = False
        
    print("PASS" if checks_passed else "FAIL")
    return 0 if checks_passed else 1


if __name__ == "__main__":
    exit(main())