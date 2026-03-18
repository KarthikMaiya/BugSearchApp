import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import os

from embedding_text import build_embedding_text

# -----------------------------
# 1. Configuration
# -----------------------------
DATA_PATH = "data/bugs_semantic.xlsx"
EMBEDDINGS_DIR = "embeddings"
MODEL_NAME = "all-MiniLM-L6-v2"

# Create embeddings folder if it does not exist
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# -----------------------------
# 2. Load Excel data
# -----------------------------
print("📥 Loading Excel data...")
df = pd.read_excel(DATA_PATH, header=0)

# -----------------------------
# 3. Validate required columns
# -----------------------------
required_columns = [
    "WorkItemId",
    "Title",
    "SemanticText"
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"❌ Missing required column: {col}")

print("✅ Required columns found")

# -----------------------------
# 4. Prepare text for embedding
# -----------------------------

# IMPORTANT:
# Embeddings and metadata must have the exact same row order and count.
# We therefore filter invalid rows ONCE here, and use the same filtered
# DataFrame for both embedding generation and metadata export.
df = df.dropna(subset=["WorkItemId", "Title"]).copy()
df["WorkItemId"] = df["WorkItemId"].astype("int64")

# Normalize for robust combined embedding text.
df["Title"] = df["Title"].fillna("").astype(str).str.strip()
df["SemanticText"] = df["SemanticText"].fillna("").astype(str).str.strip()
df.loc[df["SemanticText"] == "", "SemanticText"] = df.loc[df["SemanticText"] == "", "Title"]
df = df[df["SemanticText"].astype(str).str.strip() != ""].copy()

# Build rich embedding inputs.
texts = []
for row in df.itertuples(index=False):
    wid = getattr(row, "WorkItemId", "")
    combined_text = build_embedding_text(row)
    print(f"[EMBED] WorkItemId {wid} | Text length: {len(combined_text)}")
    texts.append(combined_text)

print(f"🧠 Total bugs to embed: {len(texts)}")

# -----------------------------
# 5. Load sentence transformer model
# -----------------------------
print("📦 Loading SentenceTransformer model...")
model = SentenceTransformer(MODEL_NAME)

# -----------------------------
# 6. Generate embeddings
# -----------------------------
print("⚙️ Generating embeddings (this may take a few minutes)...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

# -----------------------------
# 7. Save embeddings
# -----------------------------
embeddings_path = os.path.join(EMBEDDINGS_DIR, "bug_embeddings.npy")
np.save(embeddings_path, embeddings)

print(f"💾 Embeddings saved to {embeddings_path}")

# -----------------------------
# 8. Save metadata
# -----------------------------

# Keep metadata aligned 1:1 with embeddings.
cols = ["WorkItemId", "Title"]
if "Link" in df.columns:
    cols.append("Link")
elif "link" in df.columns:
    cols.append("link")

metadata = df[cols].copy()
metadata_path = os.path.join(EMBEDDINGS_DIR, "bug_metadata.csv")
metadata.to_csv(metadata_path, index=False)

print(f"💾 Metadata saved to {metadata_path}")

# -----------------------------
# 9. Done
# -----------------------------
print("✅ Embedding generation completed successfully!")
