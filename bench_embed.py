import time
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

df = pd.read_parquet("data/processed/chunks_B.parquet", columns=["chunk_id", "text", "n_tokens"])
sample = df.sample(2000, random_state=0)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = SentenceTransformer("BAAI/bge-small-en-v1.5", device=dev)
print("device:", dev, "| max_seq_length:", m.max_seq_length, "| dim:", m.get_sentence_embedding_dimension())
assert m.max_seq_length >= 512, "embedder would silently truncate B chunks"

t = time.time()
E = m.encode(sample["text"].tolist(), batch_size=64, normalize_embeddings=True,
             convert_to_numpy=True, show_progress_bar=False)
dt = time.time() - t
n = np.linalg.norm(E, axis=1)
print(f"{len(sample) / dt:.1f} chunks/s | {E.shape} {E.dtype} | norms {n.min():.4f}-{n.max():.4f} | NaN: {np.isnan(E).any()}")