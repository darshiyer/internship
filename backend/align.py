"""
Sentence alignment utilities using multilingual sentence embeddings.
Supports multilingual-e5-base and LaBSE for alignment experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def load_embedder(model_name: str = "intfloat/multilingual-e5-base") -> SentenceTransformer:
    return SentenceTransformer(model_name)


def embed_sentences(sentences: list[str], model: SentenceTransformer, is_query: bool = False) -> np.ndarray:
    if "multilingual-e5" in model._first_module().auto_model.config.name_or_path:
        prefix = "query: " if is_query else "passage: "
        sentences = [f"{prefix}{s}" for s in sentences]
    return model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)


def align_sentences(
    source_sentences: list[str],
    target_sentences: list[str],
    model_name: str = "intfloat/multilingual-e5-base",
    top_k: int = 1,
) -> pd.DataFrame:
    model = load_embedder(model_name)
    source_emb = embed_sentences(source_sentences, model, is_query=True)
    target_emb = embed_sentences(target_sentences, model, is_query=False)

    similarity = util.cos_sim(source_emb, target_emb).cpu().numpy()

    rows = []
    for src_idx, src_sent in enumerate(source_sentences):
        top_indices = np.argsort(similarity[src_idx])[::-1][:top_k]
        for rank, tgt_idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "source_index": src_idx,
                    "target_index": int(tgt_idx),
                    "source_sentence": src_sent,
                    "target_sentence": target_sentences[int(tgt_idx)],
                    "similarity": float(similarity[src_idx][int(tgt_idx)]),
                    "rank": rank,
                }
            )

    return pd.DataFrame(rows)
