import re
from pathlib import Path
from collections import defaultdict

import streamlit as st
import torch
import timm
from PIL import Image
import numpy as np

IMAGE_DIR = Path("../data/raw/Preston Montford 2026")
MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-L-384"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

id_pattern = re.compile(r"(GCN\d+)")

@st.cache_resource
def load_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0).to(DEVICE).eval()
    cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**cfg)
    return model, transform

@st.cache_data
def compute_embeddings():
    model, transform = load_model()
    filenames = sorted(f.name for f in IMAGE_DIR.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = [id_pattern.search(f).group(1) for f in filenames]

    embeddings = []
    with torch.no_grad():
        for fname in filenames:
            img = Image.open(IMAGE_DIR / fname).convert("RGB")
            x = transform(img).unsqueeze(0).to(DEVICE)
            emb = model(x).squeeze(0).cpu().numpy()
            embeddings.append(emb / np.linalg.norm(emb))

    return np.array(filenames), np.array(labels), np.stack(embeddings)

def top_matches_for_newt(query_id, filenames, labels, sim_matrix, k):
    query_idx = np.where(labels == query_id)[0]
    other_idx = np.where(labels != query_id)[0]

    sub_sim = sim_matrix[np.ix_(query_idx, other_idx)]
    best_per_other_img = sub_sim.max(axis=0)

    scores_by_newt, best_img_by_newt = defaultdict(float), {}
    for score, idx in zip(best_per_other_img, other_idx):
        newt_id = labels[idx]
        if score > scores_by_newt[newt_id]:
            scores_by_newt[newt_id] = score
            best_img_by_newt[newt_id] = filenames[idx]

    ranked = sorted(scores_by_newt.items(), key=lambda x: -x[1])[:k]
    return [(newt_id, score, best_img_by_newt[newt_id]) for newt_id, score in ranked]

# --- App ---
st.title("Newt Re-Identification: Top Matches")

filenames, labels, embeddings = compute_embeddings()
sim_matrix = embeddings @ embeddings.T
unique_ids = sorted(set(labels))

col1, col2 = st.columns(2)
query_id = col1.selectbox("Select newt:", unique_ids)
top_k = col2.selectbox("Number of matches:", [5, 10])

query_img = filenames[np.where(labels == query_id)[0][0]]
st.image(str(IMAGE_DIR / query_img), caption=f"Query: {query_id}", width=200)

st.subheader(f"Top {top_k} most similar newts")
matches = top_matches_for_newt(query_id, filenames, labels, sim_matrix, top_k)

n_cols = 5
rows = [matches[i:i + n_cols] for i in range(0, len(matches), n_cols)]
for row in rows:
    cols = st.columns(n_cols)
    for col, (newt_id, score, best_img) in zip(cols, row):
        with col:
            st.image(str(IMAGE_DIR / best_img), width=150)
            st.caption(f"{newt_id} ({score:.3f})")