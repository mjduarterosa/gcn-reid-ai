import pickle
from pathlib import Path
from collections import defaultdict

import streamlit as st
import numpy as np

st.set_page_config(layout="wide")

TOP_K_OPTIONS = [5, 10]
DATA_PATH = Path(__file__).parent / "newt_data.pkl"

@st.cache_data
def load_data():
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)

data = load_data()
filenames = data["filenames"]
labels = data["labels"]
dates = data["dates"]
embeddings = data["embeddings"]
thumbnails = data["thumbnails"]

sim_matrix = embeddings @ embeddings.T
unique_ids = sorted(set(labels))

def top_matches_for_newt(query_id, k):
    query_idx = np.where(labels == query_id)[0]
    other_idx = np.where(labels != query_id)[0]

    sub_sim = sim_matrix[np.ix_(query_idx, other_idx)]
    best_per_other_img = sub_sim.max(axis=0)

    scores_by_newt, best_idx_by_newt = defaultdict(float), {}
    for score, idx in zip(best_per_other_img, other_idx):
        newt_id = labels[idx]
        if score > scores_by_newt[newt_id]:
            scores_by_newt[newt_id] = score
            best_idx_by_newt[newt_id] = idx

    ranked = sorted(scores_by_newt.items(), key=lambda x: -x[1])[:k]
    return [(newt_id, score, best_idx_by_newt[newt_id]) for newt_id, score in ranked]

st.title("Newt Re-Identification: Top Matches")

left_col, right_col = st.columns([1, 3])

with left_col:
    dropdown_options = [f"{nid} ({dates[np.where(labels == nid)[0][0]]})" for nid in unique_ids]
    selected = st.selectbox("Select newt:", dropdown_options)
    top_k = st.selectbox("Number of matches:", TOP_K_OPTIONS)
    query_id = unique_ids[dropdown_options.index(selected)]

    query_idx = np.where(labels == query_id)[0][0]
    st.image(thumbnails[query_idx], caption=f"Query: {query_id} ({dates[query_idx]})", width=200)

with right_col:
    st.subheader(f"Top {top_k} most similar newts")
    matches = top_matches_for_newt(query_id, top_k)

    n_cols = 5
    rows = [matches[i:i + n_cols] for i in range(0, len(matches), n_cols)]
    for row in rows:
        cols = st.columns(n_cols)
        for col, (newt_id, score, idx) in zip(cols, row):
            with col:
                st.image(thumbnails[idx], width=150)
                st.caption(f"{newt_id} ({score:.3f})\n{dates[idx]}")