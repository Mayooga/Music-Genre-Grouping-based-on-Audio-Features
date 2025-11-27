# Music Genre Grouping Based on Audio Features

This project provides an unsupervised learning pipeline for clustering songs into coherent genre-like groups using engineered audio features (e.g., tempo, spectral contrast, chroma statistics). Supply a CSV of per-track features, and the script will clean, scale, optionally reduce dimensionality, run several clustering algorithms, and export assignments plus diagnostics.

## 1. Project Structure

- `data/` — place your CSV here (example name: `audio_features.csv`).
- `src/genre_cluster.py` — main pipeline (data prep → dimensionality reduction → clustering → reporting).
- `artifacts/` — automatically populated with the clustered dataset, cluster statistics, and metrics.
- `requirements.txt` — Python dependencies.

## 2. Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Python 3.11+ is recommended for best performance with scikit-learn 1.4+.

## 3. Prepare Your Dataset

1. Export one row per track with numeric audio descriptors (tempo, MFCC means, spectral rolloff, etc.).
2. Optional: include informational columns such as `track_id`, `artist`, `genre` (the latter is ignored by clustering but useful for evaluation).
3. Save the file under `data/audio_features.csv` (or pass a different path via `--data`).

The script automatically selects numeric feature columns; non-numeric columns are kept only in the final report.

## 4. Run the Pipeline

```powershell
python src/genre_cluster.py ^
  --data data/audio_features.csv ^
  --cluster-algo kmeans ^
  --n-clusters 10 ^
  --pca-components 8
```

Key arguments:

| Argument | Description |
| --- | --- |
| `--cluster-algo {kmeans,gmm,agglomerative,dbscan}` | Choose the unsupervised model. |
| `--n-clusters` | Target cluster count (ignored by DBSCAN; default 8). |
| `--pca-components` | Optional dimensionality reduction before clustering. |
| `--label-col` | Optional existing genre column for post-hoc comparison (not used for training). |
| `--output-dir` | Directory for reports (`artifacts/` by default). |

## 5. Outputs

After each run the `artifacts/` folder contains:

- `clustered_audio_features.csv` — original dataframe plus `cluster_label`.
- `cluster_summary.csv` — per-cluster mean of each feature and member counts.
- `cluster_report.json` — metrics (silhouette score, inertia, cluster sizes, etc.).

## 6. Extending the Model

- Swap in UMAP or t-SNE for dimensionality reduction (see TODOs in code).
- Add more clustering backends (HDBSCAN, spectral clustering, etc.).
- Feed embeddings from pretrained audio encoders (e.g., VGGish, OpenL3) instead of hand-crafted features.



## 7. Frontend Dashboard (Optional but Recommended)

A ready-to-demo React dashboard lives in `frontend/`. It consumes the generated artifacts and lets reviewers explore cluster metrics and sampled tracks.

```powershell
cd frontend
npm install
npm run dev
```

The app expects the CSV/JSON exports under `frontend/public/data/` (the repo already includes a sample pulled from the latest clustering run). To refresh with new results, rerun the backend pipeline and copy the updated files into that directory (or wire an API endpoint that serves them dynamically).

**Note:** The files in the `artifacts/` folder are backend-generated outputs.  
They are included here for demonstration purposes to show how the pipeline works.  
When running the backend locally, these files will be generated automatically in the same folder.

