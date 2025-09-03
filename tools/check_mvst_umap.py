import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import sys

fused = sys.argv[1]
df = pd.read_csv(fused)
meta = ["filepath","patient_id","mode","label"]
feat_cols = [c for c in df.columns if c not in meta]
X = df[feat_cols].to_numpy(dtype=float)
X = StandardScaler().fit_transform(X)

Z = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42).fit_transform(X)
df["umap_x"] = Z[:,0]; df["umap_y"] = Z[:,1]

plt.figure(figsize=(7,6))
sns.scatterplot(data=df, x="umap_x", y="umap_y", hue="label", style="mode", s=40, palette="tab10")
plt.title("UMAP — MVST fused embeddings")
plt.grid(True, alpha=0.25); plt.tight_layout(); plt.show()
