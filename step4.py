#!/usr/bin/env python3
# diversity.py — diverse medoid picker with Butina clustering (fixed)
# Usage:
#   python diversity.py [input.csv] [output.csv] [topN] [finalK] [cutoff]
# Inputs need:
#   - column 'SMILES' and either 'rank_mean' (lower=better) or 'sim_ecfp6' (higher=better)

import sys
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.ML.Cluster import Butina

# Silence RDKit chatter (optional)
RDLogger.DisableLog('rdApp.*')

inp    = sys.argv[1] if len(sys.argv) > 1 else "screen_ecfp6_ranked.csv"
outp   = sys.argv[2] if len(sys.argv) > 2 else "maybridge_graphscreen_diverse_top50.csv"
topN   = int(sys.argv[3]) if len(sys.argv) > 3 else 500
finalK = int(sys.argv[4]) if len(sys.argv) > 4 else 50
cutoff = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6  # Tanimoto distance threshold (0.6–0.7 typical)

# Fingerprint generator (ECFP6)
try:
    GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048, includeChirality=False)
    def mfp(m): return GEN.GetFingerprint(m)
except Exception:
    # Fallback (old API)
    def mfp(m): return rdMolDescriptors.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)

df = pd.read_csv(inp)

# ---------- sanity checks ----------
if df.shape[0] == 0:
    print(f"[WARN] Input has 0 rows: {inp}. Nothing to do.")
    df.to_csv(outp, index=False)
    sys.exit(0)

if "SMILES" not in df.columns:
    raise ValueError("Input must contain a 'SMILES' column.")

# Decide ranking key
use_rank = "rank_mean" in df.columns
if use_rank:
    df = df.sort_values("rank_mean", ascending=True)
    score_col = "rank_mean"
else:
    if "sim_ecfp6" not in df.columns:
        raise ValueError("Need 'rank_mean' or 'sim_ecfp6' column to rank.")
    df = df.sort_values("sim_ecfp6", ascending=False)
    score_col = "sim_ecfp6"

# Select topN safely and reset index so Butina indices match rows
topN = min(topN, len(df))
sel = df.head(topN).copy().reset_index(drop=True)

# Handle tiny cases
if len(sel) == 0:
    print("[WARN] After selection, no rows remain.")
    sel.to_csv(outp, index=False); sys.exit(0)
if len(sel) == 1:
    print("[INFO] Only one molecule; writing it out.")
    sel["cluster_id"] = 0
    sel.to_csv(outp, index=False); sys.exit(0)

# Build fingerprints, skipping invalid SMILES
fps = []
keep_idx = []
for i, s in enumerate(sel.SMILES):
    m = Chem.MolFromSmiles(str(s))
    if m is None:
        continue
    fps.append(mfp(m))
    keep_idx.append(i)

if len(fps) == 0:
    print("[WARN] No valid SMILES after parsing.")
    sel.iloc[0:0].to_csv(outp, index=False); sys.exit(0)

if len(keep_idx) != len(sel):
    sel = sel.iloc[keep_idx].reset_index(drop=True)

# Distances for Butina (condensed upper-triangular)
dists = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend([1.0 - x for x in sims])  # distance = 1 - similarity

# Cluster — NOTE: distThresh is positional, not a named arg
clusters = Butina.ClusterData(dists, len(fps), cutoff, isDistData=True)

# Assign cluster IDs
sel["cluster_id"] = -1
for cid, members in enumerate(clusters):
    for pos in members:
        sel.loc[pos, "cluster_id"] = cid

if (sel["cluster_id"] < 0).all():
    print("[WARN] Clustering returned no assignments; assigning unique clusters.")
    sel["cluster_id"] = range(len(sel))

# Pick medoid per cluster (best score in each cluster)
ascending = True if use_rank else False  # rank_mean: smaller is better; sim: larger is better
medoids = (sel.sort_values(["cluster_id", score_col], ascending=[True, ascending])
             .drop_duplicates("cluster_id"))

# Limit to finalK
finalK = min(finalK, len(medoids))
final = medoids.head(finalK).reset_index(drop=True)
final.to_csv(outp, index=False)

print(f"[OK] Wrote {outp} | input={len(df)} | selected(topN)={len(sel)} "
      f"| clusters={len(medoids)} | returned={len(final)} | cutoff={cutoff}")
