# add_view_rank.py
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint, DataStructs

ref_smi, _ = open("ref_std.smi").read().split()[:2]
ref = Chem.MolFromSmiles(ref_smi); fp2_ref = RDKFingerprint(ref)

df = pd.read_csv("screen_ecfp6_ranked.csv")
sims2=[]
for s in df.SMILES:
    m = Chem.MolFromSmiles(s)
    sims2.append(DataStructs.TanimotoSimilarity(fp2_ref, RDKFingerprint(m)))
df["sim_rdk"] = sims2
df["rank_ecfp6"] = df.sim_ecfp6.rank(ascending=False, method="min")
df["rank_rdk"]   = df.sim_rdk.rank(ascending=False, method="min")
df["rank_mean"]  = (df.rank_ecfp6 + df.rank_rdk)/2
df.sort_values("rank_mean").to_csv("screen_ensemble_ranked.csv", index=False)
