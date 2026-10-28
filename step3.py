# filter_core_optional.py
import pandas as pd
from rdkit import Chem

core = Chem.MolFromSmarts("[N;R3]1CC1")   # aziridine-like
df = pd.read_csv("screen_ensemble_ranked.csv")

def has_core(s):
    m = Chem.MolFromSmiles(s)
    return m.HasSubstructMatch(core)

df["aziridine_core"] = df.SMILES.apply(has_core)
strict = df[(df.aziridine_core==True) & (df.sim_ecfp6>=0.10)].copy()
strict.to_csv("hits_strict_core.csv", index=False)

loose  = df[df.sim_ecfp6>=0.10].copy()
loose.to_csv("hits_loose_no_core.csv", index=False)
