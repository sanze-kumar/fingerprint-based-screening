# standardize.py
from rdkit import Chem
import pandas as pd

def std_smiles(smi):
    m = Chem.MolFromSmiles(smi)
    if not m:
        return None
    try:
        m = Chem.RemoveHs(m)
        Chem.SanitizeMol(m)
        return Chem.MolToSmiles(m, isomericSmiles=True)
    except Exception:
        return None


# 1. Standardize reference compound
with open("ref.smi", "r") as f:
    ref_line = f.read().strip().split()

if not ref_line:
    raise ValueError("ref.smi is empty or contains no SMILES.")

ref_std = std_smiles(ref_line[0])

if ref_std is None:
    raise ValueError("Could not standardize reference SMILES.")

with open("ref_std.smi", "w") as f:
    f.write(f"{ref_std} REF\n")


# 2. Process Maybridge SDF
suppl = Chem.SDMolSupplier(
    "maybridge_clean.sdf",
    removeHs=False
)

rows = []

for i, mol in enumerate(suppl):
    if mol is None:
        continue

    # Retrieve compound ID in the specified priority order
    compound_id = None

    for prop in ["NAME", "ACD_Code", "code"]:
        if mol.HasProp(prop):
            value = mol.GetProp(prop).strip()
            if value:
                compound_id = value
                break

    # Fallback if none of the properties exists
    if not compound_id:
        compound_id = f"MB_{i:07d}"

    # Standardize SMILES
    smiles = std_smiles(Chem.MolToSmiles(mol))

    if smiles:
        rows.append((compound_id, smiles))


# 3. Save output
df = pd.DataFrame(rows, columns=["ID", "SMILES"])

# Remove duplicate SMILES, retaining the first matching ID
df = df.drop_duplicates(subset="SMILES", keep="first")

df.to_csv("maybridge_std.csv", index=False)

print(f"Maybridge compounds saved: {len(df)}")
print("Output file: maybridge_std.csv")