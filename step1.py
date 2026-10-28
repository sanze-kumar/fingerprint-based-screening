
#!/usr/bin/env python3

import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")


def load_reference(path="ref_std.smi"):
    with open(path, "r") as f:
        parts = f.read().strip().split()

    if not parts:
        raise ValueError("Reference file is empty.")

    smiles = parts[0]
    ref_id = parts[1] if len(parts) > 1 else "REF"
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError("Invalid reference SMILES.")

    return mol, ref_id, smiles


def main():
    ref_path = "ref_std.smi"
    input_csv = "maybridge_std.csv"
    output_csv = "screen_ecfp6_ranked.csv"

    # ECFP6 fingerprint: radius 3, 2048 bits
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=3,
        fpSize=2048,
        includeChirality=True
    )

    # Load reference compound
    ref_mol, ref_id, ref_smiles = load_reference(ref_path)
    ref_fp = generator.GetFingerprint(ref_mol)

    # Load Maybridge library
    df = pd.read_csv(input_csv)

    required = {"ID", "SMILES"}
    if not required.issubset(df.columns):
        raise ValueError(
            "CSV must contain columns named ID and SMILES."
        )

    # Calculate Tanimoto similarity
    scores = []

    for smiles in df["SMILES"]:
        mol = Chem.MolFromSmiles(str(smiles))

        if mol is None:
            scores.append(float("nan"))
            continue

        fp = generator.GetFingerprint(mol)
        score = DataStructs.TanimotoSimilarity(ref_fp, fp)
        scores.append(score)

    df["sim_ecfp6"] = scores

    # Rank compounds by similarity
    df = df.sort_values(
        "sim_ecfp6",
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    df.insert(0, "Rank", range(1, len(df) + 1))

    # Save results
    df.to_csv(output_csv, index=False)

    print(f"Reference ID: {ref_id}")
    print(f"Reference SMILES: {ref_smiles}")
    print(f"Total compounds: {len(df)}")
    print(f"Valid scores: {df['sim_ecfp6'].notna().sum()}")
    print(f"Results saved to: {output_csv}")

    print("\nTop 50 compounds:")
    print(df[["Rank", "ID", "sim_ecfp6"]].head(50).to_string(index=False))


if __name__ == "__main__":
    main()

