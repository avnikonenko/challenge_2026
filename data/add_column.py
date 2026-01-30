import pandas as pd

# df1: columns: Smiles, mol_id, activity_class
# df2: columns: smiles, mol_id

# keep only what you need from df1 to avoid accidental column collisions
df2_out = df2.merge(
    df1[['mol_id', 'activity_class']],
    on='mol_id',
    how='left',
    validate='m:1'  # df2 can have many rows per mol_id, df1 must have at most 1
)

df2_out.to_csv('chembl_stereo_class.smi', sep='\t')
