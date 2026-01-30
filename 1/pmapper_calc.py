import pmapper
from pmapper.pharmacophore import Pharmacophore as P
from rdkit import Chem
from rdkit.Chem import AllChem
from pprint import pprint
mol = Chem.MolFromSmiles('C1CC(=O)NC(=O)C1N2C(=O)C3=CC=CC=C3C2=O')  # talidomide
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.EmbedMolecule(mol, randomSeed=42)

# create pharmacophore
p = P()
p.load_from_mol(mol)

b = p.get_fp(min_features=4, max_features=4)   # set of activated bits
print(b)