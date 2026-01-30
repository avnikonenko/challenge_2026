if __name__ == '__main__':
    from read_input import read_input
else:
    from .read_input import read_input

import os
import argparse
import numpy as np
import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem

def rdkit_numpy_convert(fp):
    output = []
    for f in fp:
        arr = np.zeros((1,))
        DataStructs.ConvertToNumpyArray(f, arr)
        output.append(arr)
    return np.asarray(output)


def main(fname, path=None):
    if path is None:
        path = os.path.dirname(fname)

    names = []

    data = pd.read_csv(fname,sep='\t',names=['smi','id','act'])
    # for n,row in data.iterrows():
    #     mol = Chem.MolFromSmiles(row['smi'])
    #     fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    #

    mols = [Chem.MolFromSmiles(i) for i in data['smi'].to_list()]
    fp = [ AllChem.GetMorganFingerprintAsBitVect(mol, 2) for mol in mols]
    x = rdkit_numpy_convert(fp)
    pdres = pd.DataFrame(x)
    pdres.loc[:, 'mol_id'] = data['id']
    pdres.loc[:, 'act'] = data['act']

    out_path = os.path.join(path,
                            'MorganFprRDKit_{f_name}_{nconf}.csv'.format(f_name=os.path.basename(fname).split('.')[0],
                                                                         nconf=0))

    pdres.to_csv(out_path, index=False)

    return out_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input', metavar='input.smi', required=True,
                        help='smi for calculation of the 2D Morgan Fingerprints RDKit. sep: tab. Columns: <smi>, <name>, [<act>]. '
                             'Without colname')
    parser.add_argument('-p', '--path', metavar='path', default=None, help='out path')

    args = parser.parse_args()
    _in_fname = args.input
    _path = args.path

    main(fname=_in_fname, path=_path)
