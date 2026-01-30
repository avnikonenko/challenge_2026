__author__ = 'pavel'

import os
import sys
import gzip
import pickle
import random
import string
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.PropertyMol import PropertyMol
from io import BytesIO
import re


def read_pdbqt(fname, smi, sanitize=True, removeHs=False, sep='MODEL '):
    """
    Read all MODEL entries in input PDBQT file as separate identical molecules. If no MODEL sections then whole file is
    recognized as a single structure (list with a single molecule will be returned)

    :param fname: pdbqt file
    :param smi: SMILES of the molecule in pdbqt file to assing bond orders
    :param sanitize:
    :param removeHs:
    :return: list of molecules
    """

    def fix_pdbqt(pdbqt_block):
        pdbqt_fixed = []
        for line in pdbqt_block.split('\n'):
            if not line.startswith('HETATM') and not line.startswith('ATOM'):
                pdbqt_fixed.append(line)
                continue
            atom_type = line[12:16].strip()
            # autodock vina types
            if 'CA' in line[77:79]:  # Calcium is exception
                atom_pdbqt_type = 'CA'
            else:
                atom_pdbqt_type = re.sub('D|A', '', line[
                                                    77:79]).strip()  # can add meeko macrocycle types (G and \d (CG0 etc) in the sub expression if will be going to use it
            if re.search('\d', atom_type[0]) or len(
                    atom_pdbqt_type) == 2:  # 1HG or two-letter atom names such as CL,FE starts with 13
                atom_format_type = '{:<4s}'.format(atom_type)
            else:  # starts with 14
                atom_format_type = ' {:<3s}'.format(atom_type)
            line = line[:12] + atom_format_type + line[16:]
            pdbqt_fixed.append(line)
        return '\n'.join(pdbqt_fixed)

    def read_pdbqt_block(pdbqt_block):
        return Chem.MolFromPDBBlock('\n'.join([i[:66] for i in pdbqt_block.split('\n')]),
                                    sanitize=sanitize,
                                    removeHs=removeHs)

    mols = []
    refmol = Chem.MolFromSmiles(smi)
    if not removeHs:
        refmol = Chem.AddHs(refmol)
    with open(fname) as f:
        s = f.read()
        if sep in s:
            pdbqt_blocks = s.split(sep)
            for j, block in enumerate(pdbqt_blocks[0:]):
                title = None
                if not block.strip() or (not 'ATOM' and not 'HETATM' in block):
                    continue
                if 'TITLE' in block:
                    for line in block.split('\n'):
                        if 'TITLE ' in line:
                            title = line.strip()
                            break
                m = read_pdbqt_block(block)
                if m is None:
                    new_block = fix_pdbqt(block)
                    m = read_pdbqt_block(new_block)
                    if m is None:
                        sys.stderr.write(f'The pose #{j} cannot be read from {fname}\n {block}')
                        continue
                    else:
                        sys.stderr.write(f'Warning. The pose #{j} from {fname} was fixed \n')

                m = AllChem.AssignBondOrdersFromTemplate(refmol, m)
                if title is not None:
                    m.SetProp('_title', title)
                mols.append(m)
        else:
            title = None
            if 'TITLE' in s:
                for line in s.split('\n'):
                    if 'TITLE ' in line:
                        title = line.strip()
                        break
            m = read_pdbqt_block(s)
            if m is None:
                sys.stderr.write(f'Structure from {fname} cannot be read\n')
            else:
                m = AllChem.AssignBondOrdersFromTemplate(refmol, m)
                if title is not None:
                    m.SetProp('_title', title)
                mols.append(m)

    return mols


def __get_smi_as_molname(mol):
    try:
        name = Chem.MolToSmiles(mol, isomericSmiles=True)
    except Exception as e:
        name = ''.join(random.sample(string.ascii_uppercase, 10))
        sys.stderr.write(f'Some molecule cannot be converted to SMILES - {name} was inserted as the molecule title\n')
    return name


def __read_pkl(fname):
    with open(fname, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def __read_sdf(fname, input_format, id_field_name=None, sanitize=True):
    if input_format == 'sdf':
        suppl = Chem.SDMolSupplier(fname, sanitize=sanitize)
    elif input_format == 'sdf.gz':
        suppl = Chem.ForwardSDMolSupplier(gzip.open(fname), sanitize=sanitize)
    else:
        return
    for mol in suppl:
        if mol is not None:
            if id_field_name is not None:
                mol_title = mol.GetProp(id_field_name)
            else:
                if mol.GetProp("_Name"):
                    mol_title = mol.GetProp("_Name")
                else:
                    mol_title = __get_smi_as_molname(mol)
            yield PropertyMol(mol), mol_title


def __read_sdf_confs(fname, input_format, id_field_name=None, sanitize=True, sdf_confs=False):
    title = None
    for mol, mol_title in __read_sdf(fname, input_format, id_field_name, sanitize):
        if sdf_confs:
            if title is None:
                m = mol
                title = mol_title
            elif title == mol_title:
                m.AddConformer(mol.GetConformer(0), assignId=True)
            else:
                yield m, title
                m = mol
                title = mol_title
        else:
            yield mol, mol_title
    if sdf_confs:
        yield m, title


def __read_smiles(fname, sanitize=True, sep='\t'):
    with open(fname) as f:
        for line in f:
            tmp = line.strip().split(sep)
            mol = Chem.MolFromSmiles(tmp[0], sanitize=sanitize)
            if mol is not None:
                if len(tmp) > 1:
                    mol_title = tmp[1]
                else:
                    mol_title = __get_smi_as_molname(mol)
                mol.SetProp('_Name', mol_title)
                yield mol, mol_title


def __read_stdin_smiles(sanitize=True, sep='\t'):
    line = sys.stdin.readline()
    while line:
        tmp = line.strip().split(sep)
        if tmp:
            mol = Chem.MolFromSmiles(tmp[0], sanitize=sanitize)
            if mol is not None:
                    if len(tmp) > 1:
                        mol_title = tmp[1]
                    else:
                        mol_title = __get_smi_as_molname(mol)
                    yield mol, mol_title
        line = sys.stdin.readline()


def __read_stdin_sdf(sanitize=True):
    molblock = ''
    line = sys.stdin.readline()
    while line:
        molblock += line
        if line == '$$$$\n':
            mol = [x for x in Chem.ForwardSDMolSupplier(BytesIO(molblock.encode('utf-8')), sanitize=sanitize)][0]
            mol_title = molblock.split('\n', 1)[0]
            if not mol_title:
                mol_title = __get_smi_as_molname(mol)
            yield mol, mol_title
            molblock = ''
        line = sys.stdin.readline()


# def read_input(fname, id_field_name=None, stdin_format=None, sanitize=True):
#     if fname is None:
#         if stdin_format == 'smi':
#             suppl = read_stdin_smiles()
#         elif stdin_format == 'sdf':
#             suppl = read_stdin_sdf(sanitize=sanitize)
#         else:
#             raise Exception("Cannot read STDIN. STDIN format should be specified explicitly: smi or sdf.")
#     elif fname.lower().endswith('.sdf') or fname.lower().endswith('.sdf.gz'):
#         suppl = read_sdf(os.path.abspath(fname), id_field_name=id_field_name, sanitize=sanitize)
#     elif fname.lower().endswith('.smi') or fname.lower().endswith('.smiles'):
#         suppl = read_smiles(os.path.abspath(fname))
#     elif fname.lower().endswith('.pkl'):
#         suppl = read_pkl(os.path.abspath(fname))
#     else:
#         raise Exception("File extension can be only SDF, SMI or SMILES")
#     for mol, mol_name in suppl:
#         yield mol, mol_name


def read_input(fname, input_format=None, id_field_name=None, sanitize=True, sdf_confs=False, sep='\t'):
    """
    fname - is a file name, None if STDIN
    input_format - is a format of input data, cannot be None for STDIN
    id_field_name - name of the field containing molecule name, if None molecule title will be taken
    sdf_confs - return consecutive molecules with the same name as a single Mol object with multiple conformers
    sep - separator in SMILES format
    """
    if input_format is None:
        tmp = os.path.basename(fname).split('.')
        if tmp == 'gz':
            input_format = '.'.join(tmp[-2:])
        else:
            input_format = tmp[-1]
    input_format = input_format.lower()
    if fname is None:    # handle STDIN
        if input_format == 'sdf':
            suppl = __read_stdin_sdf(sanitize=sanitize)
        elif input_format == 'smi':
            suppl = __read_stdin_smiles(sanitize=sanitize, sep=sep)
        else:
            raise Exception("Input STDIN format '%s' is not supported. It can be only sdf, smi." % input_format)
    elif input_format in ("sdf", "sdf.gz"):
        suppl = __read_sdf_confs(os.path.abspath(fname), input_format, id_field_name, sanitize, sdf_confs)
    elif input_format in ('smi'):
        suppl = __read_smiles(os.path.abspath(fname), sanitize, sep=sep)
    elif input_format == 'pkl':
        suppl = __read_pkl(os.path.abspath(fname))
    else:
        raise Exception("Input file format '%s' is not supported. It can be only sdf, sdf.gz, smi, pkl." % input_format)
    for mol, mol_name in suppl:
        yield mol, mol_name
