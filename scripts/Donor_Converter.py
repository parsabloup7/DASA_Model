"""
Donor Converter
---------------

Identify and transform donor nitrogen sites (primary or secondary amines)
for attachment in DASA frameworks.
"""

from rdkit import Chem


class DonorType:
    PRIMARY = "primary"
    SECONDARY = "secondary"


def _is_excluded_N(mol: Chem.Mol, n_idx: int) -> bool:
    """Exclude aromatic, protonated, or sulfonamide/phosphoryl-bound nitrogens."""
    n = mol.GetAtomWithIdx(n_idx)
    if n.GetIsAromatic() or n.GetFormalCharge() != 0:
        return True

    for nei in n.GetNeighbors():
        if nei.GetAtomicNum() in (15, 16):
            has_O_dbl = any(
                b.GetBondType() == Chem.BondType.DOUBLE and (
                    (b.GetBeginAtomIdx() == nei.GetIdx() and mol.GetAtomWithIdx(b.GetEndAtomIdx()).GetAtomicNum() == 8) or
                    (b.GetEndAtomIdx() == nei.GetIdx() and mol.GetAtomWithIdx(b.GetBeginAtomIdx()).GetAtomicNum() == 8)
                )
                for b in mol.GetBonds()
            )
            if has_O_dbl:
                return True
    return False


def _amine_type(mol: Chem.Mol, n_idx: int) -> str | None:
    """
    Classify nitrogen as PRIMARY or SECONDARY based on explicit hydrogen count.

    Returns:
        DonorType.PRIMARY if 2 H & 1 heavy neighbor
        DonorType.SECONDARY if 1 H & 2 heavy neighbors
        None otherwise
    """
    n = mol.GetAtomWithIdx(n_idx)
    if n.GetAtomicNum() != 7 or _is_excluded_N(mol, n_idx):
        return None

    heavy_deg = sum(1 for a in n.GetNeighbors() if a.GetAtomicNum() != 1)
    mol_h = Chem.AddHs(mol)
    n_h = mol_h.GetAtomWithIdx(n_idx)
    explicit_H_count = sum(1 for a in n_h.GetNeighbors() if a.GetAtomicNum() == 1)

    if explicit_H_count == 2 and heavy_deg == 1:
        return DonorType.PRIMARY
    if explicit_H_count == 1 and heavy_deg == 2:
        return DonorType.SECONDARY

    return None


def _replace_explicit_Hs_with_anchors(mol: Chem.Mol, n_idx: int, attach_mapnums: list[int]) -> Chem.Mol:
    """
    Replace explicit hydrogens on a nitrogen with dummy atoms carrying map numbers.

    Args:
        mol: RDKit Mol object
        n_idx: index of nitrogen atom
        attach_mapnums: list of atom-map numbers to assign to dummy atoms

    Returns:
        Sanitized RDKit Mol object with replaced hydrogens
    """
    if mol is None:
        raise ValueError("mol is None")

    rw = Chem.RWMol(Chem.AddHs(mol))
    h_neighbors = [a.GetIdx() for a in rw.GetAtomWithIdx(n_idx).GetNeighbors() if a.GetAtomicNum() == 1]

    n_replace = min(len(h_neighbors), len(attach_mapnums))
    if n_replace == 0:
        newmol = rw.GetMol()
        Chem.SanitizeMol(newmol)
        return newmol

    # Replace H with dummy atoms carrying map numbers
    for h_idx, mapnum in sorted(zip(h_neighbors[:n_replace], attach_mapnums[:n_replace]), key=lambda x: -x[0]):
        star = Chem.Atom(0)
        star.SetAtomMapNum(mapnum)
        star_idx = rw.AddAtom(star)
        rw.AddBond(n_idx, star_idx, Chem.BondType.SINGLE)
        try:
            rw.RemoveBond(n_idx, h_idx)
        except Exception:
            pass
        try:
            rw.RemoveAtom(h_idx)
        except Exception:
            pass

    newmol = rw.GetMol()
    Chem.SanitizeMol(newmol)
    return newmol


def convert_donor_smiles(donor_smiles: str, mode: str = "all") -> list[str]:
    """
    Generate SMILES variants for donor nitrogens.

    Args:
        donor_smiles: input SMILES string
        mode: "all" for all candidate nitrogens, "first" for first only

    Returns:
        List of canonical SMILES with dummy atoms for attachment
    """
    mol = Chem.MolFromSmiles(donor_smiles)
    if mol is None:
        raise ValueError(f"Invalid donor SMILES: {donor_smiles}")

    candidates = [(atom.GetIdx(), tp) for atom in mol.GetAtoms()
                  if atom.GetAtomicNum() == 7 and (tp := _amine_type(mol, atom.GetIdx()))
                  in (DonorType.PRIMARY, DonorType.SECONDARY)]

    if not candidates:
        raise ValueError("Donor must contain a free primary or secondary amine (non-excluded).")

    outputs = []
    for n_idx, tp in (candidates if mode == "all" else candidates[:1]):
        mapnums = [5] if tp == DonorType.SECONDARY else [5, 6]
        newmol = _replace_explicit_Hs_with_anchors(mol, n_idx, mapnums)
        smi = Chem.MolToSmiles(Chem.RemoveHs(newmol), isomericSmiles=True, canonical=True)
        outputs.append(smi)

    return sorted(dict.fromkeys(outputs))
