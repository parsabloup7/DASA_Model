"""
Acceptor Converter
------------------
Identify α‑CH2 carbons adjacent to carbonyls in acceptor molecules
and convert them into anchored motifs [*:1] for DASA assembly.
"""

from rdkit import Chem
from rdkit.Chem import AllChem

# SMARTS pattern for α-CH2 adjacent to carbonyl
ALPHA_CH2_NEXT_TO_CARBONYL = Chem.MolFromSmarts("[CH2]-C(=O)")


def _canonical(smiles: str) -> str:
    """Return canonical, isomeric SMILES for a molecule."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def _add_anchor_to_hydrogens(mol: Chem.Mol, carbon_idx: int, mapnum: int = 1) -> Chem.Mol:
    """
    Replace hydrogens on a given carbon with a dummy atom [*] carrying an atom-map number.
    Returns a new sanitized molecule.
    """
    em = Chem.EditableMol(mol)
    carbon = mol.GetAtomWithIdx(carbon_idx)
    hydrogen_indices = [h.GetIdx() for h in carbon.GetNeighbors() if h.GetAtomicNum() == 1]

    for h_idx in hydrogen_indices:
        em.RemoveAtom(h_idx)

    star = Chem.Atom(0)  # atomic number 0 = '*'
    star.SetAtomMapNum(mapnum)
    s_idx = em.AddAtom(star)
    em.AddBond(carbon_idx, s_idx, Chem.BondType.DOUBLE)

    new_mol = em.GetMol()
    Chem.SanitizeMol(new_mol)
    return new_mol


def convert_acceptor_smiles(acceptor_smiles: str, mode: str = "all") -> list[str]:
    """
    Convert an acceptor SMILES into anchored motifs for DASA assembly.

    Parameters
    ----------
    acceptor_smiles : str
        Input acceptor SMILES string.
    mode : str, default="all"
        "all"  -> generate motif for all α‑CH2 sites
        "first"-> generate motif only for the first site

    Returns
    -------
    List[str]
        Canonical SMILES strings with α‑CH2 replaced by [*:1].
    """
    mol = Chem.MolFromSmiles(acceptor_smiles)
    if mol is None:
        raise ValueError(f"Invalid acceptor SMILES: {acceptor_smiles}")

    if "[*:1]" in acceptor_smiles:
        return [_canonical(acceptor_smiles)]

    matches = mol.GetSubstructMatches(ALPHA_CH2_NEXT_TO_CARBONYL)
    if not matches:
        raise ValueError("No α‑CH2 adjacent to C=O found in this acceptor.")

    alpha_indices = []
    seen = set()
    for m in matches:
        if m[0] not in seen:
            seen.add(m[0])
            alpha_indices.append(m[0])

    sites = alpha_indices if mode == "all" else alpha_indices[:1]

    anchored_smiles = []
    for idx in sites:
        anchored_mol = _add_anchor_to_hydrogens(mol, carbon_idx=idx, mapnum=1)
        anchored_smiles.append(Chem.MolToSmiles(anchored_mol, isomericSmiles=True, canonical=True))

    return sorted(set(_canonical(s) for s in anchored_smiles))
