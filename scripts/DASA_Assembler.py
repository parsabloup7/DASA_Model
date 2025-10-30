"""
DASA Assembler
--------------
Assemble donor, bridge, and acceptor fragments into DASA structures
using anchored SMILES atoms ([*:1], [*:5], etc.) and optional tethers.
"""

from rdkit import Chem

# ---------- helpers ----------

def _load_mol(smiles: str) -> Chem.Mol:
    """Load SMILES string into an RDKit Mol object."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def _canonical(mol_or_smiles) -> str:
    """Return canonical, isomeric SMILES from Mol or SMILES string."""
    if isinstance(mol_or_smiles, str):
        mol = Chem.MolFromSmiles(mol_or_smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for canonicalization: {mol_or_smiles}")
        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return Chem.MolToSmiles(mol_or_smiles, isomericSmiles=True, canonical=True)


def _find_star_neighbors(mol: Chem.Mol, mapnum: int):
    """Return list of (star_idx, neighbor_idx) for all [*] atoms with a given atom-map number."""
    pairs = []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 0 and atom.GetAtomMapNum() == mapnum:
            neighbors = atom.GetNeighbors()
            if len(neighbors) != 1:
                raise ValueError(f"Star [*:{mapnum}] must have exactly one neighbor (found {len(neighbors)}).")
            pairs.append((atom.GetIdx(), neighbors[0].GetIdx()))
    return pairs


def _has_star(mol: Chem.Mol, mapnum: int) -> bool:
    """Check if a star atom with a given map number exists."""
    return any(a.GetAtomicNum() == 0 and a.GetAtomMapNum() == mapnum for a in mol.GetAtoms())


def _combine_and_connect(mA: Chem.Mol, mB: Chem.Mol, mapnum: int, bond_type: Chem.BondType) -> Chem.Mol:
    """Combine two molecules at the neighbors of [*:mapnum] and remove the stars."""
    pairsA = _find_star_neighbors(mA, mapnum)
    pairsB = _find_star_neighbors(mB, mapnum)

    if len(pairsA) != 1 or len(pairsB) != 1:
        raise ValueError(f"Expected one [*:{mapnum}] in each fragment; found {len(pairsA)} in A and {len(pairsB)} in B.")

    starA, nbrA = pairsA[0]
    starB, nbrB = pairsB[0]

    combo = Chem.CombineMols(mA, mB)
    rw = Chem.RWMol(combo)
    offset = mA.GetNumAtoms()

    # connect the neighbor atoms
    rw.AddBond(nbrA, offset + nbrB, bond_type)

    # remove star atoms
    for idx in sorted([starA, offset + starB], reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _connect_within(mol: Chem.Mol, mapnum: int, bond_type: Chem.BondType) -> Chem.Mol:
    """Connect two [*:mapnum] within the same molecule and remove them (tether closure)."""
    pairs = _find_star_neighbors(mol, mapnum)
    if len(pairs) != 2:
        return mol

    (star1, nbr1), (star2, nbr2) = pairs
    rw = Chem.RWMol(mol)
    rw.AddBond(nbr1, nbr2, bond_type)

    # remove stars
    for idx in sorted([star1, star2], reverse=True):
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    Chem.SanitizeMol(out)
    return out


def _anchor_present(smiles: str, mapnum: int) -> bool:
    """Check if a specific anchored atom-map appears in the SMILES."""
    return f':{mapnum}]' in smiles


def _is_primary_donor(donor_smiles: str) -> bool:
    """Primary donors expose both :5] and :6] anchors."""
    return _anchor_present(donor_smiles, 5) and _anchor_present(donor_smiles, 6)


def _is_tether_bridge(bridge_smiles: str) -> bool:
    """Bridges are tether-capable if they expose :6] anchor."""
    return _anchor_present(bridge_smiles, 6)


# ---------- normalization helpers ----------

def _ensure_list(x):
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _expand_inputs(inputs, converter_func, anchor_mapnum):
    """Expand raw inputs (SMILES or anchored SMILES) into a flat list of anchored SMILES strings."""
    out = []
    for it in _ensure_list(inputs):
        if not isinstance(it, str):
            raise ValueError("Inputs must be SMILES strings or lists of SMILES strings.")
        if _anchor_present(it, anchor_mapnum):
            out.append(it)
        else:
            conv = converter_func(it) if converter_func else it
            if isinstance(conv, list):
                out.extend(conv)
            else:
                out.append(conv)
    return out


# ---------- single assembly ----------

def assemble_one(donor_smiles: str, bridge_smiles: str, acceptor_smiles: str):
    """
    Assemble a single DASA from donor, bridge, and acceptor anchored SMILES.
    Returns (product_smiles, metadata) or (None, metadata) if assembly is infeasible.
    """
    # fallback conversions if anchors missing
    if not _anchor_present(bridge_smiles, 1):
        bridge_smiles = core_to_bridge(bridge_smiles)
    if not _anchor_present(acceptor_smiles, 1):
        acceptor_smiles = convert_acceptor_smiles(acceptor_smiles)
    if not _anchor_present(donor_smiles, 5):
        donor_smiles = convert_donor_smiles(donor_smiles)

    # guard: single anchored SMILES expected
    if any(isinstance(x, list) for x in [donor_smiles, bridge_smiles, acceptor_smiles]):
        raise ValueError("assemble_one expects single anchored SMILES strings; use assemble_all_dasas for batches.")

    donor_primary = _is_primary_donor(donor_smiles)
    bridge_tether = _is_tether_bridge(bridge_smiles)

    # primary donor requires tether-ready bridge
    if donor_primary and not bridge_tether:
        return None, {"reason": "primary donor requires bridge :6]", "donor": donor_smiles, "bridge": bridge_smiles, "acceptor": acceptor_smiles}

    # tether-ready bridge requires donor with :6]
    if bridge_tether and not donor_primary:
        return None, {"reason": "tether-ready bridge requires donor with :6]", "donor": donor_smiles, "bridge": bridge_smiles, "acceptor": acceptor_smiles}

    # load molecules
    donor = _load_mol(donor_smiles)
    bridge = _load_mol(bridge_smiles)
    acceptor = _load_mol(acceptor_smiles)

    # connect acceptor[:1] to bridge[:1] (DOUBLE)
    prod = _combine_and_connect(bridge, acceptor, mapnum=1, bond_type=Chem.BondType.DOUBLE)

    # connect donor[:5] to bridge[:5] (SINGLE)
    prod = _combine_and_connect(prod, donor, mapnum=5, bond_type=Chem.BondType.SINGLE)

    # optional tether closure [:6] (SINGLE)
    if donor_primary and bridge_tether:
        prod = _connect_within(prod, mapnum=6, bond_type=Chem.BondType.SINGLE)

    smi = _canonical(prod)
    meta = {
        "donor": donor_smiles,
        "bridge": bridge_smiles,
        "acceptor": acceptor_smiles,
        "tethered": donor_primary and bridge_tether
    }
    return smi, meta


# ---------- batch assembly ----------

def assemble_all_dasas(donors: list[str], cores: list[str], acceptors: list[str],
                       dedupe: bool = True, return_meta: bool = True):
    """
    Assemble all DASA combinations from donor, core, and acceptor inputs.
    Inputs may be raw SMILES, anchored SMILES, or converter outputs.
    """
    donor_candidates = _expand_inputs(donors, convert_donor_smiles, anchor_mapnum=5)
    bridge_candidates = _expand_inputs(cores, core_to_bridge, anchor_mapnum=1)
    acceptor_candidates = _expand_inputs(acceptors, convert_acceptor_smiles, anchor_mapnum=1)

    products, metas = [], []

    for d in donor_candidates:
        for b in bridge_candidates:
            for a in acceptor_candidates:
                try:
                    smi, meta = assemble_one(d, b, a)
                    if smi:
                        products.append(smi)
                        metas.append(meta)
                except Exception as e:
                    metas.append({"donor": d, "bridge": b, "acceptor": a, "reason": str(e)})

    # deduplicate products
    if dedupe:
        seen = set()
        uniq_products, uniq_metas = [], []
        for smi, meta in zip(products, metas):
            if smi not in seen:
                seen.add(smi)
                uniq_products.append(smi)
                uniq_metas.append(meta)
        products, metas = uniq_products, uniq_metas

    return (products, metas) if return_meta else products
