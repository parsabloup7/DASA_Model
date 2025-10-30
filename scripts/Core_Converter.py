"""
Core Convertor
--------------

Detects 5-membered heteroaromatic rings containing an aldehyde and
extracts substituents at ring positions C2–C5. Handles:
- Heteroatoms in the ring (O, N, etc.) and their substituents.
- Tether motifs (TsO-like) at C5 and trims them for bridge attachment.
- Constructs triene bridge SMILES for DASA frameworks using
  [*:1], [*:5], and optionally [*:6] if tether is present.
"""

from rdkit import Chem
from rdkit.Chem import rdmolops

TSO_SMARTS = "O[S](=O)(=O)[#6]"
TSO_PATTERN = Chem.MolFromSmarts(TSO_SMARTS)

def classify_core_any5(mol):
    """Return True if mol contains a 5-membered ring connected to an aldehyde and contains heteroatom(s)."""
    if mol is None:
        return False
    rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if len(r) == 5]
    if not rings:
        return False
    carbonyls = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6
                 and any(mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()) and
                         mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                         for nb in a.GetNeighbors() if nb.GetAtomicNum() == 8)]
    if not carbonyls:
        return False
    for cidx in carbonyls:
        for nb in mol.GetAtomWithIdx(cidx).GetNeighbors():
            nbidx = nb.GetIdx()
            for r in rings:
                if nbidx in r and any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r):
                    return True
    return False

def number_ring_positions(mol):
    """Return C2-C5 positions and full ring atom IDs for a 5-membered ring with an aldehyde."""
    if mol is None:
        raise ValueError("mol is None")
    five_rings = [tuple(r) for r in mol.GetRingInfo().AtomRings() if len(r) == 5]
    if not five_rings:
        raise ValueError("No 5-membered rings found")
    carbonyls = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6
                 and any(mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()) and
                         mol.GetBondBetweenAtoms(a.GetIdx(), nb.GetIdx()).GetBondTypeAsDouble() == 2.0
                         for nb in a.GetNeighbors() if nb.GetAtomicNum() == 8)]
    if not carbonyls:
        raise ValueError("No aldehyde carbonyl found")
    for cidx in carbonyls:
        ring_neighbors = [n.GetIdx() for n in mol.GetAtomWithIdx(cidx).GetNeighbors()
                          if any(n.GetIdx() in r for r in five_rings)]
        for rn in ring_neighbors:
            for r in five_rings:
                if rn not in r or not any(mol.GetAtomWithIdx(i).GetAtomicNum() != 6 for i in r):
                    continue
                ring_list = list(r)
                C2 = rn
                pos = ring_list.index(C2)
                ordered_ring = ring_list[pos+1:] + ring_list[:pos+1]
                carbons = [idx for idx in ordered_ring if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 and idx != C2][:3]
                if len(carbons) != 3:
                    ordered_ring_rev = list(reversed(ring_list))
                    posr = ordered_ring_rev.index(C2)
                    ordered_ring2 = ordered_ring_rev[posr+1:] + ordered_ring_rev[:posr+1]
                    carbons = [idx for idx in ordered_ring2 if mol.GetAtomWithIdx(idx).GetAtomicNum() == 6 and idx != C2][:3]
                if len(carbons) < 3:
                    raise ValueError("Not enough carbons to assign C3-C5")
                C3, C4, C5 = carbons
                return {"C2": C2, "C3": C3, "C4": C4, "C5": C5, "ring_atom_ids": set(ring_list)}
    raise ValueError("Could not determine ring positions")

def extract_substituent(mol, ring_idx):
    """Return substituent SMILES and atom indices attached to ring_idx."""
    if mol is None:
        return None
    ring_atoms = next((set(r) for r in mol.GetRingInfo().AtomRings() if ring_idx in r and len(r) == 5), None)
    if ring_atoms is None:
        ring_atoms = set(a.GetIdx() for a in mol.GetAtoms() if a.IsInRing())
    atom = mol.GetAtomWithIdx(ring_idx)
    for nb in atom.GetNeighbors():
        nb_idx = nb.GetIdx()
        if nb_idx in ring_atoms or nb.GetAtomicNum() == 1:
            continue
        rwm = Chem.RWMol(mol)
        try:
            rwm.RemoveBond(atom.GetIdx(), nb_idx)
        except Exception:
            continue
        frags_idx = Chem.GetMolFrags(rwm.GetMol(), asMols=False)
        picked = next((list(frag) for frag in frags_idx if nb_idx in frag), None)
        if not picked:
            continue
        try:
            smi = Chem.MolFragmentToSmiles(mol, atomsToUse=picked, canonical=True, isomericSmiles=True)
        except Exception:
            submol = Chem.PathToSubmol(mol, picked)
            try:
                Chem.SanitizeMol(submol)
            except Exception:
                pass
            smi = Chem.MolToSmiles(submol, isomericSmiles=True)
        return {"smiles": smi, "atom_indices": picked, "attach_idx": ring_idx}
    return None

def find_tso_and_proximal_fragment(mol, sub_info):
    if sub_info is None:
        return (False, None)
    atom_indices = sub_info["atom_indices"]
    attach_idx = sub_info["attach_idx"]
    submol = Chem.PathToSubmol(mol, atom_indices)
    if submol is None:
        return (False, None)
    match = submol.GetSubstructMatch(TSO_PATTERN)
    if not match:
        return (False, None)
    tso_o_sub_idx = match[0]
    tso_o_orig_idx = atom_indices[tso_o_sub_idx]
    try:
        path = list(rdmolops.GetShortestPath(mol, attach_idx, tso_o_orig_idx))
    except Exception:
        return (False, None)
    if len(path) < 2:
        return (False, None)
    prox_indices = [p for p in path if p != tso_o_orig_idx and p in atom_indices]
    return (True, prox_indices)

def is_tether_ready_and_trim(mol, c5_idx):
    sub_info = extract_substituent(mol, c5_idx)
    if sub_info is None:
        return (False, None, None)
    full_smi = sub_info["smiles"]
    found, proximal_atom_indices = find_tso_and_proximal_fragment(mol, sub_info)
    if not found or not proximal_atom_indices:
        return (False, None, full_smi)
    try:
        prox_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=proximal_atom_indices, canonical=True, isomericSmiles=True)
    except Exception:
        prox_submol = Chem.PathToSubmol(mol, proximal_atom_indices)
        try:
            Chem.SanitizeMol(prox_submol)
        except Exception:
            pass
        prox_smi = Chem.MolToSmiles(prox_submol, isomericSmiles=True)
    if prox_smi and prox_smi.endswith("O") and not prox_smi.endswith("=O") and not prox_smi.endswith("CO"):
        if len(proximal_atom_indices) > 1:
            trimmed = proximal_atom_indices[:-1]
            try:
                prox_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=trimmed, canonical=True, isomericSmiles=True)
            except Exception:
                prox_submol = Chem.PathToSubmol(mol, trimmed)
                try:
                    Chem.SanitizeMol(prox_submol)
                except Exception:
                    pass
                prox_smi = Chem.MolToSmiles(prox_submol, isomericSmiles=True)
    return (True, prox_smi, full_smi)

def hetero_substituent_for_C2(mol, ring_atom_ids):
    for idx in ring_atom_ids:
        atom = mol.GetAtomWithIdx(idx)
        if atom.GetAtomicNum() == 6:
            continue
        sym = atom.GetSymbol()
        non_ring_neighbors = [n for n in atom.GetNeighbors() if not n.IsInRing() and n.GetAtomicNum() != 1]
        hcount = atom.GetTotalNumHs()
        new_h = hcount + 1
        if not non_ring_neighbors:
            if sym == "O":
                return "O"
            if sym == "N":
                return "[NH]" if new_h == 1 else "[NH2]" if new_h == 2 else f"[N{new_h}]"
            return f"[{sym}H]" if new_h == 1 else f"[{sym}H{new_h}]"
        ring_set = set(ring_atom_ids)
        seen, collected = set(), []
        queue = [n.GetIdx() for n in non_ring_neighbors]
        while queue:
            cur = queue.pop(0)
            if cur in seen:
                continue
            seen.add(cur)
            collected.append(cur)
            for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
                nid = nb.GetIdx()
                if nid not in seen and nid not in ring_set and nb.GetAtomicNum() != 1:
                    queue.append(nid)
        root_idx = non_ring_neighbors[0].GetIdx()
        if root_idx not in collected:
            collected.insert(0, root_idx)
        try:
            substituent_smi = Chem.MolFragmentToSmiles(mol, atomsToUse=collected, canonical=True, isomericSmiles=True, rootedAtAtom=root_idx)
        except Exception:
            submol = Chem.PathToSubmol(mol, collected)
            try:
                Chem.SanitizeMol(submol)
            except Exception:
                pass
            substituent_smi = Chem.MolToSmiles(submol, isomericSmiles=True, canonical=True)
        prefix = "O" if sym == "O" else "[NH]" if new_h == 1 else "[NH2]" if new_h == 2 else f"[{sym}H{new_h}]" if new_h > 1 else f"[{sym}H]"
        return prefix + substituent_smi
    return None

def make_bridge_smiles_any5(c2_sub=None, c3_sub=None, c4_sub=None, c5_sub=None, tether=False, tether_prox_smi=None):
    s = "[*:1]=C/C"
    if c2_sub:
        s += f"({c2_sub})"
    s += "=C"
    if c3_sub:
        s += f"({c3_sub})"
    s += "/C"
    if c4_sub:
        s += f"({c4_sub})"
    if tether and tether_prox_smi:
        s += f"=C({tether_prox_smi}[*:6])/[*:5]"
    else:
        s += f"=C({c5_sub})/[*:5]" if c5_sub else "=C/[*:5]"
    return s

def core_to_bridge(core_smiles):
    mol = Chem.MolFromSmiles(core_smiles)
    if mol is None:
        raise ValueError("Invalid core SMILES")
    pos = number_ring_positions(mol)
    c3_info = extract_substituent(mol, pos["C3"])
    c4_info = extract_substituent(mol, pos["C4"])
    c5_info = extract_substituent(mol, pos["C5"])
    tether_flag, prox_smi, full_smi = is_tether_ready_and_trim(mol, pos["C5"])
    c2_sub = hetero_substituent_for_C2(mol, pos["ring_atom_ids"])
    bridge = make_bridge_smiles_any5(
        c2_sub=c2_sub,
        c3_sub=c3_info["smiles"] if c3_info else None,
        c4_sub=c4_info["smiles"] if c4_info else None,
        c5_sub=c5_info["smiles"] if c5_info and not tether_flag else None,
        tether=tether_flag,
        tether_prox_smi=prox_smi if tether_flag else None,
    )
    return bridge
