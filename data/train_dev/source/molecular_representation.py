"""Declared FREE/AMD peptide representations; no source normalization or repair.

This constructs a database-declared chemistry, not a measured molecular identity.
Only uppercase canonical L-residue sequences and unmodified N termini are modeled.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

AA = frozenset("ARNDCEQGHILKMFPSTWYV")
GLOBAL_NAMES = ["RingCount", "NumAromaticRings", "NumSaturatedRings", "FractionCSP3",
                "HallKierAlpha", "LabuteASA", "MaxPartialCharge", "MinPartialCharge", "TPSA"]
RESIDUE_NAMES = ["MolWt", "MolLogP", "TPSA"]
CLASS = {"FREE": 0, "AMD": 1}


def sequence_sha256(sequence):
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


def molecule_key(sequence, terminal_state):
    """Versioned, unambiguous key; distinct terminal states share a sequence hash."""
    if terminal_state not in CLASS:
        raise ValueError("terminal_state must be explicit FREE or AMD")
    return hashlib.sha256(("AMP_MOLECULE_V1\n" + sequence + "\n" + terminal_state).encode("ascii")).hexdigest()


def _heavy_element_count(mol):
    return Counter(a.GetSymbol() for a in mol.GetAtoms())


def peptide_molecule(sequence, terminal_state):
    if not isinstance(sequence, str) or not 1 <= len(sequence) <= 48 or not set(sequence) <= AA:
        raise ValueError("sequence must be raw uppercase canonical20AA length1-48; no normalization")
    if terminal_state not in CLASS:
        raise ValueError("terminal_state must be explicit FREE or AMD")
    free = Chem.MolFromSequence(sequence, sanitize=True, flavor=0)
    if free is None:
        raise ValueError("RDKit MolFromSequence failed")
    sites = [a.GetIdx() for a in free.GetAtoms()
             if a.GetPDBResidueInfo() is not None and a.GetPDBResidueInfo().GetName().strip() == "OXT"]
    if len(sites) != 1:
        raise ValueError("exactly one C-terminal PDB OXT atom required")
    site = free.GetAtomWithIdx(sites[0])
    if (site.GetAtomicNum() != 8 or site.GetDegree() != 1 or
            site.GetPDBResidueInfo().GetResidueNumber() != len(sequence) or
            free.GetBondBetweenAtoms(site.GetIdx(), site.GetNeighbors()[0].GetIdx()).GetBondType() != Chem.BondType.SINGLE):
        raise ValueError("unexpected OXT atom or connectivity")
    if terminal_state == "FREE":
        return free
    edit = Chem.RWMol(free)
    terminal = edit.GetAtomWithIdx(sites[0])
    terminal.SetAtomicNum(7)
    terminal.SetNoImplicit(False)
    terminal.SetNumExplicitHs(0)
    modified = edit.GetMol()
    Chem.SanitizeMol(modified)
    Chem.AssignStereochemistry(modified, cleanIt=True, force=True)
    if modified.GetAtomWithIdx(sites[0]).GetTotalNumHs() != 2:
        raise ValueError("terminal amide must have exactly two attached H")
    expected = _heavy_element_count(free)
    expected["O"] -= 1
    expected["N"] += 1
    if _heavy_element_count(modified) != expected:
        raise ValueError("amidation changed unexpected elements")
    delta = rdMolDescriptors.CalcExactMolWt(modified) - rdMolDescriptors.CalcExactMolWt(free)
    if not math.isclose(delta, -0.984015588, abs_tol=1e-8):
        raise ValueError("unexpected C-terminal amidation mass delta")
    if any(a.GetChiralTag() != modified.GetAtomWithIdx(a.GetIdx()).GetChiralTag() for a in free.GetAtoms()):
        raise ValueError("amidation changed a residue chiral tag")
    return modified


def representation(sequence, terminal_state):
    mol = peptide_molecule(sequence, terminal_state)
    values = [float(getattr(Descriptors, name)(mol)) for name in GLOBAL_NAMES]
    if not all(math.isfinite(x) for x in values):
        raise ValueError("nonfinite global descriptor")
    return {"sequence_sha256": sequence_sha256(sequence),
            "molecule_sha256": molecule_key(sequence, terminal_state),
            "canonical_isomeric_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
            "formula": rdMolDescriptors.CalcMolFormula(mol),
            "exact_mass": rdMolDescriptors.CalcExactMolWt(mol),
            "global_raw": values}


def residue_lookup():
    # This explicitly defined isolated free-AA table is unchanged by peptide terminal state.
    return {aa: [round(float(getattr(Descriptors, name)(peptide_molecule(aa, "FREE"))), 2)
                 for name in RESIDUE_NAMES] for aa in sorted(AA)}
