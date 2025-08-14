from math import ceil

import numpy as np
from ase.spectrum.band_structure import BandStructure

from .atomic_data import atomic_data

get_N_val_electrons = lambda s: sum(
    [atomic_data[el]["valence_number"] for el in s.get_chemical_symbols()]
)


# For calculations like `neb` or `opt`
N_STEPS = 1000
F_MAX = 0.01

# Default smearing value for DFT calculations in eV
DEF_SMEARING = 0.2

KSPACING = 0.04


def get_KPoints(kspacing: float, cell):
    assert kspacing > 0
    angLattice = cell.cellpar()
    dist = np.zeros(3)
    dist[2] = cell.volume / (
        angLattice[0] * angLattice[1] * np.sin(angLattice[5] * np.pi / 180)
    )
    dist[1] = cell.volume / (
        angLattice[0] * angLattice[2] * np.sin(angLattice[4] * np.pi / 180)
    )
    dist[0] = cell.volume / (
        angLattice[1] * angLattice[2] * np.sin(angLattice[3] * np.pi / 180)
    )

    Kpoints = [int(x) for x in np.ceil(1.0 / (dist * kspacing))]
    return Kpoints


def fix_fermi_level(band_structure: BandStructure, N_val_e: int) -> BandStructure:
    assert N_val_e > 0, "Number of valence electrons should be larger than 0"

    lumo_e = band_structure.energies[:, :, ceil(N_val_e / 2)]
    homo_e = band_structure.energies[:, :, ceil(N_val_e / 2) - 1]

    bandgap = np.max([np.min(lumo_e) - np.max(homo_e), 0.0])
    print(f"bandgap = {bandgap}")
    # if bandgap > 0.0: # insulator or semi-conductor
    band_structure._reference = np.max(homo_e)
    return band_structure
