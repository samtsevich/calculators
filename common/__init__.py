from ase.spectrum.band_structure import BandStructure
from math import ceil

from common.atomic_data import atomic_data

get_N_val_electrons = lambda s: sum([atomic_data[el]['valence_number'] for el in s.get_chemical_symbols()])


def fix_fermi_level(band_structure: BandStructure, N_val_e: int) -> BandStructure:
    assert N_val_e > 0, "Number of valence electrons should be larger than 0"

    lumo_e = band_structure.energies[0, :, ceil(N_val_e/2)]
    homo_e = band_structure.energies[0, :, ceil(N_val_e/2) - 1]

    bandgap = max([min(lumo_e) - max(homo_e), 0.0])
    # print(f'bandgap = {bandgap}')
    if bandgap > 0.0: # insulator or semi-conductor
        band_structure._reference = max(homo_e)
    return band_structure
