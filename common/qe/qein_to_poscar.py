from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.io.espresso import (
    get_atomic_positions,
    get_atomic_species,
    get_cell_parameters,
    get_valence_electrons,
    ibrav_to_cell,
    label_to_symbol,
    read_fortran_namelist,
)
from ase.io.vasp import write_vasp
from pathlib import Path

import sys


# This is a reimplementation of ase.io.espresso.read_espresso_in,
# because original implementation does not take into account constraints
def read_espresso_in(fileobj):
    from ase.units import create_units

    # Quantum ESPRESSO uses CODATA 2006 internally
    units = create_units("2006")

    data, card_lines = read_fortran_namelist(fileobj)

    # get the cell if ibrav=0
    if "system" not in data:
        raise KeyError("Required section &SYSTEM not found.")
    elif "ibrav" not in data["system"]:
        raise KeyError("ibrav is required in &SYSTEM")
    elif data["system"]["ibrav"] == 0:
        # celldm(1) is in Bohr, A is in angstrom. celldm(1) will be
        # used even if A is also specified.
        if "celldm(1)" in data["system"]:
            alat = data["system"]["celldm(1)"] * units["Bohr"]
        elif "A" in data["system"]:
            alat = data["system"]["A"]
        else:
            alat = None
        cell, cell_alat = get_cell_parameters(card_lines, alat=alat)
    else:
        alat, cell = ibrav_to_cell(data["system"])

    # species_info holds some info for each element
    species_card = get_atomic_species(card_lines, n_species=data["system"]["ntyp"])
    species_info = {}
    for ispec, (label, weight, pseudo) in enumerate(species_card):
        symbol = label_to_symbol(label)
        valence = get_valence_electrons(symbol, data, pseudo)

        # starting_magnetization is in fractions of valence electrons
        magnet_key = "starting_magnetization({0})".format(ispec + 1)
        magmom = valence * data["system"].get(magnet_key, 0.0)
        species_info[symbol] = {
            "weight": weight,
            "pseudo": pseudo,
            "valence": valence,
            "magmom": magmom,
        }

    positions_card = get_atomic_positions(
        card_lines, n_atoms=data["system"]["nat"], cell=cell, alat=alat
    )

    symbols = [label_to_symbol(position[0]) for position in positions_card]
    positions = [position[1] for position in positions_card]
    fixed_idx = [
        i for i, position in enumerate(positions_card) if position[2] is not None
    ]
    magmoms = [species_info[symbol]["magmom"] for symbol in symbols]

    atoms = Atoms(
        symbols=symbols, positions=positions, cell=cell, pbc=True, magmoms=magmoms
    )

    if fixed_idx:
        atoms.set_constraint(FixAtoms(fixed_idx))

    return atoms


init_file = Path(sys.argv[1])

assert init_file.exists()

output = f"{init_file.stem}.vasp"
with open(init_file) as fp:
    struct = read_espresso_in(fp)
    assert len(struct)
    write_vasp(output, struct, direct=True, vasp5=True)
    print(f"Structure is written into {output}")

print("Done")
