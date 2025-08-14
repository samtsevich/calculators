#!/usr/bin/python3

import argparse
import os
from pathlib import Path

from ase.calculators.espresso import Espresso
from ase.io.espresso import read_fortran_namelist
from ase.io.trajectory import Trajectory, TrajectoryWriter


def copy_calc_files(origin, dest):
    origin.mkdir(parents=True, exist_ok=True)
    dest.mkdir(exist_ok=True, parents=True)
    cp_command = f"cp -r {str(origin / 'tmp')} {str(origin / 'espresso*')} {dest}"
    os.system(cp_command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EOS with QE for structure")
    parser.add_argument("-i", dest="input", help="path to the input trajectory file")
    parser.add_argument("-k", dest="options", help="path to the options file")
    parser.add_argument(
        "-pp", dest="pseudopotentials", help="dict of pseudopotentials for ASE"
    )
    parser.add_argument(
        "-o", dest="output", help="path to the output file with trajectory"
    )
    args = parser.parse_args()

    pp = eval(args.pseudopotentials)
    assert Path(args.input).exists(), (
        f"Seems like path to the input file is wrong.\n It is {Path(args.input)}"
    )
    assert Path(args.options).exists(), (
        f"Seems like path to the options file is wrong.\n It is {Path(args.options)}"
    )

    input = Path(args.input)
    calc_fold = input.parent
    traj_file = Path(args.output)

    with open(args.options) as fp:
        data, card_lines = read_fortran_namelist(fp)
        if "system" not in data:
            raise KeyError("Required section &SYSTEM not found.")

    opt_calc = Espresso(
        input_data=data,
        pseudopotentials=pp,
        kspacing=KSPACING,
        directory=str(calc_fold),
    )

    # Read data from the inputs
    structures = Trajectory(input)
    # structures = [atoms] if atoms is list else atoms

    traj = TrajectoryWriter(filename=traj_file, mode="w")

    for atoms in structures:
        for s in list(set(atoms.get_chemical_symbols())):
            assert s in pp.keys(), f"{s} is not presented in the pseudopotentials"

        atoms.calc = opt_calc

        # add rattling to the atomic positions
        # add_coords = 0.05 - 0.1 * np.random.rand(len(atoms), 3)
        # new_coords = atoms.get_scaled_positions() + add_coords
        # atoms.set_scaled_positions(new_coords)

        # add rattling to the cell
        # add_cell = 0.1 * np.random.rand(3,3)
        # new_cell = atoms.get_cell() + add_cell
        # atoms.set_cell(new_cell, scale_atoms=True)

        print(atoms.get_potential_energy())
        traj.write(atoms=atoms)

        # copy_calc_files(calc_fold, calc_fold/'opt')
        print(f"Optimization of {input} is done.")
    traj.close()
