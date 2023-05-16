#!/usr/bin/python3

import argparse
import numpy as np
import os


from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.eos import EquationOfState
from ase.io import read
from ase.io.espresso import read_fortran_namelist
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.units import kJ

from pathlib import Path
from shutil import copyfile

KSPACING = 0.04

def copy_calc_files(origin, dest):
    origin.mkdir(parents=True, exist_ok=True)
    dest.mkdir(exist_ok=True, parents=True)
    cp_command = f"cp -r {str(origin/'tmp')} {str(origin/'espresso*')} {dest}"
    os.system(cp_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EOS with QE for structure')
    parser.add_argument('-i', dest='input', help='path to the input structure')
    parser.add_argument('-k', dest='options', help='path to the options file')
    parser.add_argument('-pp', dest='pseudopotentials', help='dict of pseudopotentials for ASE')
    parser.add_argument('-o', dest='output', help='path to the directory with output files')
    args = parser.parse_args()

    assert Path(args.input).exists(), f'Seems like path to the input file is wrong.\n It is {Path(args.input)}'
    assert Path(args.options).exists(), f'Seems like path to the options file is wrong.\n It is {Path(args.options)}'


    input = Path(args.input)
    calc_fold = input.parent
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)


    # Read data from the inputs 
    atoms = read(input)

    pp = eval(args.pseudopotentials)
    for s in list(set(atoms.get_chemical_symbols())):
        assert s in pp.keys(), f'{s} is not presented in the pseudopotentials'


    with open(args.options) as fp:
        data, card_lines = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')

    # Optimization of the input structure
    # pp = {'Ru': 'Ru_ONCV_PBE-1.2.upf', 'O': 'O_ONCV_PBE-1.0.upf'}

    opt_calc = Espresso(input_data=data, pseudopotentials=pp,
                        kspacing=KSPACING, directory=str(calc_fold))
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
    copy_calc_files(calc_fold, output_dir/'opt')

    print(f'Optimization of {input} is done.')

    # SCF calculator
    data['control']['calculation'] = 'relax'
    scf_calc = Espresso(input_data=data, pseudopotentials=pp,
                        kspacing=KSPACING, directory=calc_fold)
    atoms.calc = scf_calc

    traj_file = output_dir/'res.traj'
    traj = TrajectoryWriter(filename=traj_file, mode='w')

    # EOS part
    volumes, energies = [], []
    cell = atoms.get_cell()
    for x in np.linspace(0.92, 1.08, 9):
        atoms.set_cell(cell * x, scale_atoms=True)
        # cfold = calc_fold/str(x)
        # atoms.calc.set(directory=str(cfold))
        volumes.append(atoms.get_volume())
        energies.append(atoms.get_potential_energy())
        copy_calc_files(calc_fold, output_dir/"{:.2f}".format(x))
        traj.write(atoms=atoms)
    traj.close()

    print(f'Data for EOS for {input} has been collected')

    configs = read(traj_file)
    # Extract volumes and energies:
    # volumes = [x.get_volume() for x in configs]
    # energies = [x.get_potential_energy() for x in configs]
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0, e0, B = eos.fit()
    print(B / kJ * 1.0e24, 'GPa')
    data = eos.getplotdata()
    v_fit = data[-4]/len(atoms)
    e_fit = data[-3]/len(atoms)
    v = data[-1]/len(atoms)
    e = data[-2]/len(atoms)
    np.savetxt(output_dir/'e_vs_v_sc.dat', np.column_stack((v, e)))
    np.savetxt(output_dir/'e_vs_v_sc_fit.dat', np.column_stack((v_fit, e_fit)))
    eos.plot(output_dir/'Ru_sc_eos.png')
