#!/usr/bin/python3

import argparse
import os
import yaml

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.constraints import FixAtoms
from ase.io import read
from ase.io.espresso import read_fortran_namelist
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.io.vasp import read_vasp
from ase.optimize import BFGS
from ase.units import kJ

from pathlib import Path
from pprint import pprint
from shutil import move, copy2


KSPACING = 0.04
CALC_FILES = ['espresso.pwi', 'espresso.pwo']

# def copy_calc_files(objects:list, dest):
#     cp_command = f"cp -r {str(origin/'tmp')} {str(origin/'espresso*')} {dest}"
#     os.system(cp_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASE optimization with QE')
    # parser.add_argument('params', type=argparse.FileType('r'))

    parser.add_argument('-c', dest='config', required=False,
                        help='path to the config file')

    parser.add_argument('-i', dest='input', required=False,
                        help='path to the input trajectory file')
    parser.add_argument('-options', dest='options',
                        required=False, help='path to the options file')
    parser.add_argument('-pp', dest='pseudopotentials',
                        required=False, help='dict of pseudopotentials for ASE')
    parser.add_argument('-pp_dir', dest='pp_dir', required=False,
                        help='path to folder with pseudopotentials')
    parser.add_argument('-o', '-outdir', dest='outdir', default='output',
                        required=False, help='path to the output folder')
    parser.add_argument('-kspacing', dest='kspacing',
                        default=0.05, required=False, help='Kspacing value')
    args = parser.parse_args()

    if args.config:
        assert Path(args.config).exists()
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt
    else:
        args.pseudopotentials = eval(args.pseudopotentials)
        args = vars(args)

    #####################
    # 1. READING INPUTS #
    #####################

    # assert Path(args.pseudopotentials).exists(), 'Please check paths to PP files'
    input = Path(args['input'])
    assert input.exists(), f'Seems like path to the input file is wrong.\n It is {input}'

    outdir = Path(args['outdir']) if args['outdir'] is not None else Path.cwd()/f'res_{input.stem}'
    outdir.mkdir(exist_ok=True)

    options = args['options']
    assert Path(options).exists(), f"Seems like path to the options file is wrong.\n It is {Path(args['options'])}"

    pp = args['pseudopotentials']
    pp_dir = Path(args['pp_dir'])
    if pp_dir.is_symlink():
        pp_dir = pp_dir.readlink()
    assert pp_dir.is_dir(), 'Seems like folder with pseudopotentials does not exist or wrong'
    pp_dir = pp_dir.resolve()

    kspacing = args['kspacing']

    assert kspacing > 0, 'Seems like your value for KSPACING is not >0'

    calc_fold = outdir
    # calc_fold = input.parent

    with open(options) as fp:
        data, card_lines = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')
    data['control']['outdir'] = './tmp'
    data['control']['calculation'] = 'scf'
    data['control']['wf_collect'] = True

    opt_calc = Espresso(input_data=data,
                        pseudopotentials=pp,
                        pseudo_dir=str(pp_dir),
                        kspacing=KSPACING,
                        directory=str(calc_fold))

    # Read atomic structure from the inputs
    atoms = read_vasp(input)
    # atoms = Trajectory(filename=input)[-1]

    for s in list(set(atoms.get_chemical_symbols())):
        assert s in pp.keys(), f'{s} is not presented in the pseudopotentials'

    atoms.calc = opt_calc

    # add rattling to the atomic positions
    # add_coords = 0.05 - 0.1 * np.random.rand(len(atoms), 3)
    # new_coords = atoms.get_scaled_positions() + add_coords
    # atoms.set_scaled_positions(new_coords)

    # add rattling to the cell
    # add_cell = 0.1 * np.random.rand(3,3)
    # new_cell = atoms.get_cell() + add_cell
    # atoms.set_cell(new_cell, scale_atoms=True)

    ##########
    # 2. SCF #
    ##########

    print(atoms.get_potential_energy())

    for x in CALC_FILES:
        move(calc_fold/x, outdir/f'scf{Path(x).suffix}')

    print(f'SCF of {input} is done.')
