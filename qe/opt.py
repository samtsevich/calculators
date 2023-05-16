#!/usr/bin/python3

import argparse
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


CALC_FILES = ['espresso.pwi', 'espresso.pwo']
KSPACING = 0.04
F_MAX = 0.01


# def copy_calc_files(objects:list, dest):
#     cp_command = f"cp -r {str(origin/'tmp')} {str(origin/'espresso*')} {dest}"
#     os.system(cp_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASE optimization with QE')
    # parser.add_argument('params', type=argparse.FileType('r'))

    parser.add_argument('-c', dest='config', required=False,
                        help='path to the config file')

    parser.add_argument('-i', dest='input', required=False, help='path to the input trajectory file')
    parser.add_argument('-options', dest='options', required=False, help='path to the options file')
    parser.add_argument('-pp', dest='pseudopotentials', required=False, help='dict of pseudopotentials for ASE')
    parser.add_argument('-pp_dir', dest='pp_dir', required=False, help='path to folder with pseudopotentials')
    parser.add_argument('-fixed_idx', dest='fixed_idx', required=False, help='path to the file with fixed indicies')
    parser.add_argument('-outdir', dest='outdir', default='output', required=False, help='path to the output folder')
    parser.add_argument('-kspacing', dest='kspacing', default=0.05, required=False, help='Kspacing value')
    parser.add_argument('-fmax', dest='fmax', default=0.01, required=False, help='fmax for relaxation')
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
    outdir = Path(args['outdir']) if args['outdir'] is not None else input.parent/f'res_{input.stem}'
    outdir.mkdir(exist_ok=True)
    options = args['options']
    assert Path(options).exists(), f"Seems like path to the options file is wrong.\n It is {Path(args['options'])}"

    pp = args['pseudopotentials']
    pp_dir = Path(args['pp_dir'])
    assert pp_dir.exists(), 'Seems like folder with pseudopotentials does not exist or wrong'
    pp_dir = pp_dir.resolve()

    kspacing = args['kspacing'] 
    fmax = args['fmax']

    assert kspacing > 0, 'Seems like your value for KSPACING is not >0'
    assert fmax > 0, 'Seems like your value for FMAX is not >0'

    calc_fold = outdir
    # calc_fold = input.parent

    with open(options) as fp:
        data, card_lines = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')
    data['control']['outdir'] = './tmp'
    data['control']['calculation'] = 'scf'

    opt_calc = Espresso(input_data=data,
                        pseudopotentials=pp,
                        pseudo_dir=str(pp_dir),
                        kspacing=KSPACING,
                        directory=str(calc_fold))


    # Read atomic structure from the inputs 
    atoms = read_vasp(input)
    # atoms = Trajectory(filename=input)[-1]

    # Constraints
    fixed_idx = args['fixed_idx']
    if fixed_idx is not None:
        with open(fixed_idx) as fp:
            idx = list(map(int, fp.read().split()))
            assert len(idx), 'Seems something wrong with idx of fixed atoms'
            atoms.set_constraint(FixAtoms(indices=idx))

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

    ###################
    # 2. OPTIMIZATION #
    ###################

    LOGFILE = outdir/'log'
    RES_TRAJ_FILE = outdir/'res.traj'

    opt = BFGS(atoms, logfile=LOGFILE)
    traj = Trajectory(RES_TRAJ_FILE, 'w', atoms)
    opt.attach(traj)
    opt.run(fmax=F_MAX)

    print(atoms.get_potential_energy())
    traj.write(atoms=atoms)

    for x in CALC_FILES:
        move(calc_fold/x, outdir/f'opt{Path(x).suffix}')

    # copy_calc_files(objects_to_copy, outdir)
    print(f'Optimization of {input} is done.')
