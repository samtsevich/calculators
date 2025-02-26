#!/usr/bin/python3

from pathlib import Path
from shutil import move

from ase.atoms import Atoms
from ase.calculators.vasp import Vasp
from ase.io.trajectory import Trajectory
from ase.io.vasp import read_vasp_out, write_vasp

from . import get_basic_params, get_args


def run_scf_vasp(args: dict):
    name = args['name']
    structures = args['structures']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    traj = Trajectory(outdir / f'traj_{name}.traj', 'w', properties=['energy', 'forces'])

    scf_params = get_basic_params(args)
    scf_params.update({'directory': calc_fold})

    scf_calc = Vasp(**scf_params)

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        structure.calc = scf_calc

        # 2. SCF #
        e = structure.get_potential_energy()
        print('EFermi: ', structure.calc.get_fermi_level())
        write_vasp(outdir / f'final_{ID}.vasp', structure, sort=True, vasp5=True, direct=True)

        move(calc_fold / 'INCAR', outdir / f'INCAR.{ID}')
        move(calc_fold / 'OUTCAR', outdir / f'OUTCAR.{ID}')

        traj.write(structure)

        print(f'SCF of {ID} is done.')

    traj.close()


def vasp_scf(args):
    assert args.command == 'vasp', 'This function is only for VASP'
    args: dict = get_args(args)

    run_scf_vasp(args)
