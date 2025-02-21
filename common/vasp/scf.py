#!/usr/bin/python3

from pathlib import Path
from shutil import move

from ase.atoms import Atoms
from ase.calculators.vasp import Vasp
from ase.io.trajectory import Trajectory
from ase.io.vasp import read_vasp_out, write_vasp

from . import COMMON_VASP_PARAMS, get_args


def run_scf_vasp(args: dict):
    name = args['name']
    structures = args['structures']

    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    traj = Trajectory(outdir / f'traj_{name}.traj', 'w', properties=['energy', 'forces'])

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        scf_calc = Vasp(
            system=ID,              # Job name
            directory=calc_fold,    # Directory to run VASP
            kspacing=kspacing,      # k-point grid for SCF
            **COMMON_VASP_PARAMS
        )

        structure.calc = scf_calc

        # 2. SCF #
        e = structure.get_potential_energy()
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
