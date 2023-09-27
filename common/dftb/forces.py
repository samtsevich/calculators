from ase.calculators.dftb import Dftb

from ase.io import read, write, Trajectory
from ase.io.vasp import write_vasp

from pathlib import Path

import argparse
import numpy as np

from common.dftb import (get_args,
               get_additional_params)

KSPACING = 0.02

def dftb_forces(args):


    name = args['name']
    outdir = args['outdir']

    params = args['dftb_params']
    params.update(get_additional_params(type='scf'))
    params.update({'label': f'forces_{name}',})

    scf_calc = Dftb(**params)

    in_traj = args['trajectory']
    out_traj_forces = Trajectory(outdir/f'res_{name}.traj', 'w')

    for i, atoms in enumerate(in_traj):
        print(f'Structure {i}')
        calc_fold = outdir/f'struct_{i}'
        scf_calc.directory = str(calc_fold)

        atoms.write(calc_fold/f'a_{name}_{i}.gen')
        write_vasp(calc_fold/f'a_{name}_{i}.vasp', atoms, sort=True, vasp5=True, direct=True)
        atoms.set_calculator(scf_calc)
        # try:
        forces = atoms.get_forces()
        e = atoms.get_potential_energy()
        out_traj_forces.write(atoms, forces=forces, energy=e)
        # except:
        #     print(f'name_{i} did not converge')


if __name__ == "__main__":
    args = get_args(calc_type='forces')
    dftb_forces(args)
