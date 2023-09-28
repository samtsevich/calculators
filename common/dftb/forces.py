from ase.calculators.dftb import Dftb
from ase.io import read, write, Trajectory
from ase.io.vasp import write_vasp
from copy import copy
from pathlib import Path

from common.dftb import (get_args,
                         get_additional_params,
                         get_KPoints)


def dftb_forces(args):
    name = args['name']
    outdir = args['outdir']

    calc_fold = outdir

    structures = args['structures']

    params = copy(args['dftb_params'])
    params.update(get_additional_params(type='scf'))
    params.update({'label': f'forces_{ID}',})



    scf_calc = Dftb(directory=calc_fold,
                    **params)
    out_traj_forces = Trajectory(outdir/f'res_{name}.traj', 'w')

    for i, atoms in enumerate(structures):
        ID = f'{name}_{i}'
        print(f'Structure {ID}')
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
