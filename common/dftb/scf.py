from typing import List

from ase.atoms import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.calculators.dftb import Dftb
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp

from common.dftb import get_args, get_calc_type_params


def run_dftb_scf(args: dict, calc_type: str):
    # Reading the arguments
    # ------------------------------------------------
    assert calc_type == 'scf', 'This function is only for SCF calculation'

    args = get_args(args, calc_type=calc_type)
    name = args['name']
    structures: List[Atoms] = args['structures']
    structure = structures[-1]

    outdir = args['outdir']
    calc_fold = outdir

    # Printing input parameters
    # ------------------------------------------------
    print(30 * '-')
    print(f'SCF calculation for {name}')
    print(f'kspacing = {args["kspacing"]}')
    print(f'Output directory: {outdir}')
    print(30 * '-')

    params = args['dftb_params']
    params.update(get_calc_type_params(calc_type=calc_type))

    kpts = kptdensity2monkhorstpack(atoms=structure, kptdensity=args['kspacing'])
    params.update({'label': f'out_{calc_type}_{name}',
                    'kpts': kpts,})

    structure.calc = Dftb(directory=calc_fold, **params)
    e = structure.get_potential_energy()

    write_vasp(outdir/f'final_{name}.vasp', structure, sort=True, vasp5=True, direct=True)
    # ? add 'stress'
    with Trajectory(outdir/f'final_{name}.traj', 'w', properties=['energy', 'forces']) as traj:
        traj.write(structure, energy=e, forces=structure.get_forces())

    # Final output message
    # ------------------------------------------------
    print(f'SCF of {name} is done.')
    print(30 * '-')



def dftb_scf(args):
    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type: str = args.subcommand
    args: dict = vars(args)

    run_dftb_scf(args, calc_type=calc_type)
