from copy import copy
from typing import List

from ase.atoms import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.calculators.dftb import Dftb
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase.neb import NEB
from ase.optimize import BFGS, MDMin

from common.dftb import get_args, get_calc_type_params


def run_neb_dftb(args: dict, calc_type: str):
    args = get_args(args, calc_type=calc_type)
    name = args['name']
    structures: List[Atoms] = args['structures']

    F_MAX = args['fmax']
    assert F_MAX > 0, 'F_MAX should be positive'

    N_STEPS = args['nsteps']
    assert N_STEPS > 0, 'N_STEPS should be positive'

    outdir = args['outdir']
    calc_fold = outdir

    params = copy(args['dftb_params'])
    params.update(get_calc_type_params(calc_type='scf'))

    # ? add 'stress'
    for i, structure in enumerate(structures):
        kpts = kptdensity2monkhorstpack(atoms=structure, kptdensity=args['kspacing'])
        params.update({'label': f'out_{calc_type}_{name}_{i}',
                       'kpts': kpts,})
        structure.calc = Dftb(directory=calc_fold, **params)
        e = structure.get_potential_energy()

    neb = NEB(images=structures, climb=True)
    optimizer =BFGS(neb,
                    alpha=10.,
                    restart=str(outdir/'neb.pckl'),
                    trajectory=str(outdir/'neb.traj'),
                    logfile=str(outdir/'neb.log'))
    optimizer.run(fmax=F_MAX, steps=N_STEPS)

    with open(outdir/'final_POSCARS', 'w') as fp_final_poscars:
        for structure in neb.images:
            write_vasp(fp_final_poscars, structure, vasp5=True, direct=True)

    with Trajectory(outdir/f'final_{name}.traj', 'w', properties=['energy', 'forces']) as traj:
        for structure in neb.images:
            traj.write(structure)
    print(f'Calculation: {calc_type} of {name} is done.')


def dftb_neb(args):
    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type = args.subcommand
    args: dict = vars(args)

    run_neb_dftb(args, calc_type=calc_type)
