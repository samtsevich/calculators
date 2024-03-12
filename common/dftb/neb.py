from ase.calculators.dftb import Dftb
from ase.io.vasp import write_vasp
from ase.io.trajectory import Trajectory
from ase.neb import NEB
from ase.optimize import BFGS, MDMin

from copy import copy

from common.dftb import (get_args,
                         get_additional_params,
                         get_KPoints)

F_MAX = 0.01
N_STEPS = 1000


def dftb_neb(args):
    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type = args.subcommand

    args = get_args(args)
    name = args['name']
    structures = args['structures']

    outdir = args['outdir']
    calc_fold = outdir

    params = copy(args['dftb_params'])
    params.update(get_additional_params(type='scf'))

    # ? add 'stress'
    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        kpts = get_KPoints(args['kspacing'], structure.get_cell())
        params.update({'label': f'{calc_type}_{ID}',
                       'kpts': kpts,})

        structure.set_calculator(Dftb(directory=calc_fold,
                                      **params))

    neb = NEB(images=structures, climb=True)

    optimizer =BFGS(neb,
                    alpha=10.,
                    restart=str(outdir/'neb.pckl'),
                    trajectory=str(outdir/'neb.traj'),
                    logfile=str(outdir/'neb.log'))
    optimizer.run(fmax=F_MAX, steps=N_STEPS)

    with open(outdir/'final_POSCARS', 'w') as fp_final_poscars:
        for structure in neb.images:
            write_vasp( fp_final_poscars, structure,
                        vasp5=True, direct=True)
    print(f'Calculation: {calc_type} of {name} is done.')
