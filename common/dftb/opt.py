from ase.calculators.dftb import Dftb
from ase.io.vasp import write_vasp
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from copy import copy

from common.dftb import (get_args,
                         get_additional_params,
                         get_KPoints)

F_MAX = 0.01
N_STEPS = 1000

def dftb_opt(args):
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
    traj = Trajectory(outdir/f'traj_{name}.traj', 'w', properties=['energy', 'forces'])


    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        kpts = get_KPoints(args['kspacing'], structure.get_cell())
        params.update({'label': f'{calc_type}_{ID}',
                       'kpts': kpts,})

        opt_calc = Dftb(directory=calc_fold,
                        **params)

        structure.set_calculator(opt_calc)

        opt = BFGS(structure,
                   alpha=10.,
                   restart=str(outdir/'optimization.pckl'),
                   trajectory=str(outdir/'optimization.traj'),
                   logfile=str(outdir/'optimization.log'))
        opt.run(fmax=F_MAX, steps=N_STEPS)

        # e = structure.get_potential_energy()
        write_vasp(outdir/f'final_{ID}.vasp', structure,
                   sort=True, vasp5=True, direct=True)
        traj.write(structure)
        print(f'Calculation: {calc_type} of {ID} is done.')

    traj.close()
    # print('Optimization done!')


# if __name__ == "__main__":
#     args = get_args(calc_type='opt')
#     dftb_opt(args)
