from copy import copy

from ase.calculators.dftb import Dftb
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase.optimize import BFGS

from common.dftb import get_additional_params, get_args, get_KPoints

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
        # Label is used for the output file name
        params.update({'label': f'out_{calc_type}_{ID}',
                       'kpts': kpts,})

        opt_calc = Dftb(directory=calc_fold,
                        **params)

        structure.set_calculator(opt_calc)
        ucf = UnitCellFilter(structure)
        opt = BFGS(ucf,
        # opt = BFGS(structure,
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
