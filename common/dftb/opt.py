from copy import copy

from ase.atoms import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.calculators.dftb import Dftb
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase.optimize import BFGS

from .. import get_KPoints
from . import get_args, get_calc_type_params


def run_opt_dftb(args: dict, calc_type: str):
    # Reading the arguments
    # ------------------------------------------------
    assert calc_type == 'opt', 'This function is only for SCF calculation'

    args = get_args(args, calc_type=calc_type)
    name = args['name']
    structure: Atoms = args['structures'][-1]

    F_MAX = args['fmax']
    assert F_MAX > 0, 'F_MAX should be positive'

    N_STEPS = args['nsteps']
    assert N_STEPS > 0, 'N_STEPS should be positive'

    outdir = args['outdir']
    calc_fold = outdir
    is_full_opt = args['full_opt']

    # Printing input parameters
    # ------------------------------------------------
    print(30 * '-')
    if is_full_opt:
        print(f'Full optimization calculation (cell + atomic positions) for {name}')
    else:
        print(f'Optimization of atomic positions for {name}')
    print(f'F_MAX = {F_MAX}, N_STEPS = {N_STEPS}')
    print(f'kspacing = {args["kspacing"]}')
    print(f'Output directory: {outdir}')
    print(30 * '-')

    params = copy(args['dftb_params'])
    params.update(get_calc_type_params(calc_type='scf'))

    # kpts = kptdensity2monkhorstpack(atoms=structure, kptdensity=args['kspacing'])
    kpts = get_KPoints(args['kspacing'], structure.get_cell())
    # Label is used for the output file name
    params.update(
        {
            'label': f'out_{calc_type}_{name}',
            'kpts': kpts,
        }
    )

    structure.calc = Dftb(directory=calc_fold, **params)

    # The object 'opt_struct' will be used for the optimization
    if is_full_opt:
        opt_struct = UnitCellFilter(structure)
    else:
        opt_struct = structure

    opt = BFGS(
        opt_struct,
        restart=str(outdir / 'optimization.pckl'),
        trajectory=str(outdir / 'optimization.traj'),
        logfile=str(outdir / 'optimization.log'),
    )
    opt.run(fmax=F_MAX, steps=N_STEPS)

    # e = structure.get_potential_energy()
    write_vasp(outdir / 'final.vasp', structure, sort=True, vasp5=True, direct=True)

    # ? add 'stress'
    with Trajectory(outdir / 'final.traj', 'w', properties=['energy', 'forces']) as final_traj:
        final_traj.write(structure, energy=structure.get_potential_energy(), forces=structure.get_forces())

    # Final output message
    # ------------------------------------------------
    print(f'Optimization of {name} is done.')
    print(30 * '-')


def dftb_opt(args):
    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type = args.subcommand
    args: dict = vars(args)

    run_opt_dftb(args, calc_type)
