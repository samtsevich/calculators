#!/usr/bin/python3

from pathlib import Path
from shutil import move

from ase.calculators.espresso import Espresso
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase.optimize import BFGS

from . import get_args


def run_opt_qe(args: dict):
    # Name of the input file
    name = args['name']
    structures = args['structures']

    pp = args['pseudopotentials']
    pp_dir = args['pp_dir']
    kspacing = args['kspacing']
    fmax = args['fmax']
    nsteps = args['nsteps']

    is_full_opt = args['full_opt']

    outdir = Path(args['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    calc_fold = outdir

    data = args['data']
    data['control'].update({'outdir': './tmp', 'calculation': 'scf'})

    opt_calc = Espresso(
        input_data=data, pseudopotentials=pp, pseudo_dir=str(pp_dir), kspacing=kspacing, directory=str(calc_fold)
    )

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        structure.calc = opt_calc

        # add rattling to the atomic positions
        # add_coords = 0.05 - 0.1 * np.random.rand(len(atoms), 3)
        # new_coords = atoms.get_scaled_positions() + add_coords
        # atoms.set_scaled_positions(new_coords)

        # add rattling to the cell
        # add_cell = 0.1 * np.random.rand(3,3)
        # new_cell = atoms.get_cell() + add_cell
        # atoms.set_cell(new_cell, scale_atoms=True)

        # OPTIMIZATION #

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
        opt.run(fmax=fmax, steps=nsteps)

        print(structure.get_potential_energy())
        write_vasp(outdir / f'final.vasp', structure, sort=True, vasp5=True, direct=True)
        with Trajectory(outdir / f'final.traj', 'w') as traj:
            traj.write(structure)

        # move(calc_fold/opt_calc.template.inputname, outdir/f'{ID}.scf.in')
        # move(calc_fold/opt_calc.template.outputname, outdir/f'{ID}.scf.out')

        # copy_calc_files(objects_to_copy, outdir)
        print(f'Optimization of {ID} is done.')


def qe_opt(args):
    assert args.command == 'qe', 'This function is only for QE calculations'
    assert args.subcommand == 'opt', 'This function is only for optimization calculations'
    args: dict = get_args(args)

    run_opt_qe(args)
