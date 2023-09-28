#!/usr/bin/python3

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from pathlib import Path
from shutil import move

from common.qe import get_args


def qe_opt(args):
    args = get_args()

    # Name of the input file
    name = args['name']
    structures = args['structures']

    options = args['options']
    pp = args['pseudopotentials']
    pp_dir = args['pp_dir']
    kspacing = args['kspacing']
    fmax = args['fmax']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    data = args['data']
    data['control'].update({'outdir': './tmp',
                            'calculation': 'scf'})

    opt_calc = Espresso(input_data=data,
                        pseudopotentials=pp,
                        pseudo_dir=str(pp_dir),
                        kspacing=kspacing,
                        directory=str(calc_fold))

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

        LOGFILE = outdir/f'{ID}.log'
        RES_TRAJ_FILE = outdir/'res_{ID}.traj'

        opt = BFGS(structure, logfile=LOGFILE)
        traj = Trajectory(RES_TRAJ_FILE, 'w', structure)
        opt.attach(traj)
        opt.run(fmax=fmax)

        print(structure.get_potential_energy())
        traj.write(atoms=structure)

        move(calc_fold/f'{opt_calc.prefix}.pwi', outdir/f'{ID}.scf.in')
        move(calc_fold/f'{opt_calc.prefix}.pwo', outdir/f'{ID}.scf.out')

        # copy_calc_files(objects_to_copy, outdir)
        print(f'Optimization of {ID} is done.')
