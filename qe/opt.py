#!/usr/bin/python3

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

from pathlib import Path
from shutil import move

from common_qe import get_args


if __name__ == '__main__':
    args = get_args(calc_type='opt')

    # Name of the input file
    name = args['input'].stem

    structure = args['structure']

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

    structure.calc = opt_calc

    # add rattling to the atomic positions
    # add_coords = 0.05 - 0.1 * np.random.rand(len(atoms), 3)
    # new_coords = atoms.get_scaled_positions() + add_coords
    # atoms.set_scaled_positions(new_coords)

    # add rattling to the cell
    # add_cell = 0.1 * np.random.rand(3,3)
    # new_cell = atoms.get_cell() + add_cell
    # atoms.set_cell(new_cell, scale_atoms=True)

    ###################
    # 2. OPTIMIZATION #
    ###################

    LOGFILE = outdir/'log'
    RES_TRAJ_FILE = outdir/'res.traj'

    opt = BFGS(structure, logfile=LOGFILE)
    traj = Trajectory(RES_TRAJ_FILE, 'w', structure)
    opt.attach(traj)
    opt.run(fmax=fmax)

    print(structure.get_potential_energy())
    traj.write(atoms=structure)

    move(calc_fold/opt_calc.template.inputname, outdir/f'{name}.opt.in')
    move(calc_fold/opt_calc.template.outputname, outdir/f'{name}.opt.out')

    # copy_calc_files(objects_to_copy, outdir)
    print(f'Optimization of {args["input"]} is done.')
