#!/usr/bin/python3

from ase.calculators.espresso import Espresso
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp

from pathlib import Path
from shutil import move

from common.qe import get_args


def qe_scf(args):
    args = get_args(args)

    name = args['name']
    structures = args['structures']

    options = args['options']
    pp = args['pseudopotentials']
    pp_dir = args['pp_dir']
    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    data = args['data']
    data['control'].update({'calculation': 'scf',
                            'outdir': './tmp',
                            'verbosity': 'high',
                            'wf_collect': True})


    traj = Trajectory(outdir/f'traj_{name}.traj', 'w', properties=['energy', 'forces'])

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        data['control']['prefix'] = f'{ID}.scf'

        scf_calc = Espresso(input_data=data,
                            pseudopotentials=pp,
                            pseudo_dir=str(pp_dir),
                            kspacing=kspacing,
                            directory=str(calc_fold))

        structure.set_calculator(scf_calc)

        # add rattling to the atomic positions
        # add_coords = 0.05 - 0.1 * np.random.rand(len(atoms), 3)
        # new_coords = atoms.get_scaled_positions() + add_coords
        # atoms.set_scaled_positions(new_coords)

        # add rattling to the cell
        # add_cell = 0.1 * np.random.rand(3,3)
        # new_cell = atoms.get_cell() + add_cell
        # atoms.set_cell(new_cell, scale_atoms=True)

        # 2. SCF #
        e = structure.get_potential_energy()
        write_vasp(outdir/f'final_{ID}.vasp', structure,
                   sort=True, vasp5=True, direct=True)

        move(calc_fold/f'{scf_calc.prefix}.pwi', outdir/f'{ID}.scf.in')
        move(calc_fold/f'{scf_calc.prefix}.pwo', outdir/f'{ID}.scf.out')

        # move(calc_fold/scf_calc.template.inputname, outdir/f'{name}.scf.in')
        # move(calc_fold/scf_calc.template.outputname, outdir/f'{name}.scf.out')

        traj.write(structure)

        print(f'SCF of {ID} is done.')

    traj.close()
