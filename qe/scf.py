#!/usr/bin/python3

from ase.calculators.espresso import Espresso

from pathlib import Path
from shutil import move

from common_qe import (get_args,
                       KSPACING)


if __name__ == '__main__':
    import ase
    assert ase.__version__ > '3.22.1'

    args = get_args(calc_type='scf')

    name = args['input'].stem
    structure = args['structure']

    options = args['options']
    pp = args['pseudopotentials']
    pp_dir = args['pp_dir']
    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    data = args['data']
    data['control'].update({'calculation': 'scf',
                            'outdir': './tmp',
                            'prefix': str(name),
                            'verbosity': 'high',
                            'wf_collect': True})

    scf_calc = Espresso(input_data=data,
                        pseudopotentials=pp,
                        pseudo_dir=str(pp_dir),
                        kspacing=kspacing,
                        directory=str(calc_fold))

    structure.calc = scf_calc

    # add rattling to the atomic positions
    # add_coords = 0.05 - 0.1 * np.random.rand(len(atoms), 3)
    # new_coords = atoms.get_scaled_positions() + add_coords
    # atoms.set_scaled_positions(new_coords)

    # add rattling to the cell
    # add_cell = 0.1 * np.random.rand(3,3)
    # new_cell = atoms.get_cell() + add_cell
    # atoms.set_cell(new_cell, scale_atoms=True)

    ##########
    # 2. SCF #
    ##########

    print(structure.get_potential_energy())

    move(calc_fold/scf_calc.template.inputname, outdir/f'{name}.scf.in')
    move(calc_fold/scf_calc.template.outputname, outdir/f'{name}.scf.out')

    print(f'SCF of {name} is done.')
