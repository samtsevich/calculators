from ase.calculators.dftb import Dftb

from ase.io.vasp import write_vasp

from pathlib import Path

from common_dftb import (get_args,
                         get_additional_params)


if __name__ == "__main__":

    args = get_args(calc_type='opt')

    name = args['name']
    structure = args['structure']

    outdir = args['outdir']
    calc_fold = outdir

    params = args['dftb_params']
    params.update(get_additional_params(type='opt'))
    params.update({'label': f'opt_{name}',})

    opt_calc = Dftb(atoms=structure,
                    directory=calc_fold,
                    **params)

    structure.write(outdir/f'a_{name}.gen')
    structure.set_calculator(opt_calc)
    structure.get_potential_energy()
    write_vasp(outdir/f'final_{name}.vasp', structure,
               sort=True, vasp5=True, direct=True)

    print('Optimization done!')
