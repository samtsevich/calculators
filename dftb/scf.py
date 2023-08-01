from ase.calculators.dftb import Dftb

from ase.io.vasp import write_vasp

from pathlib import Path

from common_dftb import (get_args,
                         get_additional_params)


if __name__ == "__main__":

    calc_type = 'scf'
    args = get_args(calc_type=calc_type)

    name = args['name']
    structure = args['structure']

    outdir = args['outdir']
    calc_fold = outdir

    params = args['dftb_params']
    params.update(get_additional_params(type=calc_type))
    params.update({'label': f'scf_{name}',})

    scf_calc = Dftb(atoms=structure,
                    directory=calc_fold,
                    **params)

    structure.write(outdir/f'a_{name}.gen')
    structure.set_calculator(scf_calc)
    structure.get_potential_energy()
    write_vasp(outdir/f'final_{name}.vasp', structure,
               sort=True, vasp5=True, direct=True)

    print(f'SCF of {name} is done.')
