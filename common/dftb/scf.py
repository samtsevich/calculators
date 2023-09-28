from ase.calculators.dftb import Dftb

from ase.io.vasp import write_vasp

from pathlib import Path

from common.dftb import (get_args,
               get_additional_params)

def dftb_scf(args):

    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type = args.subcommand

    args = get_args(args)
    name = args['name']
    structures = args['structures']

    outdir = args['outdir']
    calc_fold = outdir

    params = args['dftb_params']
    params.update(get_additional_params(type=calc_type))

    scf_calc = Dftb(directory=calc_fold,
                    **params)

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        scf_calc.label= f'scf_{ID}'

        structure.write(outdir/f'a_{ID}.gen')
        structure.set_calculator(scf_calc)
        structure.get_potential_energy()
        write_vasp(outdir/f'final_{ID}.vasp', structure,
                   sort=True, vasp5=True, direct=True)
        print(f'SCF of {ID} is done.')


# if __name__ == "__main__":

#     calc_type = 'scf'
#     args = get_args(calc_type=calc_type)
#     dftb_scf(args)
