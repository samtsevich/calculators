from ase.calculators.dftb import Dftb

from ase.io.vasp import write_vasp

from common.dftb import (get_args,
                         get_additional_params)

def dftb_opt(args):
    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type = args.subcommand

    args = get_args(args)
    name = args['name']
    structure = args['structure']

    outdir = args['outdir']
    calc_fold = outdir

    params = args['dftb_params']
    params.update(get_additional_params(type=calc_type))
    params.update({'label': f'{calc_type}_{name}',})

    opt_calc = Dftb(atoms=structure,
                    directory=calc_fold,
                    **params)

    structure.write(outdir/f'a_{name}.gen')
    structure.set_calculator(opt_calc)
    structure.get_potential_energy()
    write_vasp(outdir/f'final_{name}.vasp', structure,
               sort=True, vasp5=True, direct=True)

    print('Optimization done!')


# if __name__ == "__main__":
#     args = get_args(calc_type='opt')
#     dftb_opt(args)
