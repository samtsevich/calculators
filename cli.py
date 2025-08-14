import argparse

import ase


def _create_lazy_callable(calculator, subcommand):
    """Create a lazy callable that imports and runs the appropriate calculator function."""
    def lazy_wrapper(args):
        if calculator == 'qe':
            if subcommand == 'opt':
                from common.qe.opt import qe_opt
                return qe_opt(args)
            elif subcommand == 'scf':
                from common.qe.scf import qe_scf
                return qe_scf(args)
            elif subcommand == 'band':
                from common.qe.band import qe_band
                return qe_band(args)
            elif subcommand == 'eos':
                from common.qe.eos import qe_eos
                return qe_eos(args)
            elif subcommand == 'pdos':
                from common.qe.pdos import qe_pdos
                return qe_pdos(args)

        elif calculator == 'dftb':
            if subcommand == 'opt':
                from common.dftb.opt import dftb_opt
                return dftb_opt(args)
            elif subcommand == 'scf':
                from common.dftb.scf import dftb_scf
                return dftb_scf(args)
            elif subcommand == 'band':
                from common.dftb.band import dftb_band
                return dftb_band(args)
            elif subcommand == 'eos':
                from common.dftb.eos import dftb_eos
                return dftb_eos(args)
            elif subcommand == 'neb':
                from common.dftb.neb import dftb_neb
                return dftb_neb(args)

        elif calculator == 'mace':
            if subcommand == 'opt':
                from common.mace.opt import mace_opt
                return mace_opt(args)
            elif subcommand == 'scf':
                from common.mace.scf import mace_scf
                return mace_scf(args)
            elif subcommand == 'eos':
                from common.mace.eos import mace_eos
                return mace_eos(args)

        elif calculator == 'vasp':
            if subcommand == 'scf':
                from common.vasp.scf import vasp_scf
                return vasp_scf(args)
            elif subcommand == 'band':
                from common.vasp.band import vasp_band
                return vasp_band(args)
            elif subcommand == 'pdos':
                from common.vasp.pdos import vasp_pdos
                return vasp_pdos(args)

        raise ValueError(f"Unknown calculator '{calculator}' or subcommand '{subcommand}'")

    return lazy_wrapper


def _precheck():
    '''
    Raise an error if a version of package is too low
    '''
    if ase.__version__ < '3.23.0':
        raise ValueError('ASE version must be at least 3.23.0')


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    # Create a subparser for the 'dftb' command
    parser_dftb = subparsers.add_parser('dftb')
    dftb_subparsers = parser_dftb.add_subparsers(dest='subcommand')
    dftb_calc_types = ['opt', 'scf', 'band', 'eos', 'neb']
    for calc_type in dftb_calc_types:
        from common.dftb import add_dftb_arguments
        dftb_subparser = dftb_subparsers.add_parser(calc_type)
        dftb_subparser = add_dftb_arguments(dftb_subparser, calc_type)
        dftb_subparser.set_defaults(func=_create_lazy_callable('dftb', calc_type))

    # Create a subparser for the 'mace' command
    parser_mace = subparsers.add_parser('mace')
    mace_subparsers = parser_mace.add_subparsers(dest='subcommand')
    mace_calc_types = ['opt', 'scf', 'eos']
    for calc_type in mace_calc_types:
        from common.mace import add_mace_arguments
        mace_subparser = mace_subparsers.add_parser(calc_type)
        mace_subparser = add_mace_arguments(mace_subparser, calc_type)
        mace_subparser.set_defaults(func=_create_lazy_callable('mace', calc_type))

    # Create a subparser for the 'qe' command
    parser_qe = subparsers.add_parser('qe')
    qe_subparsers = parser_qe.add_subparsers(dest='subcommand')
    qe_calc_types = ['opt', 'scf', 'band', 'eos', 'pdos']
    for calc_type in qe_calc_types:
        from common.qe import add_qe_arguments
        qe_subparser = qe_subparsers.add_parser(calc_type)
        qe_subparser = add_qe_arguments(qe_subparser, calc_type)
        qe_subparser.set_defaults(func=_create_lazy_callable('qe', calc_type))

    # Create a subparser for the 'vasp' command
    parser_vasp = subparsers.add_parser('vasp')
    vasp_subparsers = parser_vasp.add_subparsers(dest='subcommand')
    vasp_calc_types = ['scf', 'band', 'pdos']
    for calc_type in vasp_calc_types:
        from common.vasp import add_vasp_arguments
        vasp_subparser = vasp_subparsers.add_parser(calc_type)
        vasp_subparser = add_vasp_arguments(vasp_subparser, calc_type)
        vasp_subparser.set_defaults(func=_create_lazy_callable('vasp', calc_type))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _precheck()
    main()
