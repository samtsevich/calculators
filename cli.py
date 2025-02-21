import argparse

import ase

from common.dftb import add_dftb_arguments
from common.dftb.band import dftb_band
from common.dftb.eos import dftb_eos
from common.dftb.neb import dftb_neb
from common.dftb.opt import dftb_opt
from common.dftb.scf import dftb_scf
from common.qe import add_qe_arguments
from common.qe.band import qe_band
from common.qe.eos import qe_eos
from common.qe.opt import qe_opt
from common.qe.pdos import qe_pdos
from common.qe.scf import qe_scf

from common.vasp import add_vasp_arguments
from common.vasp.scf import vasp_scf
from common.vasp.band import vasp_band
from common.vasp.pdos import vasp_pdos

QE_CALC_TYPES = {'opt': qe_opt, 'scf': qe_scf, 'band': qe_band, 'eos': qe_eos, 'pdos': qe_pdos}

DFTB_CALC_TYPES = {'opt': dftb_opt, 'scf': dftb_scf, 'band': dftb_band, 'eos': dftb_eos, 'neb': dftb_neb}

VASP_CALC_TYPES = {'scf': vasp_scf,
                   'band': vasp_band,
                   'pdos': vasp_pdos}


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
    for calc_type, calc_func in DFTB_CALC_TYPES.items():
        dftb_subparser = dftb_subparsers.add_parser(calc_type)
        dftb_subparser = add_dftb_arguments(dftb_subparser, calc_type)
        dftb_subparser.set_defaults(func=calc_func)

    # Create a subparser for the 'qe' command
    parser_qe = subparsers.add_parser('qe')
    qe_subparsers = parser_qe.add_subparsers(dest='subcommand')
    for calc_type, calc_func in QE_CALC_TYPES.items():
        qe_subparser = qe_subparsers.add_parser(calc_type)
        qe_subparser = add_qe_arguments(qe_subparser, calc_type)
        qe_subparser.set_defaults(func=calc_func)

    # Create a subparser for the 'vasp' command
    parser_vasp = subparsers.add_parser('vasp')
    vasp_subparsers = parser_vasp.add_subparsers(dest='subcommand')
    for calc_type, calc_func in VASP_CALC_TYPES.items():
        vasp_subparser = vasp_subparsers.add_parser(calc_type)
        vasp_subparser = add_vasp_arguments(vasp_subparser, calc_type)
        vasp_subparser.set_defaults(func=calc_func)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    _precheck()
    main()
