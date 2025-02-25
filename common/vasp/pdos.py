#!/usr/bin/python3

from pathlib import Path
from shutil import move

from ase.calculators.vasp import Vasp
from ase.io.vasp import read_vasp_out, write_vasp

from .. import fix_fermi_level
from . import get_basic_params, get_args, get_total_N_val_e


def run_pdos_vasp(args: dict):
    name = args['name']
    structures = args['structures']

    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    scf_params = get_basic_params(args)
    scf_params.update({'directory': calc_fold})

    scf_calc = Vasp(**scf_params)

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        # 2. SCF #
        structure.calc = scf_calc
        e = structure.get_potential_energy()
        write_vasp(outdir / f'final_{ID}.vasp', structure, sort=True, vasp5=True, direct=True)

        N_val_e: int = get_total_N_val_e(calc_fold / 'OUTCAR')
        assert N_val_e is not None
        print(f'Total N valence electrons: {N_val_e}')

        move(calc_fold / 'INCAR', outdir / f'INCAR.scf.{ID}')
        move(calc_fold / 'OUTCAR', outdir / f'OUTCAR.scf.{ID}')

        # 3. (P)DOS #

        nscf_params = scf_params.copy()
        nscf_params.update(
            {
                'nbands': 2*N_val_e,    # Number of bands (here increased for PDOS)
                'icharg': 11,           # Charge density from CHGCAR
                'ibrion': -1,           # no ion relaxation
                'lorbit': 11,           # This enables PDOS calculation
                'nedos': 3000,          # Increase number of points in DOS
            }
        )

        nscf_calc = Vasp(**nscf_params)
        structure.calc = nscf_calc
        e = structure.get_potential_energy()

        move(calc_fold / 'INCAR', outdir / f'INCAR.pdos.{ID}')
        move(calc_fold / 'OUTCAR', outdir / f'OUTCAR.pdos.{ID}')

        print(f'PDOSes of {ID} are calculated.')
        print('-------------------------------')


def vasp_pdos(args):
    assert args.command == 'vasp', 'This function is only for VASP'
    args: dict = get_args(args)

    run_pdos_vasp(args)
