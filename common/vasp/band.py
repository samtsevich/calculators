#!/usr/bin/python3

from pathlib import Path
from shutil import move

from ase.atoms import Atoms
from ase.calculators.vasp import Vasp
from ase.io.trajectory import Trajectory
from ase.io.vasp import read_vasp_out, write_vasp
from ase.spectrum.band_structure import BandStructure, get_band_structure

from .. import fix_fermi_level
from . import COMMON_VASP_PARAMS, get_args, get_total_N_val_e


def run_band_vasp(args: dict):
    name = args['name']
    structures = args['structures']

    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    general_params = {
        'system': name,  # Job name
        'directory': calc_fold,  # Directory to run VASP
        'kspacing': kspacing,  # k-point grid for SCF
    }
    scf_params = COMMON_VASP_PARAMS.copy()
    scf_params.update(general_params)
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

        # 3. BAND STRUCTURE #
        path = structure.cell.bandpath(npoints=200)
        print(f'BandPath: {path}')

        nscf_params = scf_params.copy()
        del nscf_params['kspacing']
        nscf_params.update(
            {
                'icharg': 11,
                # 'kpts': path.kpts,
                # # Stupid ASE does not recognize k-points for band structures, when there is no 'path' key in the dict
                'kpts': {**path.todict(), 'path': ''},
                # 'lorbit': 11,           # This enables PDOS calculation
                # 'nedos': 3000,          # Increase number of points in DOS
            }
        )

        nscf_calc = Vasp(**nscf_params)
        structure.calc = nscf_calc
        e = structure.get_potential_energy()

        move(calc_fold / 'INCAR', outdir / f'INCAR.band.{ID}')
        move(calc_fold / 'OUTCAR', outdir / f'OUTCAR.band.{ID}')

        print(structure.calc.get_eigenvalues().shape)
        # bs = BandStructure(structure.calc.get_eigenvalues(), path.kpts, reference=structure.calc.get_fermi_level())
        bs = get_band_structure(structure, calc=nscf_calc)

        bs = nscf_calc.band_structure()

        bs = fix_fermi_level(bs, N_val_e).subtract_reference()
        bs.write(outdir / f'bs_{ID}.json')
        bs.plot(filename=outdir / f'bs_{ID}.png')

        print(f'Band structure of {ID} is calculated.')
        print('---------------------------')


def vasp_band(args):
    assert args.command == 'vasp', 'This function is only for VASP'
    args: dict = get_args(args)

    run_band_vasp(args)
