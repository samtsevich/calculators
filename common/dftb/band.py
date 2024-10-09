from copy import copy
from pathlib import Path
from typing import Dict, List

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.calculators.dftb import Dftb
from ase.io.vasp import write_vasp
from ase.spectrum.band_structure import BandStructure, get_band_structure
from matplotlib import pyplot as plt

from .. import fix_fermi_level, get_KPoints, get_N_val_electrons
from . import get_args, get_calc_type_params

N_KPOINTS = 200


def run_dftb_band(args: dict, calc_type: str):
    # Reading the arguments
    # ------------------------------------------------
    assert calc_type == 'band', 'This function is only for band structure calculation'

    args = get_args(args, calc_type=calc_type)
    name = args['name']
    structures: List[Atoms] = args['structures']
    structure = structures[-1]
    emin = args['emin']
    emax = args['emax']
    assert emin < emax, 'E_min should be less than E_max'

    outdir: Path = args['outdir']
    outdir.mkdir(parents=True, exist_ok=True)

    # Printing input parameters
    # ------------------------------------------------
    print(30 * '-')
    print(f'Band structure calculation for {name}')
    print(f'kspacing = {args["kspacing"]}')
    print(f'E_min = {emin}, E_max = {emax}')
    print(f'Output directory: {outdir}')
    print(30 * '-')

    # Step 1. SCF calculation
    # ------------------------------------------------
    params = copy(args['dftb_params'])
    params.update(get_calc_type_params(calc_type='scf'))
    kpts = get_KPoints(kspacing=args['kspacing'], cell=structure.cell)
    # kpts = kptdensity2monkhorstpack(atoms=structure, kptdensity=args['kspacing'])
    params.update(
        {
            'label': f'out_scf_{name}',
            'kpts': kpts,
        }
    )

    calc_fold = outdir

    write_vasp(calc_fold / f'a_{name}.vasp', structure, sort=True, vasp5=True, direct=True)

    scf_calc = Dftb(atoms=structure, directory=calc_fold, **params)
    structure.calc = scf_calc
    e = structure.get_potential_energy()
    fermi_level = scf_calc.get_fermi_level()

    print(f'\tStep 1 for {name} done')

    # Step 2. Band structure calculation
    # ------------------------------------------------
    path = structure.cell.bandpath(npoints=N_KPOINTS)
    print(path)
    params.update(get_calc_type_params(calc_type='band'))
    params.update(
        {
            'label': f'out_band_{name}',
        }
    )

    assert params['Hamiltonian_SCC'] == 'Yes'
    assert params['Hamiltonian_MaxSCCIterations'] == 1

    # Stupid ASE does not recognize k-points for band structures, when there is no 'path' key in the dict
    params.update({'kpts': {**path.todict(), 'path': ''}})

    band_calc = Dftb(atoms=structure, directory=calc_fold, **params)
    band_calc.calculate(structure)
    bs = get_band_structure(atoms=structure, calc=band_calc)

    print(f'Fermi level = {bs._reference}')

    # Fix Fermi level
    N_val_e = get_N_val_electrons(structure)
    bs = fix_fermi_level(bs, N_val_e).subtract_reference()
    bs.write(outdir / f'bs_{name}.json')
    bs.plot(filename=outdir / f'bs_{name}.png', emin=emin, emax=emax)

    # Final output message
    # ------------------------------------------------
    print(f'\tStep 2 for {name} done')
    print(30 * '-')


def dftb_band(args):
    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type = args.subcommand
    args: dict = vars(args)
    run_dftb_band(args, calc_type)
