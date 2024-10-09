#!/usr/bin/python3

from pathlib import Path
from shutil import move

import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.espresso import Espresso
from ase.spectrum.band_structure import get_band_structure

from .. import fix_fermi_level
from . import get_args, read_valences


def qe_band(args):
    print('---------------------------')
    calc_type = args.subcommand
    args = get_args(args)

    name = args['name']
    structures = args['structures']

    options = args['options']
    pp = args['pseudopotentials']
    pp_dir = args['pp_dir']
    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    data = args['data']
    data['calculation'] = 'scf'
    data['control'].update({'outdir': './tmp', 'prefix': str(name), 'verbosity': 'high'})

    calc: Calculator = Espresso(
        input_data=data, pseudopotentials=pp, pseudo_dir=str(pp_dir), kspacing=kspacing, directory=str(calc_fold)
    )

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        # SCF #
        structure.calc = calc
        e = structure.get_potential_energy()
        fermi_level = calc.get_fermi_level()
        print('Step 1. SCF calculation is done')

        valences = read_valences(calc_fold / calc.template.outputname)
        N_val_e = sum([valences[symbol] for symbol in structure.get_chemical_symbols()])
        print(f'Total N valence electrons: {N_val_e}')

        move(calc_fold / calc.template.inputname, outdir / f'{ID}.scf.in')
        move(calc_fold / calc.template.outputname, outdir / f'{ID}.scf.out')

        # BAND STRUCTURE #

        # Update inputs to band structure calc
        data['control'].update({'calculation': 'bands', 'restart_mode': 'restart', 'verbosity': 'high'})

        path = structure.cell.bandpath(npoints=200)
        print(f'BandPath: {path}')

        calc: Calculator = Espresso(
            input_data=data, pseudopotentials=pp, pseudo_dir=str(pp_dir), kpts=path, directory=str(calc_fold)
        )
        # calc.set(kpts=path, input_data=data)
        # calc.calculate(atoms=structure)
        structure.calc = calc
        structure.get_potential_energy()

        move(calc_fold / calc.template.inputname, outdir / f'{ID}.band.in')
        move(calc_fold / calc.template.outputname, outdir / f'{ID}.band.out')

        bs = get_band_structure(atoms=structure, calc=calc)
        bs = fix_fermi_level(bs, N_val_e).subtract_reference()
        bs.write(outdir / f'bs_{ID}.json')
        bs.plot(filename=outdir / f'bs_{ID}.png')

        print(f'Band structure of {ID} is calculated.')
        print('---------------------------')
