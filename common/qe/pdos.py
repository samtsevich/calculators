#!/usr/bin/python3

from pathlib import Path
from shutil import move

import numpy as np
import os
from ase.calculators.calculator import Calculator
from ase.calculators.espresso import Espresso
from ase.spectrum.band_structure import get_band_structure
from ase.units import Ry

from .. import fix_fermi_level
from . import get_args, read_valences


def qe_pdos(args):
    print('---------------------------')
    calc_type = args.subcommand
    args = get_args(args)

    name = args['name']
    structures = args['structures']

    pp = args['pseudopotentials']
    pp_dir = args['pp_dir']
    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    data = args['data']
    data['calculation'] = 'scf'
    qe_outdir = './tmp'
    data['control'].update({'outdir': qe_outdir, 'prefix': str(name), 'verbosity': 'high'})
    data['system'].update({'nosym': True})

    calc: Calculator = Espresso(
        input_data=data,
        pseudopotentials=pp,
        pseudo_dir=str(pp_dir),
        kspacing=kspacing,
        directory=str(calc_fold)
    )

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        # 1. SCF #
        ##########
        structure.calc = calc
        e = structure.get_potential_energy()
        fermi_level = calc.get_fermi_level()
        print('Step 1. SCF calculation is done')

        valences = read_valences(calc_fold / calc.template.outputname)
        N_val_e = sum([valences[symbol] for symbol in structure.get_chemical_symbols()])
        print(f'Total N valence electrons: {N_val_e}')
        print(f'Fermi level: {fermi_level}')
        print('---------------------------')

        move(calc_fold / calc.template.inputname, outdir / f'{ID}.scf.in')
        move(calc_fold / calc.template.outputname, outdir / f'{ID}.scf.out')

        #      2. NSCF      #
        #####################
        # Update inputs to band structure calc
        data['control'].update({'calculation': 'nscf', 'verbosity': 'high'})

        calc: Calculator = Espresso(
            input_data=data,
            pseudopotentials=pp,
            pseudo_dir=str(pp_dir),
            kspacing=kspacing,
            directory=str(calc_fold)
        )
        structure.calc = calc
        structure.get_potential_energy()

        move(calc_fold / calc.template.inputname, outdir / f'{ID}.nscf.in')
        move(calc_fold / calc.template.outputname, outdir / f'{ID}.nscf.out')

        #      3. PDOS      #
        #####################

        emin, emax = -25.0 + fermi_level, 20.0 + fermi_level
        emin, emax = np.round(emin, 3), np.round(emax, 3)
        # PDOS calculation
        input_dos_file = outdir / f'{ID}.pdos.in'
        output_dos_file = outdir / f'{ID}.pdos.out'
        output_dos_data = outdir / f'{ID}_pdos.dat'
        with open(input_dos_file, 'w') as f:
            f.write(f"&PROJWFC\n")
            f.write(f"  prefix = '{str(name)}',\n")
            f.write(f"degauss={args['smearing'] / Ry},\n")
            f.write(f"outdir='{qe_outdir}',\n")
            f.write(f"filpdos='{output_dos_data.name}',\n")
            f.write(f'emin={emin},\n')
            f.write(f'emax={emax},\n')
            f.write(f"/\n")

        exec_command = f'projwfc.x < {input_dos_file.name} > {output_dos_file.name}'
        os.system(f'cd {calc_fold}; ' + exec_command)

        print(f'PDOSes of {ID} are calculated.')
        print('-------------------------------')
