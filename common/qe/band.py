#!/usr/bin/python3

from pathlib import Path
from shutil import move

import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.espresso import Espresso
from ase.spectrum.band_structure import get_band_structure

from common import fix_fermi_level
from common.qe import get_args, read_valences


def get_bandpath_for_dftb(atoms, kpts, pbc=[True, True, True]):
    """This function sets up the band path according to Setyawan-Curtarolo conventions.

    Parameters:
    -----------
    atoms: ase.Atoms object
        The molecule or crystal structure.
    kpts: int
        The number of k-points among two special kpoint positions.

    Returns:
        list: List of strings containing the k-path sections.
    """
    from ase.dft.kpoints import kpoint_convert, parse_path_string

    # path = parse_path_string(
    #     atoms.cell.get_bravais_lattice(pbc=atoms.pbc).bandpath().path
    # )
    path = parse_path_string(kpts['path'])
    # list Of lists of path segments
    points = atoms.cell.get_bravais_lattice(
        pbc=atoms.pbc).bandpath().special_points
    segments = []
    for seg in path:
        section = [(i, j) for i, j in zip(seg[:-1], seg[1:])]
        segments.append(section)
    output_bands = []
    output_bands = np.empty(shape=(0, 3))
    index = kpts['npoints']
    for seg in segments:
        # output_bands.append("## Brillouin Zone section Nr. {:d}\n".format(index))
        for num, sec in enumerate(seg):
            dist = np.array(points[sec[1]]) - np.array(points[sec[0]])
            npoints = index
            if num == 0:
                dist_matrix = np.linspace(
                    points[sec[0]], points[sec[1]], npoints)
            else:
                dist_matrix = np.linspace(
                    points[sec[0]], points[sec[1]], npoints)[1:, :]
            output_bands = np.vstack((output_bands, dist_matrix))
    return {'path': path, 'kpts': output_bands}


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
    data['control'].update({'outdir': './tmp',
                            'prefix': str(name),
                            'verbosity': 'high'})

    calc: Calculator = Espresso(input_data=data,
                                pseudopotentials=pp,
                                pseudo_dir=str(pp_dir),
                                kspacing=kspacing,
                                directory=str(calc_fold))

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

        move(calc_fold/calc.template.inputname, outdir/f'{ID}.scf.in')
        move(calc_fold/calc.template.outputname, outdir/f'{ID}.scf.out')

        # BAND STRUCTURE #

        # Update inputs to band structure calc
        data['control'].update({'calculation': 'bands',
                                'restart_mode': 'restart',
                                'verbosity': 'high'})

        path = structure.cell.bandpath(npoints=200)
        print(f'BandPath: {path}')

        if args['is_training']:
            path = get_bandpath_for_dftb(
                structure, {'path': path.path, 'npoints': 101})

        calc: Calculator = Espresso(input_data=data,
                                    pseudopotentials=pp,
                                    pseudo_dir=str(pp_dir),
                                    kpts=path,
                                    directory=str(calc_fold))
        # calc.set(kpts=path, input_data=data)
        # calc.calculate(atoms=structure)
        structure.calc = calc
        structure.get_potential_energy()

        move(calc_fold/calc.template.inputname, outdir/f'{ID}.band.in')
        move(calc_fold/calc.template.outputname, outdir/f'{ID}.band.out')

        bs = get_band_structure(atoms=structure, calc=calc)
        bs = fix_fermi_level(bs, N_val_e).subtract_reference()
        bs.write(outdir/f'bs_{ID}.json')
        bs.plot(filename=outdir/f'bs_{ID}.png')

        print(f'Band structure of {ID} is calculated.')
        print('---------------------------')
