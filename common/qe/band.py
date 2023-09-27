#!/usr/bin/python3

import numpy as np

from ase.calculators.espresso import Espresso

from pathlib import Path
from shutil import move

from common.qe import get_args


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
    calc_type = args.subcommand
    args = get_args(args)

    name = args['name']
    structure = args['structure']

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

    calc = Espresso(input_data=data,
                    pseudopotentials=pp,
                    pseudo_dir=str(pp_dir),
                    kspacing=kspacing,
                    directory=str(calc_fold))

    ##########
    # 2. SCF #
    ##########
    structure.calc = calc
    e = structure.get_potential_energy()
    fermi_level = calc.get_fermi_level()
    print('Step 1. SCF calculation is done')

    move(calc_fold/f'{calc.prefix}.pwi', outdir/f'{name}.scf.in')
    move(calc_fold/f'{calc.prefix}.pwo', outdir/f'{name}.scf.out')

    #####################
    # 3. BAND STRUCTURE #
    #####################

    # Update inputs to band structure calc
    data['control'].update({'calculation': 'bands',
                            'restart_mode': 'restart',
                            'verbosity': 'high'})

    path = structure.cell.bandpath(npoints=200)
    print(f'BandPath: {path}')

    if args['is_training']:
        path = get_bandpath_for_dftb(
            structure, {'path': path.path, 'npoints': 101})

    calc = Espresso(input_data=data,
                    pseudopotentials=pp,
                    pseudo_dir=str(pp_dir),
                    kpts=path,
                    directory=str(calc_fold))
    # calc.set(kpts=path, input_data=data)
    calc.calculate(structure)

    move(calc_fold/f'{calc.prefix}.pwi', outdir/f'{name}.band.in')
    move(calc_fold/f'{calc.prefix}.pwo', outdir/f'{name}.band.out')

    bs = calc.band_structure()
    bs._reference = fermi_level
    bs.subtract_reference()
    bs.write(outdir/f'bs_{name}.json')
    bs.plot(filename=outdir/f'bs_{name}.png')

    print(f'Band structure of {name} is calculated.')

