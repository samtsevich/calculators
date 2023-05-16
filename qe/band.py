#!/usr/bin/python3

import argparse
import numpy as np
import yaml

from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.io import read
from ase.io.espresso import read_fortran_namelist
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.io.vasp import read_vasp
from ase.optimize import BFGS
from ase.units import kJ

from pathlib import Path
from shutil import move, copy2

KSPACING = 0.04

CALC_FILES = ['espresso.pwi', 'espresso.pwo']

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


# def copy_calc_files(objects:list, dest):
#     cp_command = f"cp -r {str(origin/'tmp')} {str(origin/'espresso*')} {dest}"
#     os.system(cp_command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EOS with QE for structure')
    parser.add_argument('-c', dest='config', required=False,
                        help='path to the config file')

    parser.add_argument('-i', dest='input',
                        help='path to the input trajectory file')
    parser.add_argument('-k', dest='options', help='path to the options file')
    parser.add_argument('-pp', dest='pseudopotentials',
                        help='dict of pseudopotentials for ASE')
    parser.add_argument('-o', dest='outdir', help='path to the output folder')
    parser.add_argument('-train', dest='train', action='store_true',
                        help='whether calculation is made for the training of DFTB params from the band structure')
    args = parser.parse_args()

    #####################
    # 1. READING INPUTS #
    #####################

    if args.config:
        assert Path(args.config).exists()
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt
    else:
        args.pseudopotentials = eval(args.pseudopotentials)
        args = vars(args)

    # assert Path(args.pseudopotentials).exists(), 'Please check paths to PP files'
    input = Path(args['input'])
    assert input.exists(), f'Seems like path to the input file is wrong.\n It is {input}'
    outdir = Path(args['outdir']) if args['outdir'] is not None else Path.cwd()/f'res_{input.stem}'
    outdir.mkdir(exist_ok=True)
    options = args['options']
    assert Path(options).exists(), f"Seems like path to the options file is wrong.\n It is {Path(args['options'])}"

    pp = args['pseudopotentials']
    pp_dir = Path(args['pp_dir'])
    assert pp_dir.exists(), 'Seems like folder with pseudopotentials does not exist or wrong'
    pp_dir = pp_dir.resolve()

    calc_fold = outdir

    with open(options) as fp:
        data, card_lines = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')
    data['control']['outdir'] = './tmp'
    data['control']['prefix'] = input.stem
    data['calculation'] = 'scf'

    calc = Espresso(input_data=data,
                    pseudopotentials=pp,
                    pseudo_dir=str(pp_dir),
                    kspacing=KSPACING,
                    directory=str(calc_fold))

    # Read atomic structure from the inputs
    atoms = read_vasp(input)
    # if isinstance(structures, Atoms):
    # structures = [structures]

    for s in list(set(atoms.get_chemical_symbols())):
        assert s in pp.keys(), f'{s} is not presented in the pseudopotentials'

    ##########
    # 2. SCF #
    ##########
    atoms.calc = calc
    e = atoms.get_potential_energy()
    fermi_level = calc.get_fermi_level()
    print('Step 1. SCF calculation is done')

    for x in CALC_FILES:
        move(calc_fold/x, outdir/f'scf{Path(x).suffix}')

    #####################
    # 3. BAND STRUCTURE #
    #####################

    # Update inputs to band structure calc
    data['control'].update({'calculation': 'bands',
                            'restart_mode': 'restart',
                            'verbosity': 'high'})

    path = atoms.cell.bandpath()

    if args['train']:
        path = get_bandpath_for_dftb(
            atoms, {'path': path.path, 'npoints': 101})

    band_calc = Espresso(input_data=data,
                         pseudopotentials=pp,
                         pseudo_dir=str(pp_dir),
                         kpts=path,
                         directory=str(calc_fold))

    # calc.set(kpts=path, input_data=data)
    band_calc.calculate(atoms)

    for x in CALC_FILES:
        move(calc_fold/x, outdir/f'band{Path(x).suffix}')

    bs = band_calc.band_structure()
    bs.subtract_reference()
    # bs.reference = fermi_level
    bs.write(outdir/f'bs_{input.stem}.json')

    # 2. OPTIMIZATION

    # LOGFILE = outdir/'log'
    # RES_TRAJ_FILE = outdir/'res.traj'
    #
    # opt = BFGS(atoms, logfile=LOGFILE)
    # traj = Trajectory(RES_TRAJ_FILE, 'w', atoms)
    # opt.attach(traj)
    # opt.run(fmax=0.01)
    #
    # print(atoms.get_potential_energy())
    # traj.write(atoms=atoms)
    #
    # objects_to_move = ['espresso.pwi', 'espresso.pwo']
    # for file in objects_to_move:

    # copy_calc_files(objects_to_copy, outdir)
    print(f'Band structure of {input} is calcultaed.')
