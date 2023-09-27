from ase.calculators.dftb import Dftb

from ase.io import read, write, Trajectory
from ase.io.vasp import write_vasp, read_vasp

from pathlib import Path
from shutil import move

import argparse
import numpy as np
import os
import sys


KSPACING = 0.02
DFTB_COMMAND = 'dftb+'


def get_additional_params(species, type: str = 'opt'):
    '''
    type: 'band' or 'static' or 'opt_geometry'
    '''

    params = {}

    if type == 'scf':
        params.update({'Hamiltonian_MaxSCCIterations': 100,
                       'Hamiltonian_ReadInitialCharges': 'No',  # Static calculation
                       })
    elif type == 'band':
        params.update({'Hamiltonian_MaxSCCIterations': 1,
                       'Hamiltonian_ReadInitialCharges': 'Yes',
                       })
    elif type == 'opt':
        params.update({'Driver_': 'LBFGS',
                       'Driver_MaxForceComponent': 1e-4,
                       'Driver_MaxSteps': 1000,
                       'Hamiltonian_MaxSCCIterations': 100,
                       })
    else:
        raise ValueError('type must be band or static or opt_geometry')
    return params


def get_KPoints(kspacing: float, cell):
    assert kspacing > 0
    angLattice = cell.cellpar()
    dist = np.zeros(3)
    dist[2] = cell.volume / (angLattice[0] * angLattice[1]
                             * np.sin(angLattice[5]*np.pi/180))
    dist[1] = cell.volume / (angLattice[0] * angLattice[2]
                             * np.sin(angLattice[4]*np.pi/180))
    dist[0] = cell.volume / (angLattice[1] * angLattice[2]
                             * np.sin(angLattice[3]*np.pi/180))

    Kpoints = [int(x) for x in np.ceil(1.0 / (dist * kspacing))]
    return Kpoints


# path to .skf files (parameters)
# d='/nexus/posix0/FHI-Theory/chiarapan/shared/ligra/fundamentals/dftb-parameters/sp-s_norep/'


GENERAL_PARAMS = {
    'CalculateForces': 'YES',
    'Hamiltonian_': 'DFTB',
    'Hamiltonian_Filling_': 'Fermi',
    'Hamiltonian_Filling_Temperature': 0.0001,  # T in atomic units
    'Hamiltonian_ForceEvaluation': 'dynamics',
    'Hamiltonian_MaxSCCIterations': 500,
    # filling and mixer flags are optional
    'Hamiltonian_Mixer_': 'Broyden',
    'Hamiltonian_Mixer_MixingParameter': 0.1,
    # 'Hamiltonian_Dispersion' : 'LennardJones{Parameters = UFFParameters{}}',	## no dispersion
    'Hamiltonian_PolynomialRepulsive': 'SetForAll {YES}',
    'Hamiltonian_SCC': 'YES',
    'Hamiltonian_SCCTolerance': 1.0E-8,

    'Options_': '',
    'Options_WriteResultsTag': 'Yes',        # Default:No
    'Options_WriteDetailedOut': 'Yes'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Forces calculation by DFTB for Gprep usage')
    parser.add_argument("-i", dest="input",
                        help="path to the file with input structures in POSCAR format")
    parser.add_argument("-s", dest="dir_skf", help='path to .skf files folder')
    parser.add_argument("-o", dest="res", default='res',
                        help=' outtup dir of the calculation')
    args = parser.parse_args()

    input_traj_path = Path(args.input)
    assert input_traj_path.exists()

    dir_skf = Path.cwd()/args.dir_skf
    assert dir_skf.is_dir()

    # assert args.type in ['scc', 'band', 'opt'], 'Seems like you chose wrong type of the calculation'

    curr_fold = Path.cwd()
    dftb_in_files = ['dftb_in.hsd', 'dftb_pin.hsd']
    res_fold = curr_fold/args.res
    res_fold.mkdir(exist_ok=True)

    # traj of structures

    atoms = read_vasp(input_traj_path)
    # output traj with dftb forces

    name = input_traj_path.stem

    params = GENERAL_PARAMS.copy()
    species = set(atoms.get_chemical_symbols())
    additional_params = {
        'slako_dir': str(dir_skf)+'/',
        'Hamiltonian_SlaterKosterFiles_Prefix': str(dir_skf)+'/',
        'Hamiltonian_SlaterKosterFiles_Separator': '"-"',
        'Hamiltonian_SlaterKosterFiles_Suffix': '".skf"',
        # 'run_manyDftb_steps': False,
        # 'Hamiltonian_MaxAngularMomentum_C': 'p',
        # 'Hamiltonian_MaxAngularMomentum_Li': 's',
    }

    additional_params.update(get_additional_params(species, type='scf'))
    params.update(additional_params)

    os.chdir(res_fold)
    print(f'\tFolder has been changed to {res_fold}')

    # path to dftb+ executable > output_file.out
    dftb_command = f'{DFTB_COMMAND} > dftb_{name}.out'  # cluster

    params.update({
        'label': f'{name}',
        'command': dftb_command,
        'kpts': get_KPoints(KSPACING, atoms.get_cell()),
    })

    calc = Dftb(**params)

    atoms.write(f'a_{name}.gen')
    write_vasp(f'a_{name}.vasp', atoms, sort=True, vasp5=True, direct=True)
    atoms.set_calculator(calc)

    e = atoms.get_potential_energy()
    fermi_level = calc.get_fermi_level()

    print(f'\tStep 1 for {name} done')

    # for name in dftb_in_files:
    #     move(calc_fold/name, calc_fold/f'{name}_1')

    # Step 2.
    path = atoms.cell.bandpath()

    calc.calculate(atoms)

    params.update(get_additional_params(species, type='band'))
    params.update({'kpts': {**path.todict(), 'path': ''}})

    band_calc = Dftb(atoms=atoms, **params)
    band_calc.calculate(atoms)

    bs = band_calc.band_structure()
    bs.subtract_reference()
    bs.write(f'bs_{name}.json')

    # for name in dftb_in_files:
    #     move(curr_fold/name, curr_fold/f'{name}_2')

    print(f'\tStep 2 for {name} done')

    # t_forces.write(atoms, forces=forces, energy=e)

    os.chdir(curr_fold)
    print(f'Folder has been changed back to {curr_fold}')
