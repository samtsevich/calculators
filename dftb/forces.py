from ase.calculators.dftb import Dftb

from ase.io import read, write, Trajectory
from ase.io.vasp import write_vasp

from pathlib import Path

import argparse
import numpy as np
import sys


KSPACING = 0.02
DFTB_COMMAND = 'mpirun -np 8 dftb+'


def get_additional_params(species, type: str='opt'):
    '''
    type: 'band' or 'static' or 'opt_geometry'
    '''

    params = {}

    if type == 'scc':
        params.update({ 'Hamiltonian_MaxSCCIterations': 100,
                        'Hamiltonian_ReadInitialCharges': 'No', # Static calculation
                        'Hamiltonian_SCC': 'YES',
                      })
    elif type == 'band':
        params.update({ 'Hamiltonian_MaxSCCIterations': 1,
                        'Hamiltonian_ReadInitialCharges': 'Yes',
                        'Hamiltonian_SCC': 'False',
                       })
    elif type == 'opt':
        params.update({ 'Driver_': 'LBFGS',
                        'Driver_MaxForceComponent': 1e-4,
                        'Driver_MaxSteps': 1000,
                        'Hamiltonian_MaxSCCIterations': 100,
                        'Hamiltonian_SCC': 'YES',
                       })
    else:
        raise ValueError('type must be band or static or opt_geometry')
    return params



def get_KPoints(kspacing: float, cell):
    assert kspacing > 0
    angLattice = cell.cellpar()
    dist = np.zeros(3)
    dist[2] = cell.volume / (angLattice[0] * angLattice[1] * np.sin(angLattice[5]*np.pi/180))
    dist[1] = cell.volume / (angLattice[0] * angLattice[2] * np.sin(angLattice[4]*np.pi/180))
    dist[0] = cell.volume / (angLattice[1] * angLattice[2] * np.sin(angLattice[3]*np.pi/180))

    Kpoints = [int(x) for x in np.ceil(1.0 / (dist * kspacing))]
    return Kpoints


## path to .skf files (parameters) 
# d='/nexus/posix0/FHI-Theory/chiarapan/shared/ligra/fundamentals/dftb-parameters/sp-s_norep/'


GENERAL_PARAMS = {
    'CalculateForces': 'YES',
    'Hamiltonian_': 'DFTB',
    'Hamiltonian_Filling_': 'Fermi',
    'Hamiltonian_Filling_Temperature': 0.0001,   ## T in atomic units
    'Hamiltonian_ForceEvaluation' :'dynamics',
    'Hamiltonian_MaxSCCIterations': 500,
    ## filling and mixer flags are optional
    'Hamiltonian_Mixer_': 'Broyden',
    'Hamiltonian_Mixer_MixingParameter': 0.1,
    # 'Hamiltonian_Dispersion' : 'LennardJones{Parameters = UFFParameters{}}',	## no dispersion
    'Hamiltonian_PolynomialRepulsive': 'SetForAll {YES}',
    'Hamiltonian_SCCTolerance': 1.0E-8,

    'Options_': '',
    'Options_WriteResultsTag': 'Yes',        # Default:No
    'Options_WriteDetailedOut': 'Yes'
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Forces calculation by DFTB for Gprep usage')
    parser.add_argument("-i", dest="input",
                        help="path to the file with input structures in .traj file")
    parser.add_argument("-s", dest="dir_skf", help='path to .skf files folder')
    parser.add_argument("-t", dest="type", default='scc', help='type of the calculation')
    parser.add_argument("-o", dest="output", default=None, help='output file in .traj file')
    args = parser.parse_args()

    input_traj_path = Path(args.input)
    assert input_traj_path.exists()
    name = input_traj_path.stem

    dir_skf = Path(args.dir_skf)
    assert dir_skf.is_dir()

    assert args.type in ['scc', 'band', 'opt'], 'Seems like you chose wrong type of the calculation'

    output = f'{name}_out.traj' if args.output is None else args.output


    ## traj of structures
    t = Trajectory(input_traj_path, 'r')

    ## output traj with dftb forces
    t_forces = Trajectory(output, 'w')

    params = GENERAL_PARAMS.copy()

    species = set()
    for atoms in t:
        species.update(atoms.get_chemical_symbols())


    additional_params = {
        'slako_dir': str(dir_skf)+'/',
        'Hamiltonian_SlaterKosterFiles_Prefix': str(dir_skf)+'/',
        'Hamiltonian_SlaterKosterFiles_Separator':'"-"',
        'Hamiltonian_SlaterKosterFiles_Suffix':'".skf"',
        # 'run_manyDftb_steps': False,
        # 'Hamiltonian_MaxAngularMomentum_C': 'p',
        # 'Hamiltonian_MaxAngularMomentum_Li': 's',
    }

    additional_params.update(get_additional_params(species, type=args.type))

    params.update(additional_params)

    print('1')

    for i, atoms in enumerate(t):
        print(f'Structure {i}')
        ## path to dftb+ executable > output_file.out
        dftb_command = f'{DFTB_COMMAND} > dftb_{name}_{i}.out'  ## cluster


        params.update({
            'label': f'{name}_{i}',
            'command': dftb_command,
            'kpts': get_KPoints(KSPACING, atoms.get_cell()),
        })

        calc=Dftb(**params)

        atoms.write(f'a_{name}_{i}.gen')
        write_vasp(f'a_{name}_{i}.vasp', atoms, sort=True, vasp5=True, direct=True)
        atoms.set_calculator(calc)
        # try:
        forces = atoms.get_forces()
        e = atoms.get_potential_energy()
        t_forces.write(atoms, forces=forces, energy=e)
        # except:
        #     print(f'name_{i} did not converge')
