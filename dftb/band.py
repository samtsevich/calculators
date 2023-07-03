import argparse
import numpy as np

from pathlib import Path
from shutil import copy

from ase.calculators.dftb import Dftb
from ase.io import write, Trajectory
from ase.io.vasp import write_vasp, read_vasp


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
                       'Hamiltonian_SCC': 'YES',
                       })
    elif type == 'band':
        params.update({'Hamiltonian_SCC': 'No',
                       'Hamiltonian_ReadInitialCharges': 'Yes',
                       })
    elif type == 'opt':
        params.update({'Driver_': 'LBFGS',
                       'Driver_MaxForceComponent': 1e-4,
                       'Driver_MaxSteps': 1000,
                       'Hamiltonian_SCC': 'YES',
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
    'Hamiltonian_Filling_Temperature': 0.0001,   # T in atomic units
    'Hamiltonian_ForceEvaluation': 'dynamics',
    'Hamiltonian_MaxSCCIterations': 500,
    # filling and mixer flags are optional
    'Hamiltonian_Mixer_': 'Broyden',
    'Hamiltonian_Mixer_MixingParameter': 0.1,
    # 'Hamiltonian_Dispersion' : 'LennardJones{Parameters = UFFParameters{}}',	# no dispersion
    'Hamiltonian_PolynomialRepulsive': 'SetForAll {YES}',
    'Hamiltonian_SCCTolerance': 1.0E-8,

    'Options_': '',
    'Options_WriteResultsTag': 'Yes',        # Default:No
    'Options_WriteDetailedOut': 'Yes'
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Band structure calculation by DFTB')
    parser.add_argument(
        "-i", dest="input", help="path to the file with input structures in POSCAR format")
    parser.add_argument("-s", dest="dir_skf",
                        default=Path.cwd(), help='path to .skf files folder')
    parser.add_argument("-o", dest="output", required=False,
                        help=' output dir of the calculation')
    args = parser.parse_args()

    input_traj_path = Path(args.input)
    assert input_traj_path.exists()

    dir_skf = Path(args.dir_skf)
    assert dir_skf.is_dir(), 'Please, check carefully path to the directory with .skf files'

    curr_fold = Path.cwd()
    dftb_in_files = ['dftb_in.hsd', 'dftb_pin.hsd']
    if args.output is not None:
        res_fold = curr_fold/args.output
    else:
        res_fold = curr_fold/f'res_{input_traj_path.stem}'

    res_fold.mkdir(parents=True, exist_ok=True)

    atoms = read_vasp(input_traj_path)

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

    calc_fold = res_fold
    # path to dftb+ executable > output_file.out
    dftb_command = f'{DFTB_COMMAND} > dftb_{name}.out'  # cluster

    params.update({
        'label': f'{name}',
        'command': dftb_command,
        'kpts': get_KPoints(KSPACING, atoms.get_cell()),
    })

    calc = Dftb(**params, directory=calc_fold)

    atoms.write(calc_fold/f'a_{name}.gen')
    write_vasp(calc_fold/f'a_{name}.vasp', atoms,
               sort=True, vasp5=True, direct=True)
    atoms.calc = calc

    e = atoms.get_potential_energy()
    fermi_level = calc.get_fermi_level()

    print(f'\tStep 1 for {name} done')

    for x in dftb_in_files:
        copy(calc_fold/x, res_fold/f'calc_scc_{name}')

    # Step 2.
    path = atoms.cell.bandpath()
    print(path)

    calc.calculate(atoms)

    params.update(get_additional_params(species, type='band'))

    assert params['Hamiltonian_SCC'] == 'No'

    # Stupid ASE does not recognize k-points for band structures, when there is no 'path' key in the dict
    params.update({'kpts': {**path.todict(), 'path': ''}})

    band_calc = Dftb(atoms=atoms, directory=calc_fold, **params)
    band_calc.calculate(atoms)

    bs = band_calc.band_structure()
    bs = bs.subtract_reference()
    bs.write(res_fold/f'bs_{name}.json')

    for n in dftb_in_files:
        copy(calc_fold/n, res_fold/f'calc_bs_{n}')

    print(f'\tStep 2 for {name} done')
