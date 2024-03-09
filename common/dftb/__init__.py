import numpy as np
import os

from ase.io import read
from ase.io.trajectory import TrajectoryReader
from pathlib import Path


DFTB_COMMAND = 'dftb+'
DFTB_IN_FILES = ['dftb_in.hsd', 'dftb_pin.hsd']
KSPACING = 0.02


def add_dftb_arguments(parser, calc_type):
    if calc_type == 'opt':
        description = 'ASE optimization with DFTB'
    elif calc_type == 'scf':
        description = 'ASE single point calculation with DFTB'
    elif calc_type == 'band':
        description = 'ASE band structure calculation with DFTB'
    elif calc_type == 'forces':
        description = 'ASE forces calculation with DFTB'
    else:
        raise ValueError(f'Unknown type {calc_type}')

    parser.description = description
    parser.add_argument("-i",
                        "--input",
                        dest="input",
                        help="path to the POSCAR-like file with input structure")
    # parser.add_argument("-t",
    #                     "--traj",
    #                     dest="trajectory",
    #                     required=False,
    #                     help="path to the `.traj`-file with input structures")
    parser.add_argument("-s",
                        "--skf",
                        dest="dir_skf",
                        default=Path.cwd(),
                        help='path to the folder with `.skf` files')
    # parser.add_argument('--no-rep',
    #                     dest='no_repulsion',
    #                     type=bool,
    #                     default=False,
    #                     required=False,
    #                     help='Whether we are taking into account repulsion or not.')
    parser.add_argument('--kspacing',
                        dest='kspacing',
                        default=KSPACING,
                        required=False,
                        help='Kspacing value')
    parser.add_argument("--pol_rep",
                        dest="polynomial_repulsion",
                        required=False,
                        default=True,
                        help='Whether use polynomial repulsion or splines inside SKF files. Default: True')
    parser.add_argument("-o",
                        "--outdir",
                        dest="outdir",
                        required=False,
                        help=' output dir of the calculation')
    return parser


def get_args(args) -> dict:
    calc_type = args.subcommand
    args = vars(args)

    if 'input' in args.keys() and args['input'] is not None:
        input_path = Path(args['input'])
        assert input_path.exists()
        args['structures'] = read(input_path, index=':')
        name = input_path.stem
    else:
        raise ValueError('Please, specify input file or trajectory')

    args['name'] = name

    dir_skf = Path(args['dir_skf'])
    if dir_skf.is_symlink():
        dir_skf = dir_skf.readlink()
    else:
        dir_skf = dir_skf.resolve()
    assert dir_skf.is_dir(), 'Please, check carefully path to the directory with .skf files'
    args['dir_skf'] = dir_skf

    if args['outdir'] is None:
        outdir = Path.cwd()/f'{calc_type}_{name}'
    else:
        outdir = Path(args['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    args['outdir'] = outdir

    params = GENERAL_PARAMS.copy()
    additional_params = {
        'slako_dir': str(dir_skf)+'/',
        'Hamiltonian_SlaterKosterFiles_Prefix': str(dir_skf)+'/',
        'Hamiltonian_SlaterKosterFiles_Separator': '"-"',
        'Hamiltonian_SlaterKosterFiles_Suffix': '".skf"',
        # 'run_manyDftb_steps': False,
        # 'Hamiltonian_MaxAngularMomentum_C': 'p',
        # 'Hamiltonian_MaxAngularMomentum_Li': 's',
    }

    if calc_type == 'scf' or calc_type == 'opt':
        additional_params.update(get_additional_params(type='scf'))
    # elif :
    #     additional_params.update(get_additional_params(type='opt'))
    params.update(additional_params)

    # path to dftb+ executable > output_file.out
    try:
        dftb_command = os.environ['DFTB_COMMAND']
    except KeyError:
        dftb_command = DFTB_COMMAND

    params.update({
        'label': f'{name}',
        'command': dftb_command,
    })

    args['dftb_params'] = params

    return args


def get_additional_params(type: str = 'opt'):
    '''
    type: 'band' or 'static' or 'opt_geometry'
    '''

    params = {}

    if type == 'scf':
        params.update({ 'Hamiltonian_MaxSCCIterations': 500,
                        'Hamiltonian_ReadInitialCharges': 'No',  # Static calculation
                        'Analysis_': '',
                        'Analysis_CalculateForces': 'Yes',})
    elif type == 'band':
        params.update({ 'Hamiltonian_ReadInitialCharges': 'Yes',
                        'Hamiltonian_MaxSCCIterations': 1,
                        'Hamiltonian_SCCTolerance': 1e6,})
    elif type == 'opt':
        params.update({'Driver_': 'LBFGS',
                       'Driver_MaxForceComponent': 1e-4,
                       'Driver_MaxSteps': 1000,
                       'Hamiltonian_MaxSCCIterations': 300,
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


GENERAL_PARAMS = {
    'Hamiltonian_': 'DFTB',
    'Hamiltonian_SCC': 'Yes',
    'Hamiltonian_Filling_': 'Fermi',
    'Hamiltonian_Filling_Temperature': 0.0001,  # T in atomic units
    'Hamiltonian_ForceEvaluation': 'dynamics',
    'Hamiltonian_MaxSCCIterations': 500,
    # filling and mixer flags are optional
    'Hamiltonian_Mixer_': 'Broyden',
    'Hamiltonian_Mixer_MixingParameter': 0.1,
    # 'Hamiltonian_Dispersion' : 'LennardJones{Parameters = UFFParameters{}}',	## no dispersion
    'Hamiltonian_SCCTolerance': 1.0E-8,
    'Options_': '',
    'Options_WriteResultsTag': 'Yes',        # Default:No
    'Options_WriteDetailedOut': 'Yes'
}
