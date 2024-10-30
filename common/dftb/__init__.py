from pathlib import Path

import numpy as np
from ase.io import read

from .. import F_MAX, KSPACING, N_STEPS

N_PROCESS = 1

# Fermi temperature for the filling
FERMI_TEMP = 0.0001


MIXERS = ['Broyden', 'Andersen']

# For band structure plotting
E_MIN = -20
E_MAX = 25

SCC_TOLERANCE = 1.0e-6


def add_dftb_arguments(parser, calc_type):
    if calc_type == 'opt':
        description = 'ASE optimization with DFTB'
    elif calc_type == 'scf':
        description = 'ASE single point calculation with DFTB'
    elif calc_type == 'band':
        description = 'ASE band structure calculation with DFTB'
    elif calc_type == 'eos':
        description = 'ASE EOS calculation with DFTB'
    elif calc_type == 'neb':
        description = 'ASE NEB calculation with DFTB'
    else:
        raise ValueError(f'Unknown type {calc_type}')

    parser.description = description

    msg = 'path to the POSCAR-like file with input structure'
    parser.add_argument("-i", "--input", dest="input", help=msg)

    msg = 'path to the folder with `.skf` files'
    parser.add_argument("-s", "--skf", dest="skfs_dir", default=Path.cwd(), help=msg)

    msg = 'Number of processors'
    parser.add_argument('-np', dest='nproc', default=N_PROCESS, type=int, required=False, help=msg)

    msg = 'Kspacing value'
    parser.add_argument('--kspacing', dest='kspacing', default=KSPACING, type=float, required=False, help=msg)

    parser.add_argument('-m', '--mixer', dest='mixer', required=False, default='Broyden', help='Name of the mixer')

    parser.add_argument('-d3', dest='d3', action='store_true', default=False, help='Whether use D3 dispersion or not')

    msg = 'Whether use polynomial repulsion or splines inside SKF files. Default: False'
    parser.add_argument("--pol-rep", dest="pol_rep", action="store_true", default=False, help=msg)

    msg = 'Fermi Temperature for the filling'
    parser.add_argument("--fermi-temp", dest="fermi_temp", default=FERMI_TEMP, type=float, required=False, help=msg)

    if calc_type == 'opt' or calc_type == 'neb' or calc_type == 'eos':
        msg = 'F_MAX for optimization'
        parser.add_argument("--fmax", dest="fmax", default=F_MAX, type=float, required=False, help=msg)

        msg = 'Number of steps for optimization'
        parser.add_argument("--nsteps", dest="nsteps", default=N_STEPS, type=int, required=False, help=msg)

        msg = 'Whether we optimize the cell or not'
        parser.add_argument("--full", dest="full_opt", action="store_true", default=False, help=msg)

    if calc_type == 'band':
        msg = 'E min for the band structure plotting'
        parser.add_argument("--emin", dest="emin", default=E_MIN, type=float, required=False, help=msg)

        msg = 'E max for the band structure plotting'
        parser.add_argument("--emax", dest="emax", default=E_MAX, type=float, required=False, help=msg)

    # Output
    msg = 'output dir of the calculation'
    parser.add_argument("-o", "--outdir", dest="outdir", required=False, help=msg)

    return parser


def get_args(args: dict, calc_type: str) -> dict:
    if 'structures' in args.keys() and args['structures'] is not None:
        name = args['name']
    elif 'input' in args.keys() and args['input'] is not None:
        input_path = Path(args['input'])
        assert input_path.exists(), f'File {input_path} does not exist'
        args['structures'] = read(input_path, index=':')
        name = input_path.stem
    else:
        raise ValueError('Please, specify input file or trajectory')

    args['name'] = name

    skfs_dir = Path(args['skfs_dir'])
    if skfs_dir.is_symlink():
        skfs_dir = skfs_dir.readlink()
    else:
        skfs_dir = skfs_dir.resolve()
    msg = f'Please, check carefully path to the directory with .skf files\nNow it is {skfs_dir}'
    assert skfs_dir.is_dir(), msg
    args['skfs_dir'] = skfs_dir

    # check mixer
    mixer = args['mixer']
    assert mixer in MIXERS, f'Unknown mixer {mixer}'

    if args['outdir'] is None:
        outdir = Path.cwd() / f'{calc_type}_{name}'
    else:
        outdir = Path(args['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    args['outdir'] = outdir

    params = GENERAL_PARAMS.copy()

    # update mixer
    params.update({'Hamiltonian_Mixer_': mixer})

    additional_params = {
        'slako_dir': str(skfs_dir) + '/',
        'Hamiltonian_SlaterKosterFiles_Prefix': str(skfs_dir) + '/',
        'Hamiltonian_SlaterKosterFiles_Separator': '"-"',
        'Hamiltonian_SlaterKosterFiles_Suffix': '".skf"',
        'Hamiltonian_Filling_Temperature': args['fermi_temp'],  # T in atomic units
        # 'run_manyDftb_steps': False,
        # 'Hamiltonian_MaxAngularMomentum_C': 'p',
        # 'Hamiltonian_MaxAngularMomentum_Li': 's',
        # TODO
        # 'Parallel_': '',
        # 'Parallel_Groups': nproc,
    }

    if args['pol_rep']:
        additional_params.update(
            {
                'Hamiltonian_PolynomialRepulsive': 'SetForAll {YES}',
            }
        )

    additional_params.update(get_calc_type_params(calc_type=calc_type))

    if args['d3']:
        additional_params.update(get_dispersion_params())

    params.update(additional_params)
    args['dftb_params'] = params

    return args


def get_dispersion_params() -> dict:
    # TODO these parameters must be tunable
    params = {
        'Hamiltonian_Dispersion_': 'DftD3',
        'Hamiltonian_Dispersion_s6': 1.0,
        'Hamiltonian_Dispersion_s8': 0.5883,
        'Hamiltonian_Dispersion_Damping_': 'BeckeJohnson',
        'Hamiltonian_Dispersion_Damping_a1': 0.5719,
        'Hamiltonian_Dispersion_Damping_a2': 3.6017,
    }
    return params


def get_calc_type_params(calc_type: str) -> dict:
    '''
    type: 'band' or 'static' or 'opt_geometry'
    '''

    params = {}
    common_scf_opt = {
        'Hamiltonian_MaxSCCIterations': 2000,
        'Analysis_': '',
        'Analysis_CalculateForces': 'Yes',
    }

    if calc_type == 'scf':
        params.update(common_scf_opt)
        params.update(
            {
                'Hamiltonian_ReadInitialCharges': 'Yes',
                'Hamiltonian_SCCTolerance': SCC_TOLERANCE,
            }
        )
    elif calc_type == 'opt' or calc_type == 'eos':
        params.update(common_scf_opt)
        params.update(
            {
                'Hamiltonian_ReadInitialCharges': 'Yes',
                'Hamiltonian_SCCTolerance': SCC_TOLERANCE,
            }
        )  # Static calculation
    elif calc_type == 'band':
        params.update(
            {
                'Hamiltonian_ReadInitialCharges': 'Yes',
                'Hamiltonian_MaxSCCIterations': 1,
                'Hamiltonian_SCCTolerance': 1e6,
            }
        )

    # elif type == 'opt':
    #     params.update({'Driver_': 'LBFGS',
    #                    'Driver_MaxForceComponent': 1e-4,
    #                    'Driver_MaxSteps': 1000,
    #                    'Hamiltonian_ReadInitialCharges': 'Yes',  # Static calculation
    #                    'Hamiltonian_MaxSCCIterations': 300,
    #                    })
    else:
        raise ValueError('type must be band or static or opt_geometry')
    return params


GENERAL_PARAMS = {
    'Hamiltonian_': 'DFTB',
    'Hamiltonian_SCC': 'Yes',
    'Hamiltonian_ShellResolvedSCC': 'Yes',
    'Hamiltonian_Filling_': 'Fermi',
    # 'Hamiltonian_Filling_Temperature': FERMI_TEMP,  # T in atomic units
    'Hamiltonian_ForceEvaluation': 'dynamics',
    'Hamiltonian_MaxSCCIterations': 500,
    # filling and mixer flags are optional
    'Hamiltonian_Mixer_': 'Broyden',
    'Hamiltonian_Mixer_MixingParameter': 0.1,
    # 'Hamiltonian_Dispersion' : 'LennardJones{Parameters = UFFParameters{}}',	## no dispersion
    # TODO
    'Hamiltonian_SCCTolerance': SCC_TOLERANCE,
    # 'Hamiltonian_SCCTolerance': 1.0E-5,
    'Options_': '',
    'Options_WriteResultsTag': 'Yes',  # Default:No
    'Options_WriteDetailedOut': 'Yes',
    # Parser options
    'ParserOptions_': '',
    'ParserOptions_IgnoreUnprocessedNodes': 'Yes',
    'ParserOptions_ParserVersion': 10,
}
