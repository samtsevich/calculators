from pathlib import Path
from typing import Dict

import yaml
from ase.data import chemical_symbols
from ase.io import read

from .. import F_MAX, KSPACING, N_STEPS, DEF_SMEARING



COMMON_VASP_PARAMS = {
    'xc': 'PBE',               # Exchange-correlation functional
    'prec': 'Accurate',        # Precision
    'encut': 600,              # Plane-wave cutoff
    'ediff': 1E-8,             # Energy convergence criterion
    'nsw': 0,                  # Number of steps for ionic relaxation
    'symprec': 0.01,           # Symmetry precision
    'ibrion': -1,              # Ion relaxation: no relaxation
    'isif': 3,                 # Stress tensor calculation
    'nelm': 200,               # Maximum number of electronic steps
    'ncore': 4,                # Number of cores
    'ismear': 0,               # Gaussian smearing
    'istart': 0,               # Start from scratch
    'lcharg': True,            # Don't write CHGCAR
    'lwave': True,             # Don't write WAVECAR
    'potim': 0.05,             # Time step for ionic relaxation
    'lreal': 'Auto',           # Use real-space projection for GPU acceleration
}


def add_vasp_arguments(parser, calc_type):
    """Construct argumentparser with subcommands and sections"""
    if calc_type == 'opt':
        description = 'ASE optimization with VASP'
    elif calc_type == 'scf':
        description = 'ASE single point calculation with VASP'
    elif calc_type == 'band':
        description = 'ASE band structure calculation with VASP'
    elif calc_type == 'eos':
        description = 'EOS with VASP for structure'
    elif calc_type == 'pdos':
        description = '(P)DOS calculation with VASP'
    else:
        raise ValueError(f'Unknown type {calc_type}')

    parser.description = description

    # Arguments common to all actions
    parser.add_argument('-c', '--config', dest='config', required=False, help='path to the config file')

    msg = 'path to the input file written in POSCAR format'
    parser.add_argument('-i', '--input', dest='input', required=False, help=msg)

    parser.add_argument('-o', '--outdir', dest='outdir', required=False, help='path to the output folder')
    msg = 'Kspacing value'
    parser.add_argument('--kspacing', dest='kspacing', type=float, default=KSPACING, required=False, help=msg)

    msg = 'Additional setups for the atoms in the structure'
    parser.add_argument('-s', '--setups', dest='setup', type=str, default=None, required=False, help=msg)

    msg = 'Smearing for DFT calculations in eV'
    parser.add_argument('-b', '--smearing', dest='smearing', type=float, default=DEF_SMEARING, required=False, help=msg)

    msg = 'Whether to perform spin-pol calculation'
    parser.add_argument('--spin', dest='spin', action='store_true', default=False, help=msg)

    if calc_type == 'opt' or calc_type == 'eos':
        msg = 'fmax for relaxation'
        parser.add_argument('-f', '--fmax', dest='fmax', type=float, default=F_MAX, required=False, help=msg)

        msg = 'Whether we optimize the cell or not'
        parser.add_argument("--full", dest="full_opt", action="store_true", default=False, help=msg)

        msg = 'Number of steps for optimization'
        parser.add_argument("--nsteps", dest="nsteps", default=N_STEPS, type=int, required=False, help=msg)

    elif calc_type == 'band':
        msg = 'whether calculation is made for the training of DFTB params from the band structure'
        parser.add_argument('--train', dest='is_training', action='store_true', help=msg)
    return parser


def get_args(args) -> dict:
    calc_type = args.subcommand

    if args.config:
        assert Path(args.config).exists()
        opt = vars(args)
        args = yaml.load(open(args.config), Loader=yaml.FullLoader)
        opt.update(args)
        args = opt
    else:
        args = vars(args)

    input = Path(args['input'])
    assert input.exists(), f'Seems like path to the input file is wrong.\n It is {input}'
    args['input'] = input
    structures = read(input, index=':')
    args['name'] = input.stem

    args['structures'] = structures

    if 'outdir' not in args or args['outdir'] is None:
        outdir = Path.cwd() / f'{calc_type}_{input.stem}'
    else:
        outdir = Path(args['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    args['outdir'] = outdir

    assert args['kspacing'] > 0, 'Seems like your value for KSPACING is not >0'
    if calc_type == 'opt':
        assert args['fmax'] > 0, 'Seems like your value for FMAX is not >0'

    if args['setups'] is not None:
        all_chemical_symbols = list(set([atom.symbol for structure in structures for atom in structure]))
        for el in all_chemical_symbols:
            assert el in chemical_symbols, f'Unknown element {el}'
            assert el in args['setups'], f'No setup for {el}'

    assert args['smearing'] > 0, 'Seems like your value for smearing is not >0'

    return args


def get_basic_params(args) -> dict:
    name = args['name']
    smearing = args['smearing']
    kspacing = args['kspacing']
    general_params = {
        'system': name,         # Job name
        'kspacing': kspacing,   # k-point grid for SCF
        'sigma': smearing,      # Smearing parameter
    }
    basic_params = COMMON_VASP_PARAMS.copy()
    basic_params.update(general_params)

    if args['spin']:
        basic_params['ispin'] = 2

    if args['setups'] is not None:
        basic_params['setups'] = args['setups']

    return basic_params


def get_total_N_val_e(file: Path) -> int:
    """Read valences from OUTCAR"""
    assert file.exists(), f'File {file} does not exist'
    with open(file) as f:
        for line in f:
            if 'NELECT' in line:
                TON_N_VAL_E = int(float(line.split()[2]))
                return TON_N_VAL_E
    return None
