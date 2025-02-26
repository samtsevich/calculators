from pathlib import Path
from typing import Dict

import yaml
from ase.data import chemical_symbols
from ase.io import read
from ase.io.espresso import read_fortran_namelist

from .. import F_MAX, KSPACING, N_STEPS, DEF_SMEARING


def add_qe_arguments(parser, calc_type):
    """Construct argumentparser with subcommands and sections"""
    if calc_type == 'opt':
        description = 'ASE optimization with QE'
    elif calc_type == 'scf':
        description = 'ASE single point calculation with QE'
    elif calc_type == 'band':
        description = 'ASE band structure calculation with QE'
    elif calc_type == 'eos':
        description = 'EOS with QE for structure'
    elif calc_type == 'pdos':
        description = 'PDOS with QE for structure'
    else:
        raise ValueError(f'Unknown type {calc_type}')

    parser.description = description

    # Arguments common to all actions
    parser.add_argument('-c', '--config', dest='config', required=False, help='path to the config file')

    msg = 'path to the input file written in POSCAR format'
    parser.add_argument('-i', '--input', dest='input', required=False, help=msg)

    parser.add_argument('-k', '--options', dest='options_file', required=False, help='path to the options file')
    parser.add_argument('-pp', dest='pseudopotentials', required=False, help='dict of pseudopotentials for ASE')
    parser.add_argument('--pp_dir', dest='pp_dir', required=False, help='path to folder with pseudopotentials')
    parser.add_argument('-o', '--outdir', dest='outdir', required=False, help='path to the output folder')
    parser.add_argument('--kspacing', dest='kspacing', type=float, default=KSPACING, required=False, help='Kspacing value')

    if calc_type == 'opt' or calc_type == 'eos':
        msg = 'fmax for relaxation'
        parser.add_argument('-f', '--fmax', dest='fmax', type=float, default=F_MAX, required=False, help=msg)

        msg = 'Whether we optimize the cell or not'
        parser.add_argument("--full", dest="full_opt", action="store_true", default=False, help=msg)

        msg = 'Number of steps for optimization'
        parser.add_argument("--nsteps", dest="nsteps", default=N_STEPS, type=int, required=False, help=msg)

    elif calc_type == 'pdos':
        msg = 'Smearing for PDOS calculation in eV'
        parser.add_argument('--smearing', dest='smearing', type=float, default=DEF_SMEARING, required=False, help=msg)
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
        args.pseudopotentials = eval(args.pseudopotentials)
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

    options_file = Path(args['options'])
    assert options_file.exists(), f"Seems like path to the options file is wrong.\n It is {options_file}"
    # Read options from the file
    with open(options_file) as fp:
        data, _ = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')
    args['data'] = data

    pp = args['pseudopotentials']
    all_elements = set()
    for struc in structures:
        all_elements.update(struc.get_chemical_symbols())

    for s in all_elements:
        assert s in pp.keys(), f'{s} is not presented in the pseudopotentials'

    pp_dir = Path(args['pp_dir'])
    if pp_dir.is_symlink():
        pp_dir = pp_dir.readlink()
    assert pp_dir.is_dir(), 'Seems like folder with pseudopotentials does not exist or wrong'
    pp_dir = pp_dir.resolve()
    args['pp_dir'] = pp_dir

    assert args['kspacing'] > 0, 'Seems like your value for KSPACING is not >0'
    if calc_type == 'opt':
        assert args['fmax'] > 0, 'Seems like your value for FMAX is not >0'

    return args


# Read valences from the QE output file
def read_valences(qe_output_file: str) -> Dict[str, float]:
    valences = {}
    is_reading = False
    with open(qe_output_file, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.strip() == '' and is_reading:
                break
            if 'atomic species' in line and 'valence' in line:
                is_reading = True
                continue
            if not is_reading:
                continue
            data = line.split()
            specie, N_val_electrons = data[0], float(data[1])
            assert specie in chemical_symbols, f'Unknown specie {specie}'
            assert N_val_electrons > 0, f'Valence electrons for {specie} is not >0'
            valences[specie] = N_val_electrons
    return valences
