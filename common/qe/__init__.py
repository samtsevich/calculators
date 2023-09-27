import yaml

from ase.constraints import FixAtoms
from ase.io import read
from ase.io.espresso import read_fortran_namelist
from pathlib import Path


KSPACING = 0.04
F_MAX = 0.01


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
    else:
        raise ValueError(f'Unknown type {calc_type}')

    parser.description = description

    # Arguments common to all actions
    parser.add_argument('-c',
                        '--config',
                        dest='config',
                        required=False,
                        help='path to the config file')

    parser.add_argument('-i',
                        '--input',
                        dest='input',
                        required=False,
                        help='path to the input file written in POSCAR format')
    parser.add_argument('--fixed_idx',
                        dest='fixed_idx',
                        required=False,
                        help='path to the file with fixed indices')
    parser.add_argument('-k',
                        '--options',
                        dest='options_file',
                        required=False,
                        help='path to the options file')
    parser.add_argument('-pp',
                        dest='pseudopotentials',
                        required=False,
                        help='dict of pseudopotentials for ASE')
    parser.add_argument('--pp_dir',
                        dest='pp_dir',
                        required=False,
                        help='path to folder with pseudopotentials')
    parser.add_argument('-o',
                        '--outdir',
                        dest='outdir',
                        required=False,
                        help='path to the output folder')
    parser.add_argument('-kspacing',
                        dest='kspacing',
                        default=KSPACING,
                        required=False,
                        help='Kspacing value')
    if calc_type == 'opt':
        parser.add_argument('-f',
                            '--fmax',
                            dest='fmax',
                            default=F_MAX,
                            required=False,
                            help='fmax for relaxation')
    elif calc_type == 'band':
        parser.add_argument('--train',
                            dest='is_training',
                            action='store_true',
                            help='whether calculation is made for the training of DFTB params from the band structure')
    return parser


def get_args(args):
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
    assert input.exists(
    ), f'Seems like path to the input file is wrong.\n It is {input}'
    args['input'] = input
    structure = read(input, format='vasp')

    args['name'] = input.stem

    # Constraints
    fixed_idx = args['fixed_idx']
    if fixed_idx is not None:
        with open(fixed_idx) as fp:
            idx = list(map(int, fp.read().split()))
            assert len(idx), 'Seems something wrong with idx of fixed atoms'
            structure.set_constraint(FixAtoms(indices=idx))
    args['structure'] = structure

    if 'outdir' not in args or args['outdir'] is None:
        outdir = Path.cwd()/f'{calc_type}_{input.stem}'
    else:
        outdir = Path(args['outdir'])
    outdir.mkdir(parents=True, exist_ok=True)
    args['outdir'] = outdir

    options_file = args['options']
    assert Path(options_file).exists(
    ), f"Seems like path to the options file is wrong.\n It is {options_file}"
    # Read options from the file
    with open(options_file) as fp:
        data, _ = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')
    args['data'] = data

    pp = args['pseudopotentials']
    for s in list(set(args['structure'].get_chemical_symbols())):
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
