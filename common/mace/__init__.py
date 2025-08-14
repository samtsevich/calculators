from pathlib import Path
from typing import Dict, List

from ase import Atoms
from ase.io import read

from .. import F_MAX, N_STEPS

# Default parameters for MACE calculations
DEFAULT_DEVICE = "cpu"
DEFAULT_DTYPE = "float64"
DEFAULT_COMPILE_MODE = None


def add_mace_arguments(parser, calc_type: str):
    """Add MACE-specific command line arguments to the parser.

    Args:
        parser: ArgumentParser instance to add arguments to
        calc_type: Type of calculation ('scf', 'opt', 'eos')
    """
    if calc_type == "scf":
        description = "ASE single point calculation with MACE"
    elif calc_type == "opt":
        description = "ASE optimization with MACE"
    elif calc_type == "eos":
        description = "ASE equation of state calculation with MACE"
    else:
        raise ValueError(f"Unknown calculation type: {calc_type}")

    parser.description = description

    # Common arguments for all calculation types
    msg = "Path to the POSCAR-like file with input structure"
    parser.add_argument("-i", "--input", dest="input", required=True, help=msg)

    msg = "Path to the MACE model file (.model extension)"
    parser.add_argument("--model", dest="model", required=True, help=msg)

    msg = f"Device to use for calculations (cpu, cuda, mps). Default: {DEFAULT_DEVICE}"
    parser.add_argument("--device", dest="device", default=DEFAULT_DEVICE, help=msg)

    msg = f"Default data type for calculations (float32, float64). Default: {DEFAULT_DTYPE}"
    parser.add_argument(
        "--default_dtype", dest="default_dtype", default=DEFAULT_DTYPE, help=msg
    )

    msg = "PyTorch compilation mode for performance optimization"
    parser.add_argument(
        "--compile_mode", dest="compile_mode", default=DEFAULT_COMPILE_MODE, help=msg
    )

    msg = "Output directory for calculation results"
    parser.add_argument("-o", "--outdir", dest="outdir", required=False, help=msg)

    # Arguments specific to optimization and EOS calculations
    if calc_type in ["opt", "eos"]:
        msg = f"Force convergence criterion for optimization. Default: {F_MAX}"
        parser.add_argument(
            "--fmax", dest="fmax", default=F_MAX, type=float, required=False, help=msg
        )

        msg = f"Maximum number of optimization steps. Default: {N_STEPS}"
        parser.add_argument(
            "--nsteps",
            dest="nsteps",
            default=N_STEPS,
            type=int,
            required=False,
            help=msg,
        )

        msg = "Optimize both atomic positions and unit cell parameters"
        parser.add_argument(
            "--full", dest="full_opt", action="store_true", default=False, help=msg
        )

    return parser


def get_args(args: Dict, calc_type: str) -> Dict:
    """Process and validate command line arguments.

    Args:
        args: Dictionary of parsed command line arguments
        calc_type: Type of calculation ('scf', 'opt', 'eos')

    Returns:
        Dictionary with processed and validated arguments

    Raises:
        AssertionError: If validation fails
        ValueError: If required arguments are missing
        RuntimeError: If device validation fails
    """
    # Validate MACE dependencies first
    validate_mace_dependencies()

    # Handle input structure file and create structures list
    if "structures" in args and args["structures"] is not None:
        name = args["name"]
        # Ensure structures is a list
        if not isinstance(args["structures"], list):
            args["structures"] = [args["structures"]]
    elif "input" in args and args["input"] is not None:
        input_path = Path(args["input"])
        assert input_path.exists(), f"Input structure file {input_path} does not exist"
        assert input_path.is_file(), f"Input path {input_path} is not a file"

        try:
            # Read all structures from file (supports multi-structure files)
            structures = read(input_path, index=":")
            # Ensure structures is always a list
            if isinstance(structures, Atoms):
                structures = [structures]
            args["structures"] = structures
        except Exception as e:
            raise ValueError(f"Failed to read structure file {input_path}: {e}")

        name = input_path.stem
    else:
        raise ValueError("Please specify input structure file with -i/--input")

    args["name"] = name

    # Validate MACE model file existence and readability
    model_path = Path(args["model"])
    assert model_path.exists(), f"MACE model file {model_path} does not exist"
    assert model_path.is_file(), f"MACE model path {model_path} is not a file"

    # Check if file is readable
    try:
        with open(model_path, "rb") as f:
            # Try to read first few bytes to ensure file is readable
            f.read(1024)
    except (PermissionError, OSError) as e:
        raise ValueError(f"MACE model file {model_path} is not readable: {e}")

    # Validate model file extension
    if model_path.suffix not in [".model", ".pth"]:
        raise ValueError(
            f"MACE model file must have .model or .pth extension, got {model_path.suffix}"
        )

    # Validate MACE model file format and compatibility
    validate_mace_model_file(model_path)

    args["model_path"] = model_path.resolve()

    # Validate device availability (CPU/CUDA/MPS)
    device = args["device"].lower()
    valid_devices = ["cpu", "cuda", "mps"]
    assert device in valid_devices, (
        f"Device must be one of {valid_devices}, got {device}"
    )

    # Check device availability
    validated_device = check_device_availability(device)
    args["device"] = validated_device

    # Validate data type
    dtype = args["default_dtype"]
    valid_dtypes = ["float32", "float64"]
    assert dtype in valid_dtypes, (
        f"Data type must be one of {valid_dtypes}, got {dtype}"
    )
    args["default_dtype"] = dtype

    # Set up output directory structure
    if args["outdir"] is None:
        outdir = Path.cwd() / f"{calc_type}_{name}"
    else:
        outdir = Path(args["outdir"])

    # Create output directory with proper error handling
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise ValueError(f"Cannot create output directory {outdir}: {e}")

    args["outdir"] = outdir.resolve()

    # Validate numerical parameters (fmax > 0, nsteps > 0)
    if calc_type in ["opt", "eos"]:
        fmax = args["fmax"]
        nsteps = args["nsteps"]

        # Validate fmax
        if not isinstance(fmax, (int, float)):
            raise ValueError(
                f"Force convergence criterion (fmax) must be a number, got {type(fmax).__name__}"
            )
        if fmax <= 0:
            raise ValueError(
                f"Force convergence criterion (fmax) must be positive, got {fmax}"
            )

        # Validate nsteps
        if not isinstance(nsteps, int):
            raise ValueError(
                f"Number of steps (nsteps) must be an integer, got {type(nsteps).__name__}"
            )
        if nsteps <= 0:
            raise ValueError(f"Number of steps (nsteps) must be positive, got {nsteps}")

    return args


def validate_mace_dependencies():
    """Validate that required MACE dependencies are available.

    Raises:
        ImportError: If required packages are not available
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "PyTorch is required for MACE calculations.\n"
            "Install with one of the following commands:\n"
            "  - CPU only: pip install torch\n"
            "  - With CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
            "  - Check https://pytorch.org/get-started/locally/ for more options"
        )

    try:
        from mace.calculators import MACECalculator
    except ImportError:
        raise ImportError(
            "mace-torch package is required for MACE calculations.\n"
            "Install with: pip install mace-torch\n"
            "For development version: pip install git+https://github.com/ACEsuit/mace.git"
        )

    # Check PyTorch version compatibility
    torch_version = torch.__version__
    major, minor = map(int, torch_version.split(".")[:2])
    if major < 1 or (major == 1 and minor < 12):
        raise ImportError(
            f"MACE requires PyTorch >= 1.12.0, but found {torch_version}.\n"
            "Please upgrade PyTorch: pip install --upgrade torch"
        )


def validate_mace_model_file(model_path: Path):
    """Validate MACE model file format and compatibility.

    Args:
        model_path: Path to the MACE model file

    Raises:
        ValueError: If model file is invalid or incompatible
    """
    import torch

    try:
        # Try to load the model file to check if it's a valid PyTorch model
        if model_path.suffix == ".model":
            # MACE .model files are typically PyTorch state dicts
            state_dict = torch.load(model_path, map_location="cpu")

            # Check if it looks like a MACE model by looking for expected keys
            expected_keys = ["model_state_dict", "model"]
            if not any(key in state_dict for key in expected_keys):
                # Try to load as direct state dict
                if not isinstance(state_dict, dict):
                    raise ValueError(
                        "Model file does not contain a valid state dictionary"
                    )

        elif model_path.suffix == ".pth":
            # PyTorch model files
            torch.load(model_path, map_location="cpu")

    except Exception as e:
        if "corrupted" in str(e).lower() or "truncated" in str(e).lower():
            raise ValueError(
                f"MACE model file {model_path} appears to be corrupted. "
                "Please re-download or regenerate the model file."
            )
        elif "version" in str(e).lower():
            raise ValueError(
                f"MACE model file {model_path} was created with an incompatible PyTorch version. "
                f"Current PyTorch version: {torch.__version__}. "
                "Try updating PyTorch or using a compatible model file."
            )
        else:
            raise ValueError(
                f"MACE model file {model_path} is not a valid PyTorch model file: {e}"
            )


def check_device_availability(device: str) -> str:
    """Check if the specified device is available.

    Args:
        device: Device string ('cpu', 'cuda', 'mps')

    Returns:
        Validated device string

    Raises:
        RuntimeError: If device is not available
    """
    import torch

    if device == "cpu":
        return device
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but CUDA is not available.\n"
                "Possible solutions:\n"
                "  - Use --device cpu for CPU-only calculations\n"
                "  - Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118\n"
                "  - Check CUDA installation: nvidia-smi"
            )

        # Check for CUDA memory issues
        try:
            # Try to allocate a small tensor to check if CUDA is working
            test_tensor = torch.zeros(1, device="cuda")
            del test_tensor
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    "CUDA device is available but out of memory.\n"
                    "Try:\n"
                    "  - Using --device cpu\n"
                    "  - Closing other GPU applications\n"
                    "  - Using a smaller model or batch size"
                )
            else:
                raise RuntimeError(f"CUDA device error: {e}")

        return device
    elif device == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                "MPS device requested but MPS is not available.\n"
                "MPS requires:\n"
                "  - macOS 12.3 or later\n"
                "  - Apple Silicon (M1/M2) processor\n"
                "Use --device cpu instead"
            )

        # Check if MPS is built and working
        try:
            if not torch.backends.mps.is_built():
                raise RuntimeError(
                    "MPS backend is not built in this PyTorch installation.\n"
                    "Install MPS-enabled PyTorch or use --device cpu"
                )

            # Try to allocate a small tensor to check if MPS is working
            test_tensor = torch.zeros(1, device="mps")
            del test_tensor
        except Exception as e:
            raise RuntimeError(f"MPS device error: {e}. Use --device cpu instead")

        return device
    else:
        raise ValueError(f"Unknown device: {device}. Valid options: cpu, cuda, mps")
