#!/usr/bin/python3

from typing import List

import numpy as np
from ase.atoms import Atoms
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp

from . import get_args


def run_mace_scf(args: dict, calc_type: str):
    """Run MACE single point calculation (SCF).
    
    Args:
        args: Dictionary of processed command line arguments
        calc_type: Type of calculation (should be 'scf')
        
    Raises:
        AssertionError: If calc_type is not 'scf'
        ImportError: If MACE dependencies are not available
        RuntimeError: If calculation fails
    """
    # Validate calculation type
    assert calc_type == 'scf', 'This function is only for SCF calculation'

    # Import MACE calculator (with error handling)
    try:
        from mace.calculators import MACECalculator
    except ImportError:
        raise ImportError(
            "mace-torch package is required for MACE calculations. "
            "Install with: pip install mace-torch"
        )

    # Process arguments
    args = get_args(args, calc_type=calc_type)
    name = args['name']
    structures: List[Atoms] = args['structures']
    structure = structures[-1]  # Use the last structure (following existing patterns)

    outdir = args['outdir']
    model_path = args['model_path']
    device = args['device']
    default_dtype = args['default_dtype']
    compile_mode = args['compile_mode']

    # Print input parameters (following existing patterns)
    print(30 * '-')
    print(f'SCF calculation for {name}')
    print(f'MACE model: {model_path}')
    print(f'Device: {device}')
    print(f'Data type: {default_dtype}')
    if compile_mode:
        print(f'Compile mode: {compile_mode}')
    print(f'Output directory: {outdir}')
    print(30 * '-')

    try:
        # Load MACE model and set up calculator with specified parameters
        calc_params = {
            'model_paths': [str(model_path)],
            'device': device,
            'default_dtype': default_dtype,
        }
        
        # Add compile_mode if specified
        if compile_mode is not None:
            calc_params['compile_mode'] = compile_mode

        try:
            calculator = MACECalculator(**calc_params)
        except Exception as e:
            if "model" in str(e).lower():
                raise RuntimeError(
                    f"Failed to load MACE model from {model_path}.\n"
                    f"Error: {e}\n"
                    "Please check that the model file is valid and compatible."
                )
            elif "device" in str(e).lower() or "cuda" in str(e).lower():
                raise RuntimeError(
                    f"Failed to initialize MACE calculator on device '{device}'.\n"
                    f"Error: {e}\n"
                    "Try using --device cpu or check your GPU setup."
                )
            else:
                raise RuntimeError(f"Failed to initialize MACE calculator: {e}")

        # Attach calculator to structure and compute energy/forces
        structure.calc = calculator
        
        try:
            # Calculate energy (this will also compute forces)
            energy = structure.get_potential_energy()
            forces = structure.get_forces()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"GPU out of memory during MACE calculation.\n"
                    f"Error: {e}\n"
                    "Try using --device cpu or a smaller structure."
                )
            elif "nan" in str(e).lower() or "inf" in str(e).lower():
                raise RuntimeError(
                    f"MACE calculation produced invalid results (NaN/Inf).\n"
                    f"Error: {e}\n"
                    "This may indicate an incompatible structure or model."
                )
            else:
                raise RuntimeError(f"MACE energy/force calculation failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during MACE calculation: {e}")

        # Validate calculation results
        if not isinstance(energy, (int, float)) or not np.isfinite(energy):
            raise RuntimeError(f"MACE calculation produced invalid energy: {energy}")
        
        if not isinstance(forces, np.ndarray) or not np.all(np.isfinite(forces)):
            raise RuntimeError("MACE calculation produced invalid forces")

        print(f'Energy: {energy:.6f} eV')
        print(f'Max force: {max(abs(forces.flatten())):.6f} eV/Å')

    except RuntimeError:
        # Re-raise RuntimeError as-is (these are our custom error messages)
        raise
    except Exception as e:
        # Catch any other unexpected errors
        raise RuntimeError(f'Unexpected error in MACE SCF calculation: {e}')

    # Save final structure in VASP format using write_vasp()
    try:
        write_vasp(outdir / 'final.vasp', structure, sort=True, vasp5=True, direct=True)
    except PermissionError:
        raise RuntimeError(f'Permission denied writing to {outdir / "final.vasp"}. Check directory permissions.')
    except OSError as e:
        raise RuntimeError(f'Failed to write final structure to {outdir / "final.vasp"}: {e}')
    except Exception as e:
        raise RuntimeError(f'Unexpected error writing final structure: {e}')

    # Create trajectory file with energy and forces using ASE Trajectory
    try:
        with Trajectory(outdir / 'final.traj', 'w', properties=['energy', 'forces']) as traj:
            traj.write(structure, energy=energy, forces=forces)
    except PermissionError:
        raise RuntimeError(f'Permission denied writing to {outdir / "final.traj"}. Check directory permissions.')
    except OSError as e:
        raise RuntimeError(f'Failed to write trajectory file to {outdir / "final.traj"}: {e}')
    except Exception as e:
        raise RuntimeError(f'Unexpected error writing trajectory file: {e}')

    # Generate console output with calculation summary following existing patterns
    print(f'SCF of {name} is done.')
    print(30 * '-')


def mace_scf(args):
    """CLI wrapper function for MACE SCF calculation.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        AssertionError: If command is not 'mace'
    """
    assert args.command == 'mace', 'This function is only for MACE'
    calc_type: str = args.subcommand
    args: dict = vars(args)

    try:
        run_mace_scf(args, calc_type=calc_type)
    except KeyboardInterrupt:
        print("\nMACE SCF calculation interrupted by user")
        raise
    except Exception as e:
        print(f"\nMACE SCF calculation failed: {e}")
        raise