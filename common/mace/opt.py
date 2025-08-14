#!/usr/bin/python3

from typing import List

import numpy as np
from ase.atoms import Atoms
from ase.filters import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp
from ase.optimize import BFGS

from . import get_args


def cleanup_temp_files(temp_files):
    """Clean up temporary files on error.

    Args:
        temp_files: List of Path objects to clean up
    """
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            # Ignore cleanup errors to avoid masking the original error
            pass


def run_mace_opt(args: dict, calc_type: str):
    """Run MACE structural optimization.

    Args:
        args: Dictionary of processed command line arguments
        calc_type: Type of calculation (should be 'opt')

    Raises:
        AssertionError: If calc_type is not 'opt'
        ImportError: If MACE dependencies are not available
        RuntimeError: If calculation fails
    """
    # Validate calculation type
    assert calc_type == "opt", "This function is only for optimization calculation"

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
    name = args["name"]
    structures: List[Atoms] = args["structures"]
    structure = structures[-1]  # Use the last structure (following existing patterns)

    outdir = args["outdir"]
    model_path = args["model_path"]
    device = args["device"]
    default_dtype = args["default_dtype"]
    compile_mode = args["compile_mode"]

    # Optimization parameters
    fmax = args["fmax"]
    assert fmax > 0, "fmax should be positive"

    nsteps = args["nsteps"]
    assert nsteps > 0, "nsteps should be positive"

    is_full_opt = args["full_opt"]

    # Print input parameters (following existing patterns)
    print(30 * "-")
    if is_full_opt:
        print(f"Full optimization calculation (cell + atomic positions) for {name}")
    else:
        print(f"Optimization of atomic positions for {name}")
    print(f"MACE model: {model_path}")
    print(f"Device: {device}")
    print(f"Data type: {default_dtype}")
    if compile_mode:
        print(f"Compile mode: {compile_mode}")
    print(f"fmax = {fmax}, nsteps = {nsteps}")
    print(f"Output directory: {outdir}")
    print(30 * "-")

    # Temporary files to clean up on error
    temp_files = [
        outdir / "optimization.pckl",
        outdir / "optimization.traj",
        outdir / "optimization.log",
    ]

    try:
        # Load MACE model and set up MACECalculator with specified parameters
        calc_params = {
            "model_paths": [str(model_path)],
            "device": device,
            "default_dtype": default_dtype,
        }

        # Add compile_mode if specified
        if compile_mode is not None:
            calc_params["compile_mode"] = compile_mode

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

        # Attach calculator to structure
        structure.calc = calculator

        # Configure optimization target (positions only vs full cell+positions)
        try:
            if is_full_opt:
                opt_struct = UnitCellFilter(structure)
            else:
                opt_struct = structure
        except Exception as e:
            raise RuntimeError(f"Failed to set up optimization structure: {e}")

        # Set up BFGS optimizer with trajectory and log file outputs
        try:
            opt = BFGS(
                opt_struct,
                restart=str(outdir / "optimization.pckl"),
                trajectory=str(outdir / "optimization.traj"),
                logfile=str(outdir / "optimization.log"),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BFGS optimizer: {e}")

        # Run optimization with user-specified fmax and nsteps parameters
        try:
            converged = opt.run(fmax=fmax, steps=nsteps)
            if not converged:
                print(f"Warning: Optimization did not converge within {nsteps} steps")
                print("Consider increasing --nsteps or relaxing --fmax")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                raise RuntimeError(
                    f"GPU out of memory during optimization.\n"
                    f"Error: {e}\n"
                    "Try using --device cpu or a smaller structure."
                )
            elif "nan" in str(e).lower() or "inf" in str(e).lower():
                raise RuntimeError(
                    f"Optimization failed due to invalid results (NaN/Inf).\n"
                    f"Error: {e}\n"
                    "This may indicate an incompatible structure or model."
                )
            else:
                raise RuntimeError(f"Optimization failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during optimization: {e}")

        # Get final energy for output
        try:
            final_energy = structure.get_potential_energy()
            final_forces = structure.get_forces()
        except Exception as e:
            raise RuntimeError(f"Failed to get final energy and forces: {e}")

        # Validate final results
        if not isinstance(final_energy, (int, float)) or not np.isfinite(final_energy):
            raise RuntimeError(
                f"Optimization produced invalid final energy: {final_energy}"
            )

        if not isinstance(final_forces, np.ndarray) or not np.all(
            np.isfinite(final_forces)
        ):
            raise RuntimeError("Optimization produced invalid final forces")

        print(f"Final energy: {final_energy:.6f} eV")
        print(f"Max final force: {max(abs(final_forces.flatten())):.6f} eV/Å")

    except RuntimeError:
        # Clean up temporary files on error
        cleanup_temp_files(temp_files)
        raise
    except Exception as e:
        # Clean up temporary files on error
        cleanup_temp_files(temp_files)
        raise RuntimeError(f"Unexpected error in MACE optimization: {e}")

    # Save final optimized structure in VASP format
    try:
        write_vasp(outdir / "final.vasp", structure, sort=True, vasp5=True, direct=True)
    except PermissionError:
        raise RuntimeError(
            f"Permission denied writing to {outdir / 'final.vasp'}. Check directory permissions."
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to write final structure to {outdir / 'final.vasp'}: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error writing final structure: {e}")

    # Create final trajectory file with energy and forces
    try:
        with Trajectory(
            outdir / "final.traj", "w", properties=["energy", "forces"]
        ) as final_traj:
            final_traj.write(structure, energy=final_energy, forces=final_forces)
    except PermissionError:
        raise RuntimeError(
            f"Permission denied writing to {outdir / 'final.traj'}. Check directory permissions."
        )
    except OSError as e:
        raise RuntimeError(
            f"Failed to write final trajectory file to {outdir / 'final.traj'}: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Unexpected error writing final trajectory file: {e}")

    # Generate completion message with final energy
    print(f"Optimization of {name} is done.")
    print(30 * "-")


def mace_opt(args):
    """CLI wrapper function for MACE optimization calculation.

    Args:
        args: Parsed command line arguments

    Raises:
        AssertionError: If command is not 'mace'
    """
    assert args.command == "mace", "This function is only for MACE"
    calc_type: str = args.subcommand
    args: dict = vars(args)

    try:
        run_mace_opt(args, calc_type=calc_type)
    except KeyboardInterrupt:
        print("\nMACE optimization interrupted by user")
        raise
    except Exception as e:
        print(f"\nMACE optimization failed: {e}")
        raise
