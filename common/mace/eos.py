#!/usr/bin/python3

from copy import copy
from typing import List

import numpy as np
from ase.atoms import Atoms
from ase.eos import EquationOfState
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.units import kJ

from . import get_args
from .opt import run_mace_opt


def cleanup_eos_temp_files(outdir, scales):
    """Clean up temporary EOS calculation files on error.
    
    Args:
        outdir: Output directory Path
        scales: List of scale factors that were attempted
    """
    import shutil
    
    # Clean up scale directories
    for scale in scales:
        scale_dir = outdir / f'{scale:.2f}'
        try:
            if scale_dir.exists():
                shutil.rmtree(scale_dir)
        except Exception:
            pass
    
    # Clean up other temporary files
    temp_files = [
        outdir / 'res_eos.traj',
        outdir / 'e_vs_v_sc.dat',
        outdir / 'e_vs_v_sc_fit.dat',
        outdir / 'eos.png'
    ]
    
    for temp_file in temp_files:
        try:
            if temp_file.exists():
                temp_file.unlink()
        except Exception:
            pass


def run_mace_eos(args: dict, calc_type: str):
    """Run MACE equation of state calculation.
    
    Args:
        args: Dictionary of processed command line arguments
        calc_type: Type of calculation (should be 'eos')
        
    Raises:
        AssertionError: If calc_type is not 'eos'
        ImportError: If MACE dependencies are not available
        RuntimeError: If calculation fails
    """
    # Validate calculation type
    assert calc_type == 'eos', 'This function is only for EOS calculation'

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
    outdir = copy(args['outdir'])
    model_path = args['model_path']
    device = args['device']
    default_dtype = args['default_dtype']
    compile_mode = args['compile_mode']

    # Print input parameters (following existing patterns)
    print(30 * '-')
    print(f'EOS calculation for {name}')
    print(f'MACE model: {model_path}')
    print(f'Device: {device}')
    print(f'Data type: {default_dtype}')
    if compile_mode:
        print(f'Compile mode: {compile_mode}')
    print(f'Output directory: {outdir}')
    print(30 * '-')

    # Step 0. Full optimization of the reference structure
    # ------------------------------------------------
    scales = np.linspace(0.96, 1.04, 9)  # For cleanup purposes
    
    try:
        new_args = copy(args)
        new_args['outdir'] = args['outdir'] / 'orig_struct'
        
        # We need to perform full optimization of the reference structure
        new_args['full_opt'] = True
        
        try:
            run_mace_opt(new_args, calc_type='opt')
        except Exception as e:
            raise RuntimeError(f"Failed to optimize reference structure for EOS: {e}")
        
        try:
            optimized_struct: Atoms = read(new_args['outdir'] / 'final.traj', index=-1)
        except Exception as e:
            raise RuntimeError(f"Failed to read optimized reference structure: {e}")
            
        # Validate optimized structure
        if len(optimized_struct) == 0:
            raise RuntimeError("Optimized reference structure is empty")
            
    except RuntimeError:
        cleanup_eos_temp_files(outdir, scales)
        raise
    except Exception as e:
        cleanup_eos_temp_files(outdir, scales)
        raise RuntimeError(f"Unexpected error in reference structure optimization: {e}")

    try:
        traj_file = outdir / 'res_eos.traj'
        res_eos_traj = Trajectory(filename=traj_file, mode='w')
    except Exception as e:
        cleanup_eos_temp_files(outdir, scales)
        raise RuntimeError(f"Failed to create EOS trajectory file: {e}")

    # Step 1. EOS part - Generate volume-scaled structures and calculate energies
    # ------------------------------------------------
    volumes, energies = [], []
    
    try:
        cell = optimized_struct.get_cell()
        
        # Validate cell
        if np.any(np.isnan(cell)) or np.any(np.isinf(cell)):
            raise RuntimeError("Reference structure has invalid cell parameters")
            
        if np.linalg.det(cell) <= 0:
            raise RuntimeError("Reference structure has invalid cell (zero or negative volume)")
    
    except Exception as e:
        res_eos_traj.close()
        cleanup_eos_temp_files(outdir, scales)
        raise RuntimeError(f"Failed to get cell from reference structure: {e}")
    
    # Generate volume-scaled structures from 0.96 to 1.04 with 9 points
    completed_scales = []
    try:
        for i, x in enumerate(scales):
            try:
                structure = optimized_struct.copy()
                structure.set_cell(cell * x, scale_atoms=True)
                volume = structure.get_volume()
                
                # Validate volume
                if not np.isfinite(volume) or volume <= 0:
                    raise RuntimeError(f"Invalid volume {volume} for scale {x}")
                    
                volumes.append(volume)

                # Calculate energies for each scaled structure using position optimization
                new_args['structures'] = [structure]
                new_args['outdir'] = outdir / f'{x:.2f}'
                new_args['full_opt'] = False  # Only optimize positions, not cell
                
                try:
                    run_mace_opt(new_args, calc_type='opt')
                except Exception as e:
                    raise RuntimeError(f"Failed optimization for scale {x:.2f}: {e}")

                # Collect volume and energy data for EOS fitting
                try:
                    calc_struct: Atoms = read(new_args['outdir'] / 'final.traj', index=-1)
                    energy = calc_struct.get_potential_energy()
                    
                    # Validate energy
                    if not np.isfinite(energy):
                        raise RuntimeError(f"Invalid energy {energy} for scale {x:.2f}")
                        
                    energies.append(energy)
                    res_eos_traj.write(atoms=calc_struct)
                    completed_scales.append(x)
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to read results for scale {x:.2f}: {e}")
                    
                print(f"Completed scale {x:.2f} ({i+1}/{len(scales)})")
                
            except RuntimeError as e:
                res_eos_traj.close()
                cleanup_eos_temp_files(outdir, completed_scales)
                raise RuntimeError(f"EOS calculation failed at scale {x:.2f}: {e}")
        
        res_eos_traj.close()
        
    except Exception as e:
        res_eos_traj.close()
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f"Unexpected error during EOS volume scaling: {e}")
    
    # Validate we have enough data points
    if len(volumes) < 5:
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f"Insufficient data points for EOS fitting: {len(volumes)} < 5")

    # Step 2. Fitting EOS - Fit Birch-Murnaghan equation of state
    # ------------------------------------------------
    try:
        eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
        v0, e0, B = eos.fit()
        
        # Validate fit results
        if not all(np.isfinite([v0, e0, B])):
            raise RuntimeError(f"EOS fit produced invalid results: v0={v0}, e0={e0}, B={B}")
            
        if B <= 0:
            raise RuntimeError(f"EOS fit produced negative bulk modulus: {B}")
        
    except Exception as e:
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f"Failed to fit equation of state: {e}")
    
    # Calculate and output bulk modulus in GPa
    try:
        bulk_modulus_gpa = B / kJ * 1.0e24
        print(f'Bulk modulus: {bulk_modulus_gpa:.2f} GPa')
    except Exception as e:
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f"Failed to calculate bulk modulus: {e}")

    # Step 3. Plotting and saving the results - Generate energy vs volume data and EOS plot
    # ------------------------------------------------
    try:
        plot_data = eos.getplotdata()
        v_fit = plot_data[-4] / len(optimized_struct)
        e_fit = plot_data[-3] / len(optimized_struct)
        v = plot_data[-1] / len(optimized_struct)
        e = plot_data[-2] / len(optimized_struct)
        
        # Validate plot data
        if not all(np.all(np.isfinite(arr)) for arr in [v_fit, e_fit, v, e]):
            raise RuntimeError("EOS plot data contains invalid values")
        
    except Exception as e:
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f"Failed to generate EOS plot data: {e}")
    
    # Save energy vs volume data files
    try:
        np.savetxt(outdir / 'e_vs_v_sc.dat', np.column_stack((v, e)))
        np.savetxt(outdir / 'e_vs_v_sc_fit.dat', np.column_stack((v_fit, e_fit)))
    except PermissionError:
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f'Permission denied writing EOS data files to {outdir}. Check directory permissions.')
    except OSError as e:
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f'Failed to write EOS data files: {e}')
    except Exception as e:
        cleanup_eos_temp_files(outdir, completed_scales)
        raise RuntimeError(f'Unexpected error writing EOS data files: {e}')
    
    # Generate EOS plot
    try:
        eos.plot(outdir / 'eos.png')
    except Exception as e:
        # Plot failure is not critical, just warn
        print(f"Warning: Failed to generate EOS plot: {e}")
        print("EOS data files were saved successfully.")

    # Final output message
    # ------------------------------------------------
    print(f'Data for EOS for {name} has been collected')
    print(30 * '-')


def mace_eos(args):
    """CLI wrapper function for MACE EOS calculation.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        AssertionError: If command is not 'mace'
    """
    assert args.command == 'mace', 'This function is only for MACE'
    calc_type: str = args.subcommand
    args: dict = vars(args)

    try:
        run_mace_eos(args, calc_type=calc_type)
    except KeyboardInterrupt:
        print("\nMACE EOS calculation interrupted by user")
        raise
    except Exception as e:
        print(f"\nMACE EOS calculation failed: {e}")
        raise