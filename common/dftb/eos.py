from copy import copy

import numpy as np
from ase.atoms import Atoms
from ase.eos import EquationOfState
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.units import kJ

from common.dftb import get_args, get_calc_type_params
from common.dftb.opt import run_opt_dftb


def run_dftb_eos(args: dict, calc_type: str):
    # Reading the arguments
    # ------------------------------------------------
    assert calc_type == 'eos', 'This function is only for EOS calculation'

    args = get_args(args, calc_type=calc_type)
    name = args['name']
    outdir = copy(args['outdir'])

    # Printing input parameters
    # ------------------------------------------------
    print(30 * '-')
    print(f'EOS calculation for {name}')
    print(f'kspacing = {args["kspacing"]}')
    print(f'Output directory: {args["outdir"]}')
    print(30 * '-')

    # Step 0. Full optimization of the reference structure
    # ------------------------------------------------
    new_args = copy(args)

    new_args['outdir'] = args['outdir'] / 'orig_struct'
    run_opt_dftb(new_args, calc_type='opt')
    optimized_struct: Atoms = read(new_args['outdir'] / 'final.traj', index=-1)

    traj_file = outdir / 'res_eos.traj'
    res_eos_traj = Trajectory(filename=traj_file, mode='w')

    # Step 1. EOS part
    volumes, energies = [], []
    cell = optimized_struct.get_cell()
    for x in np.linspace(0.96, 1.04, 9):
        structure = optimized_struct.copy()
        structure.set_cell(cell * x, scale_atoms=True)
        volumes.append(structure.get_volume())

        new_args['structures'] = [structure]
        new_args['outdir'] = outdir / f'{x:.2f}'
        new_args['full_opt'] = False
        run_opt_dftb(new_args, calc_type='opt')

        calc_struct: Atoms = read(new_args['outdir'] / 'final.traj', index=-1)
        energies.append(calc_struct.get_potential_energy())
        res_eos_traj.write(atoms=calc_struct)
    res_eos_traj.close()

    # Step 2. Fitting EOS
    # ------------------------------------------------
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0, e0, B = eos.fit()
    print(B / kJ * 1.0e24, 'GPa')

    # Step 3. Plotting and saving the results
    # ------------------------------------------------
    plot_data = eos.getplotdata()
    v_fit = plot_data[-4] / len(structure)
    e_fit = plot_data[-3] / len(structure)
    v = plot_data[-1] / len(structure)
    e = plot_data[-2] / len(structure)
    np.savetxt(outdir / 'e_vs_v_sc.dat', np.column_stack((v, e)))
    np.savetxt(outdir / 'e_vs_v_sc_fit.dat', np.column_stack((v_fit, e_fit)))
    eos.plot(outdir / 'eos.png')

    # Final output message
    # ------------------------------------------------
    print(f'Data for EOS for {name} has been collected')
    # print(f'SCF of {name} is done.')
    print(30 * '-')


def dftb_eos(args):
    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type: str = args.subcommand
    args: dict = vars(args)

    run_dftb_eos(args, calc_type=calc_type)
