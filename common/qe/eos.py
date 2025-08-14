from copy import copy
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from ase.calculators.espresso import Espresso
from ase.eos import EquationOfState
from ase.io import read
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.units import kJ

from . import get_args
from .opt import run_qe_opt


def qe_eos(args):
    assert args.command == "qe", "This function is only for QE calculations"
    assert args.subcommand == "eos", "This function is only for EOS calculation"

    args: dict = get_args(args)

    run_qe_eos(args)


def run_qe_eos(args: dict):
    # Reading the arguments
    # ------------------------------------------------

    name = args["name"]
    outdir = copy(args["outdir"])

    # Printing input parameters
    # ------------------------------------------------
    print(30 * "-")
    print(f"EOS calculation for {name}")
    print(f"kspacing = {args['kspacing']}")
    print(f"Output directory: {args['outdir']}")
    print(30 * "-")

    # Step 0. Full optimization of the reference structure
    # ------------------------------------------------
    new_args = copy(args)
    new_args["full_opt"] = True
    new_args["outdir"] = args["outdir"] / "orig_struct"
    run_qe_opt(new_args)
    optimized_struct: Atoms = read(new_args["outdir"] / "final.traj", index=-1)

    traj_file = outdir / "res_eos.traj"
    res_eos_traj = Trajectory(filename=traj_file, mode="w")

    # Step 1. EOS part
    volumes, energies = [], []
    cell = optimized_struct.get_cell()
    for x in np.linspace(0.96, 1.04, 9):
        structure = optimized_struct.copy()
        structure.set_cell(cell * x, scale_atoms=True)
        volumes.append(structure.get_volume())

        new_args["structures"] = [structure]
        new_args["outdir"] = outdir / f"{x:.2f}"
        new_args["full_opt"] = False
        run_qe_opt(new_args)

        calc_struct: Atoms = read(new_args["outdir"] / "final.traj", index=-1)
        energies.append(calc_struct.get_potential_energy())
        res_eos_traj.write(atoms=calc_struct)
    res_eos_traj.close()

    # Step 2. Fitting EOS
    # ------------------------------------------------
    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    v0, e0, B = eos.fit()
    print(B / kJ * 1.0e24, "GPa")

    # Step 3. Plotting and saving the results
    # ------------------------------------------------
    plot_data = eos.getplotdata()
    v_fit = plot_data[-4] / len(structure)
    e_fit = plot_data[-3] / len(structure)
    v = plot_data[-1] / len(structure)
    e = plot_data[-2] / len(structure)
    np.savetxt(outdir / "e_vs_v_sc.dat", np.column_stack((v, e)))
    np.savetxt(outdir / "e_vs_v_sc_fit.dat", np.column_stack((v_fit, e_fit)))
    eos.plot(outdir / "eos.png")

    # Final output message
    # ------------------------------------------------
    print(f"Data for EOS for {name} has been collected")
    # print(f'SCF of {name} is done.')
    print(30 * "-")


def qe_eos_old(args):
    args = get_args(args)
    name = args["input"].stem
    structures = args["structures"]

    options = args["options"]
    pp = args["pseudopotentials"]
    pp_dir = args["pp_dir"]
    kspacing = args["kspacing"]

    outdir = Path(args["outdir"])
    calc_fold = outdir

    data = args["data"]
    data["control"]["calculation"] = "vc-relax"
    data["control"]["outdir"] = "./tmp"
    data["control"]["prefix"] = str(name)

    opt_calc = Espresso(
        input_data=data,
        pseudopotentials=pp,
        pseudo_dir=str(pp_dir),
        kspacing=kspacing,
        directory=str(calc_fold / "opt"),
    )

    for i, structure in enumerate(structures):
        ID = f"{name}_{i}"

        structure.calc = opt_calc

        # add rattling to the atomic positions
        # add_coords = 0.05 - 0.1 * np.random.rand(len(atoms), 3)
        # new_coords = atoms.get_scaled_positions() + add_coords
        # atoms.set_scaled_positions(new_coords)

        # add rattling to the cell
        # add_cell = 0.1 * np.random.rand(3,3)
        # new_cell = atoms.get_cell() + add_cell
        # atoms.set_cell(new_cell, scale_atoms=True)

        print(structure.get_potential_energy())
        print(f"Optimization of {ID} is done.")

        # SCF calculator
        data["control"]["calculation"] = "relax"
        relax_calc = Espresso(
            input_data=data,
            pseudopotentials=pp,
            pseudo_dir=str(pp_dir),
            kspacing=kspacing,
        )
        structure.calc = relax_calc

        traj_file = outdir / f"res_{ID}.traj"
        traj = TrajectoryWriter(filename=traj_file, mode="w")

        # EOS part
        volumes, energies = [], []
        cell = structure.get_cell()
        for x in np.linspace(0.92, 1.08, 9):
            structure.calc.directory = calc_fold / f"{x:.2f}"
            structure.set_cell(cell * x, scale_atoms=True)
            volumes.append(structure.get_volume())
            energies.append(structure.get_potential_energy())
            traj.write(atoms=structure)
        traj.close()

        print(f"Data for EOS for {ID} has been collected")

        configs = read(traj_file)
        # Extract volumes and energies:
        # volumes = [x.get_volume() for x in configs]
        # energies = [x.get_potential_energy() for x in configs]
        eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
        v0, e0, B = eos.fit()
        print(B / kJ * 1.0e24, "GPa")
        plot_data = eos.getplotdata()
        v_fit = plot_data[-4] / len(structure)
        e_fit = plot_data[-3] / len(structure)
        v = plot_data[-1] / len(structure)
        e = plot_data[-2] / len(structure)
        np.savetxt(outdir / "e_vs_v_sc.dat", np.column_stack((v, e)))
        np.savetxt(outdir / "e_vs_v_sc_fit.dat", np.column_stack((v_fit, e_fit)))
        eos.plot(outdir / "Ru_sc_eos.png")
