#!/usr/bin/python3

import numpy as np

from ase.calculators.espresso import Espresso
from ase.eos import EquationOfState
from ase.io import read
from ase.io.trajectory import Trajectory, TrajectoryWriter
from ase.units import kJ

from pathlib import Path

from common_qe import get_args


if __name__ == '__main__':

    args = get_args(calc_type='eos')

    name = args['input'].stem
    structure = args['structure']

    options = args['options']
    pp = args['pseudopotentials']
    pp_dir = args['pp_dir']
    kspacing = args['kspacing']

    outdir = Path(args['outdir'])
    calc_fold = outdir

    data = args['data']
    data['control']['calculation'] = 'vc-relax'
    data['control']['outdir'] = './tmp'
    data['control']['prefix'] = str(name)

    opt_calc = Espresso(input_data=data,
                        pseudopotentials=pp,
                        pseudo_dir=str(pp_dir),
                        kspacing=kspacing,
                        directory=str(calc_fold/'opt'))
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
    print(f'Optimization of {name} is done.')

    # SCF calculator
    data['control']['calculation'] = 'relax'
    relax_calc = Espresso(input_data=data,
                          pseudopotentials=pp,
                          pseudo_dir=str(pp_dir),
                          kspacing=kspacing)
    structure.calc = relax_calc

    traj_file = outdir/'res.traj'
    traj = TrajectoryWriter(filename=traj_file, mode='w')

    # EOS part
    volumes, energies = [], []
    cell = structure.get_cell()
    for x in np.linspace(0.92, 1.08, 9):
        structure.calc.directory = calc_fold/f"{x:.2f}"
        structure.set_cell(cell * x, scale_atoms=True)
        volumes.append(structure.get_volume())
        energies.append(structure.get_potential_energy())
        traj.write(atoms=structure)
    traj.close()

    print(f'Data for EOS for {name} has been collected')

    configs = read(traj_file)
    # Extract volumes and energies:
    # volumes = [x.get_volume() for x in configs]
    # energies = [x.get_potential_energy() for x in configs]
    eos = EquationOfState(volumes, energies, eos='birchmurnaghan')
    v0, e0, B = eos.fit()
    print(B / kJ * 1.0e24, 'GPa')
    plot_data = eos.getplotdata()
    v_fit = plot_data[-4]/len(structure)
    e_fit = plot_data[-3]/len(structure)
    v = plot_data[-1]/len(structure)
    e = plot_data[-2]/len(structure)
    np.savetxt(outdir/'e_vs_v_sc.dat', np.column_stack((v, e)))
    np.savetxt(outdir/'e_vs_v_sc_fit.dat', np.column_stack((v_fit, e_fit)))
    eos.plot(outdir/'Ru_sc_eos.png')
