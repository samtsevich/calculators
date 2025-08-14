#!/usr/bin/python3

from pathlib import Path
from shutil import move

from ase.calculators.vasp import Vasp
from ase.io.vasp import write_vasp
from ase.spectrum.band_structure import BandStructure, get_band_structure

from . import get_basic_params, get_args, get_total_N_val_e


def run_vasp_band(args: dict):
    name = args["name"]
    structures = args["structures"]

    outdir = Path(args["outdir"])
    calc_fold = outdir

    scf_params = get_basic_params(args)
    scf_params.update({"directory": calc_fold})
    scf_calc = Vasp(**scf_params)

    for i, structure in enumerate(structures):
        ID = f"{name}_{i}"

        # 2. SCF #
        structure.calc = scf_calc
        e = structure.get_potential_energy()
        fermi_level = scf_calc.get_fermi_level()
        write_vasp(
            outdir / f"final_{ID}.vasp", structure, sort=True, vasp5=True, direct=True
        )

        N_val_e: int = get_total_N_val_e(calc_fold / "OUTCAR")
        assert N_val_e is not None
        print(f"Total N valence electrons: {N_val_e}")
        print(f"Fermi level: {fermi_level}")
        print("---------------------------")

        move(calc_fold / "INCAR", outdir / f"INCAR.scf.{ID}")
        move(calc_fold / "OUTCAR", outdir / f"OUTCAR.scf.{ID}")

        # 3. BAND STRUCTURE #
        path = structure.cell.bandpath(npoints=200)
        print(f"BandPath: {path}")

        nscf_params = scf_params.copy()
        del nscf_params["kspacing"]
        nscf_params.update(
            {
                "icharg": 11,
                # # Stupid ASE does not recognize k-points for band structures, when there is no 'path' key in the dict
                "kpts": {**path.todict(), "path": ""},
            }
        )

        nscf_calc = Vasp(**nscf_params)
        structure.calc = nscf_calc
        e = structure.get_potential_energy()

        move(calc_fold / "INCAR", outdir / f"INCAR.band.{ID}")
        move(calc_fold / "OUTCAR", outdir / f"OUTCAR.band.{ID}")

        # bs = BandStructure(structure.calc.get_eigenvalues(), path.kpts, reference=structure.calc.get_fermi_level())
        bs: BandStructure = get_band_structure(structure, calc=nscf_calc)
        bs._reference = fermi_level
        # bs: BandStructure = fix_fermi_level(bs, N_val_e).subtract_reference()
        bs.write(outdir / f"bs_{ID}.json")

        bs = bs.subtract_reference()
        bs.plot(filename=outdir / f"bs_{ID}.png")

        print(f"Band structure of {ID} is calculated.")
        print("---------------------------")


def vasp_band(args):
    assert args.command == "vasp", "This function is only for VASP"
    args: dict = get_args(args)

    run_vasp_band(args)
