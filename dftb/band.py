from pathlib import Path

from ase.calculators.dftb import Dftb
from ase.io.vasp import write_vasp

from common_dftb import (get_args,
                         get_additional_params)


if __name__ == "__main__":

    args = get_args(calc_type='band')

    name = args['name']
    structure = args['structure']
    species = set(structure.get_chemical_symbols())

    outdir = args['outdir']

    params = args['dftb_params']
    params.update(get_additional_params(type='scf'))
    params.update({'label': f'scf_{name}',})

    calc_fold = outdir/'scf'
    structure.write(calc_fold/f'a_{name}.gen')
    write_vasp(calc_fold/f'a_{name}.vasp', structure,
               sort=True, vasp5=True, direct=True)

    scf_calc = Dftb(atoms=structure,
                    directory=calc_fold,
                    **params)
    structure.calc = scf_calc
    e = structure.get_potential_energy()
    fermi_level = scf_calc.get_fermi_level()

    print(f'\tStep 1 for {name} done')

    # Step 2.
    calc_fold = outdir/'band'
    path = structure.cell.bandpath()
    print(path)
    scf_calc.calculate(structure)
    params.update(get_additional_params(type='band'))
    params.update({'label': f'band_{name}',})

    assert params['Hamiltonian_SCC'] == 'No'

    # Stupid ASE does not recognize k-points for band structures, when there is no 'path' key in the dict
    params.update({'kpts': {**path.todict(), 'path': ''}})

    band_calc = Dftb(atoms=structure,
                     directory=calc_fold,
                     **params)
    band_calc.calculate(structure)

    bs = band_calc.band_structure()
    bs = bs.subtract_reference()
    bs.write(outdir/f'bs_{name}.json')

    print(f'\tStep 2 for {name} done')
