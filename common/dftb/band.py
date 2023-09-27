from ase.calculators.dftb import Dftb
from ase.io.vasp import write_vasp

from common.dftb import (get_args,
                         get_additional_params)
from common import get_N_val_electrons, fix_fermi_level


def dftb_band(args):
    args = get_args(args)
    name = args['name']
    structure = args['structure']
    species = set(structure.get_chemical_symbols())

    outdir = args['outdir']
    outdir.mkdir(parents=True, exist_ok=True)

    params = args['dftb_params']
    params.update(get_additional_params(type='scf'))
    params.update({'label': f'scf_{name}',})

    calc_fold = outdir
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

    # Fix Fermi level
    N_val_e = get_N_val_electrons(structure)
    # bs = fix_fermi_level(bs, N_val_e)
    bs = bs.subtract_reference()
    bs.write(outdir/f'bs_{name}.json')
    emin=-20
    emax=25
    bs.plot(filename=outdir/f'bs_{name}.png', emin=emin, emax=emax)

    print(f'\tStep 2 for {name} done')
