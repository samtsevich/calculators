from copy import copy

from ase.calculators.dftb import Dftb
from ase.io.vasp import write_vasp
from ase.spectrum.band_structure import get_band_structure

from common import fix_fermi_level, get_N_val_electrons
from common.dftb import get_additional_params, get_args, get_KPoints


def dftb_band(args):
    args = get_args(args)
    name = args['name']
    structures = args['structures']

    outdir = args['outdir']
    outdir.mkdir(parents=True, exist_ok=True)

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        kpts = get_KPoints(args['kspacing'], structure.get_cell())

        params = copy(args['dftb_params'])
        params.update(get_additional_params(type='scf'))
        params.update({'label': f'out_scf_{ID}',
                       'kpts': kpts,})

        calc_fold = outdir

        scf_calc = Dftb(atoms=structure,
                        directory=calc_fold,
                        **params)

        structure.write(calc_fold/f'a_{ID}.gen')
        write_vasp(calc_fold/f'a_{ID}.vasp', structure,
                sort=True, vasp5=True, direct=True)

        structure.calc = scf_calc
        e = structure.get_potential_energy()
        fermi_level = scf_calc.get_fermi_level()

        print(f'\tStep 1 for {ID} done')

        # Step 2.
        path = structure.cell.bandpath(npoints=100)
        print(path)
        params.update(get_additional_params(type='band'))
        params.update({'label': f'out_band_{ID}',})

        assert params['Hamiltonian_SCC'] == 'Yes'
        assert params['Hamiltonian_MaxSCCIterations'] == 1

        # Stupid ASE does not recognize k-points for band structures, when there is no 'path' key in the dict
        params.update({'kpts': {**path.todict(), 'path': ''}})

        band_calc = Dftb(atoms=structure, directory=calc_fold, **params)
        band_calc.calculate(structure)
        bs = get_band_structure(atoms=structure, calc=band_calc)

        # Fix Fermi level
        N_val_e = get_N_val_electrons(structure)
        bs = fix_fermi_level(bs, N_val_e).subtract_reference()
        bs.write(outdir/f'bs_{ID}.json')
        emin=-20
        emax=25
        bs.plot(filename=outdir/f'bs_{ID}.png', emin=emin, emax=emax)

        print(f'\tStep 2 for {ID} done')
