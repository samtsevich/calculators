from ase.calculators.dftb import Dftb
from ase.io.trajectory import Trajectory
from ase.io.vasp import write_vasp

from common.dftb import get_additional_params, get_args, get_KPoints


def dftb_scf(args):

    assert args.command == 'dftb', 'This function is only for DFTB'
    calc_type = args.subcommand

    args = get_args(args)
    name = args['name']
    structures = args['structures']

    outdir = args['outdir']
    calc_fold = outdir

    params = args['dftb_params']
    params.update(get_additional_params(type=calc_type))


    # ? add 'stress'
    traj = Trajectory(outdir/'out.traj', 'w', properties=['energy', 'forces'])

    for i, structure in enumerate(structures):
        ID = f'{name}_{i}'

        kpts = get_KPoints(args['kspacing'], structure.get_cell())
        params.update({'label': f'out_{calc_type}_{ID}',
                       'kpts': kpts,})

        scf_calc = Dftb(directory=calc_fold,
                        **params)

        structure.write(outdir/f'a_{ID}.gen')
        structure.set_calculator(scf_calc)
        e = structure.get_potential_energy()
        write_vasp(outdir/f'final_{ID}.vasp', structure,
                   sort=True, vasp5=True, direct=True)
        # print(structure.get_forces())
        traj.write(structure, energy=e, forces=structure.get_forces())
        print(f'SCF of {ID} is done.')
    traj.close()

# if __name__ == "__main__":

#     calc_type = 'scf'
#     args = get_args(calc_type=calc_type)
#     dftb_scf(args)
