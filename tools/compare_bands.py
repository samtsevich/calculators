import argparse
from math import ceil
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure as BS
from matplotlib.pyplot import cm
from scipy.spatial import distance_matrix

# from ..common import fix_fermi_level, get_N_val_electrons
# from ..common.qe import read_valences

FONT_SIZE = 20
Y_AXIS_STEP = 2


def read_band_structure(filename):
    bs = read_json(filename)
    if not isinstance(bs, BS):
        raise TypeError(f'Expected band structure, but file contains: {bs}')
    return bs


def special_points_energies(band_structure: BS) -> Dict[str, list]:
    '''
    Extract the energies of the special points from the band structure.
    '''

    round = lambda x: np.mod(np.round(x, 6), 1)

    special_pts = band_structure.path.special_points
    sp_pts_names = [key for key in special_pts.keys() if len(key) == 1]
    kpts = round(band_structure.path.kpts)
    energies = band_structure.energies

    kpts_energies: Dict[str, list] = {}
    for sp_pt_name, sp_pt_coord in special_pts.items():
        # Skipping the special points with the same name,
        # e.g. if we have `L` and `L1` on a band structure, we skip `L1`,
        # because it is the same as `L`
        if len(sp_pt_name) > 1:
            continue

        for i, kpt in enumerate(kpts):
            if np.allclose(round(sp_pt_coord), kpt, atol=1e-3):
                kpts_energies[sp_pt_name] = energies[:, i, :]
                break
    msg = 'The number of special points is not the same as the number of k-points'
    assert len(kpts_energies) == len(sp_pts_names), msg
    return kpts_energies


def decode_band(bandstructure: BS, nbands: int):
    """
    Decode the whole band structure into a distance matrix.
    """
    assert nbands > 0, "Number of bands should be larger than 0"
    spp_e = special_points_energies(bandstructure)
    feature = []
    for i, (key, e) in enumerate(spp_e.items()):
        for en in e[:nbands]:
            feature.append([i, en])
        # feature[0].extend(np.full(nbands, i))
        # feature[1].extend(e[:nbands])
    feature = np.array(feature)
    x = distance_matrix(feature, feature)
    return x


def align_band_structure(band_structure: BS, N_val_e: int, fermi_label: Optional[str] = None, **kwargs) -> BS:
    assert N_val_e > 0, 'Number of valence electrons should be larger than 0'

    homo_id, lumo_id = ceil(N_val_e / 2) - 1, ceil(N_val_e / 2)
    lumo_e = band_structure.energies[:, :, lumo_id]
    homo_e = band_structure.energies[:, :, homo_id]

    if fermi_label is not None:
        assert fermi_label in band_structure.path.special_points
        # align to chosen/arbitrary special point;
        # e.g. for conductors: if the position of the fermi level in the bandpath is known, align to it
        kpts_energies: Dict[str, list] = special_points_energies(band_structure)
        # Aligning always by the spin up bands
        band_structure._reference = kpts_energies[fermi_label][0, homo_id]
    else:
        # Aligning always by the spin up bands
        band_structure._reference = np.max(homo_e[0])

    return band_structure


def decode_band2(bandstructure: BS, n_val_e: int, n_cond_bands: int = 2):
    """
    Decode the cunduction and valence bands separately.

    Input:
        bandstructure: BandStructure object
        nbands: number of counduction bands to be decoded.
                normanlly it is all valence bands and several conduction bands
    Output:
        x: distance matrix of the band structure
    """

    assert n_val_e > 0, "Number of valence electrons should be larger than 0"
    assert n_cond_bands > 0, "Number of conduction bands to be decoded should be larger than 0"
    spp_e = special_points_energies(bandstructure)

    val_band_feature = []
    cond_band_feature = []
    for i, (key, e) in enumerate(spp_e.items()):
        val_band_feature.extend([[i, en] for en in e[:n_val_e]])
        cond_band_feature.extend([[i, en] for en in e[n_val_e + 1 : n_val_e + n_cond_bands]])
    val_band_feature = np.array(val_band_feature)
    cond_band_feature = np.array(cond_band_feature)
    val_x = distance_matrix(val_band_feature, val_band_feature)
    cond_x = distance_matrix(cond_band_feature, cond_band_feature)
    return val_x, cond_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot several BSs on 1')
    parser.add_argument('-i', '--input', nargs='*', type=str)
    parser.add_argument('-o', '--output', help='Write image to a file')

    msg = ('Default: "-3.0 3.0" ' '(in eV relative to Fermi level).',)
    parser.add_argument('-r', '--range', nargs=2, default=['-3', '3'], metavar=('emin', 'emax'), help=msg)
    parser.add_argument('-a', '--alignment', type=str, default=None)
    parser.add_argument('-N', '--nvalence', nargs=2, type=int, default=None)
    parser.add_argument('-f', '--font', type=int, default=FONT_SIZE)
    parser.add_argument('-s', '--step', type=float, default=Y_AXIS_STEP)
    parser.add_argument('-e', '--eref', nargs=1, type=float, default=None)

    args = parser.parse_args()

    plt.figure(figsize=(12, 8))

    # set of band structures
    bss = []
    labels = []
    for input_file in args.input:
        bs_path = Path(input_file)
        assert bs_path.exists()
        bss.append(read_band_structure(bs_path))
        labels.append(f'{bs_path.parent.stem}/{bs_path.stem}')

    all_special_points = set()
    for bs in bss:
        all_special_points.update(bs.path.special_points.keys())

    for bs in bss:
        msg = "The special points are not the same for selected BSs"
        assert set(list(bs.path.special_points.keys())) == all_special_points, msg

    color = iter(cm.rainbow(np.linspace(0, 1, len(bss))))

    if args.alignment is not None:
        msg = "Number of valence electrons should be provided for alignment"
        assert args.nvalence is not None, msg

    if args.nvalence is not None:
        msg = "Number of valence electrons should be the same as the number of band structures"
        assert len(args.nvalence) == len(args.input), msg

    for i, (label, bs) in enumerate(zip(labels, bss)):
        if args.alignment is not None:
            N_val_e = args.nvalence[i]
            bs = align_band_structure(bs, N_val_e, fermi_label=args.alignment)

        bs = bs.subtract_reference()

        emin, emax = (float(e) for e in args.range)
        fig = plt.gcf()
        # fig.canvas.set_window_title(args.calculation)
        ax = fig.gca()

        c = next(color)
        bs.plot(ax=ax, color=c, label=label, linewidth=3.0, emin=emin + bs.reference, emax=emax + bs.reference)

    ax.legend(
        loc='best',
        # bbox_to_anchor=(0.5, 1.05),
        fancybox=True,
        shadow=True,
        ncol=5,
        prop={'size': args.font-5}
    )
    ax.set_yticks(np.arange(emin, emax, args.step))
    ax.yaxis.label.set_size(args.font)
    plt.xticks(fontsize=args.font)
    plt.yticks(fontsize=args.font)

    if args.output is None:
        plt.show()
    else:
        output = Path(args.output)
        if output.is_dir():
            output = output / f'c_{Path(args.input[0]).stem}.png'
        plt.savefig(output)
