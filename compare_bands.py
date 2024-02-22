from ase.io.espresso import get_valence_electrons
from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure
from pathlib import Path
from scipy.spatial import distance_matrix

import argparse
import matplotlib.pyplot as plt
import numpy as np


from common import get_N_val_electrons, fix_fermi_level


def read_band_structure(filename):
    bs = read_json(filename)
    if not isinstance(bs, BandStructure):
        raise TypeError(f'Expected band structure, but file contains: {bs}')
    return bs


def special_points_energies(band_structure: BandStructure) -> dict:
    """Returns the energies of the special points in the band structure.

    Args:
        band_structure (BandStructure): The band structure.
        nbands (int): number of band we want to extract from the band structure.

    Returns:
        dict: The energies of the special points, where the keys are the special points and values are the energies.
    """
    spe_kpts, kpts = band_structure.path.special_points, band_structure.path.kpts
    energies = band_structure.energies[0]   # energies of the first spin

    kpts_energies = {}
    for sp, vec in spe_kpts.items():
        for kpt, e in zip(kpts, energies):
            if np.allclose(vec, kpt, atol=1e-3):
                kpts_energies[sp] = e
                break
    assert len(kpts_energies) == len(
        spe_kpts), "The number of special points is not the same as the number of k-points"
    return kpts_energies


def decode_band(bandstructure: BandStructure, nbands: int):
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


def decode_band2(bandstructure: BandStructure, n_val_e: int, n_cond_bands: int = 2):
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
        cond_band_feature.extend([[i, en]
                                 for en in e[n_val_e+1:n_val_e+n_cond_bands]])
    val_band_feature = np.array(val_band_feature)
    cond_band_feature = np.array(cond_band_feature)
    val_x = distance_matrix(val_band_feature, val_band_feature)
    cond_x = distance_matrix(cond_band_feature, cond_band_feature)
    return val_x, cond_x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plot several BSs on 1')
    parser.add_argument('-i', '--input', nargs=2, type=str)
    parser.add_argument('-o', '--output', help='Write image to a file')
    parser.add_argument('-r', '--range', nargs=2, default=['-3', '3'],
                        metavar=('emin', 'emax'),
                        help='Default: "-3.0 3.0" '
                        '(in eV relative to Fermi level).')
    parser.add_argument('-e', '--eref', nargs=1, type=float, default=None)

    args = parser.parse_args()

    # set of band structures
    bss = []
    for input_file in args.input:
        assert Path(input_file).exists()
        bss.append(read_band_structure(input_file))

    bs1, bs2 = bss

    if args.eref is not None:
        bs2._reference = args.eref[0]

    bs1 = bs1.subtract_reference()
    bs2 = bs2.subtract_reference()
    assert bs1.path.special_points.keys() == bs1.path.special_points.keys(), "The special points are not the same for selected BSs"

    # print(f'Fermi level 1: {bs1.reference}')
    # print(f'Fermi level 2: {bs2.reference}')

    emin, emax = (float(e) for e in args.range)
    fig = plt.gcf()
    # fig.canvas.set_window_title(args.calculation)
    ax = fig.gca()
    # for bs in bss:
    #     bs.plot(ax=ax,
    #             emin=emin + bs.reference,
    #             emax=emax + bs.reference)

    bs1.plot(ax=ax,
             colors='r',
             label=args.input[0],
             emin=emin + bs1.reference,
             emax=emax + bs1.reference)

    bs2.plot(ax=ax,
             colors='b',
             linestyle='dashed',
             label=args.input[1],
             emin=emin + bs2.reference,
             emax=emax + bs2.reference)

    ax.legend(loc='upper center', bbox_to_anchor=(
        0.5, 1.05), fancybox=True, shadow=True, ncol=5)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)

    if args.output is None:
        plt.show()
    else:
        output = Path(args.output)
        if output.is_dir():
            output = output/f'c_{Path(args.input[0]).stem}.png'
        plt.savefig(output)
