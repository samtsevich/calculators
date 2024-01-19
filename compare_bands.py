from ase.cli import band_structure
from ase.io.jsonio import read_json
from ase.spectrum.band_structure import BandStructure
from pathlib import Path

import argparse
import matplotlib.pyplot as plt


def read_band_structure(filename):
    bs = read_json(filename)
    if not isinstance(bs, BandStructure):
        raise TypeError(f'Expected band structure, but file contains: {bs}')
    return bs


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

    if args.output is None:
        plt.show()
    else:
        output = Path(args.output)
        if output.is_dir:
            output = output/f'c_{Path(args.input[0]).stem}'
        plt.savefig(output)
