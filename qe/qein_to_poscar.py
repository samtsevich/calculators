from ase.io.espresso import read_espresso_out, read_espresso_in
from ase.io.vasp import write_vasp

from pathlib import Path

import sys

init_file = Path(sys.argv[1])

assert init_file.exists()

with open(init_file) as fp:
    struct = read_espresso_in(fp)
    assert len(struct)
    write_vasp(f'{init_file}.vasp', struct, direct=True, vasp5=True)

print('Done')

