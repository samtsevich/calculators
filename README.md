# Calculators

Calculators for scf, optimization and band structure by VASP, QE and DFTB.

The package contains a suite of tools for setting up and running electronic structure calculations using Quantum ESPRESSO or VASP.
It also includes scripts to run semiempirical DFTB calculations for a specific parameters set.
Also, this code includes tool to visually compare band structures and (p)DOS obtained by different methods.
This project is a good companion to `DSKO` code that generates system-specific parameters.

Each of presented calculators have `scf`, `opt`, `band`, `opt` and `eos` subkeys for specific purposes.

Overall, the code orchestrates workflows for structural, band, and equation-of-state studies, integrating input generation, execution, and result parsing.



# Examples

## DFTB

```bash
py path/to/cli.py dftb (scf | band | opt | eos | pdos) -i wz_ZnO.vasp -s path/to/dftb_params_folder
```

## QE or VASP

### 1. SCF

```bash
py path/to/cli.py qe (scf | band | opt | eos | pdos) -i input_geometry.vasp -c config.yaml
```
