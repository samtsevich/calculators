# Calculators

Calculators for scf, optimization and band structure by DFTB and QE

# Examples

## DFTB

```bash
py ../../dftb/band.py -i wz_ZnO.vasp -s params
```

## QE

### 1. SCF

```bash
py ../../qe/scf.py -c scf_config.yaml
```

### 2. Band structure

```bash
py ../../qe/band.py -c band_config.yaml
```

### 3. Optimization

```bash
py ../../qe/opt.py -c opt_config.yaml
```
