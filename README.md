# Calculators

Calculators for scf, optimization and band structure by DFTB and QE

# Examples

## DFTB

```bash
py ../../cli.py dftb band -i wz_ZnO.vasp -s params
```

## QE

### 1. SCF

```bash
py ../../cli.py qe scf -c scf_config.yaml
```

### 2. Band structure

```bash
py ../../cli.py qe band -c band_config.yaml
```

### 3. Optimization

```bash
py ../../cli.py qe opt -c opt_config.yaml
```
