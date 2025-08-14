#!/usr/bin/env python3
"""
Unit tests for DFTB calculator functionality.

Tests the argument parsing, validation, and calculation workflows for DFTB.
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
from ase import Atoms

from common.dftb import add_dftb_arguments, get_args
from common.dftb.scf import dftb_scf, run_dftb_scf
from common.dftb.opt import dftb_opt
from common.dftb.band import dftb_band
from common.dftb.eos import dftb_eos
from common.dftb.neb import dftb_neb


class TestDFTBArgumentParsing:
    """Test cases for DFTB argument parsing functions."""

    def test_add_dftb_arguments_scf(self):
        """Test add_dftb_arguments() with SCF calculation type."""
        parser = argparse.ArgumentParser()
        result_parser = add_dftb_arguments(parser, 'scf')

        assert result_parser is parser
        assert 'ASE single point calculation with DFTB' in parser.description

        # Test parsing with minimal required arguments
        args = parser.parse_args([
            '-i', 'test.vasp',
            '-s', '/path/to/skf'
        ])

        assert args.input == 'test.vasp'
        assert args.skfs_dir == '/path/to/skf'

    def test_add_dftb_arguments_opt(self):
        """Test add_dftb_arguments() with optimization calculation type."""
        parser = argparse.ArgumentParser()
        add_dftb_arguments(parser, 'opt')

        assert 'ASE optimization with DFTB' in parser.description

        # Test optimization-specific arguments
        args = parser.parse_args([
            '-i', 'test.vasp',
            '-s', '/path/to/skf',
            '--fmax', '0.01',
            '--nsteps', '100',
            '--full'
        ])

        assert args.fmax == 0.01
        assert args.nsteps == 100
        assert args.full_opt is True

    def test_add_dftb_arguments_band(self):
        """Test add_dftb_arguments() with band calculation type."""
        parser = argparse.ArgumentParser()
        add_dftb_arguments(parser, 'band')

        assert 'ASE band structure calculation with DFTB' in parser.description

        # Test band-specific arguments
        args = parser.parse_args([
            '-i', 'test.vasp',
            '-s', '/path/to/skf',
            '--emin', '-15',
            '--emax', '20'
        ])

        assert args.emin == -15
        assert args.emax == 20

    def test_add_dftb_arguments_neb(self):
        """Test add_dftb_arguments() with NEB calculation type."""
        parser = argparse.ArgumentParser()
        add_dftb_arguments(parser, 'neb')

        assert 'ASE NEB calculation with DFTB' in parser.description

        # Test NEB-specific arguments (same as opt)
        args = parser.parse_args([
            '-i', 'test.vasp',
            '-s', '/path/to/skf',
            '--fmax', '0.05',
            '--nsteps', '200'
        ])

        assert args.fmax == 0.05
        assert args.nsteps == 200

    def test_add_dftb_arguments_invalid_calc_type(self):
        """Test add_dftb_arguments() with invalid calculation type."""
        parser = argparse.ArgumentParser()

        with pytest.raises(ValueError, match='Unknown type invalid'):
            add_dftb_arguments(parser, 'invalid')

    def test_add_dftb_arguments_default_values(self):
        """Test that default values are set correctly."""
        parser = argparse.ArgumentParser()
        add_dftb_arguments(parser, 'scf')

        args = parser.parse_args([
            '-i', 'test.vasp',
            '-s', '/path/to/skf'
        ])

        assert args.nproc == 1  # Default N_PROCESS
        assert args.kspacing == 0.04  # Default KSPACING
        assert args.mixer == 'Broyden'  # Default mixer
        assert args.d3 is False  # Default D3 dispersion
        assert args.pol_rep is False  # Default polynomial repulsion
        assert args.fermi_temp == 0.0001  # Default Fermi temperature
        assert args.max_l is None  # Default max angular momentum
        assert args.outdir is None


class TestDFTBArgumentValidation:
    """Test cases for DFTB argument validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        # Create mock structure file in proper VASP format
        self.structure_file = self.temp_path / 'test_structure.vasp'
        vasp_content = """H2 molecule
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
H
2
Direct
0.0 0.0 0.0
0.0 0.0 0.1
"""
        self.structure_file.write_text(vasp_content)

        # Create mock SKF directory
        self.skf_dir = self.temp_path / 'skf_files'
        self.skf_dir.mkdir()
        (self.skf_dir / 'H-H.skf').write_text("Mock SKF file")

        # Test structure
        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('ase.io.read')
    def test_get_args_valid_input(self, mock_read):
        """Test get_args() with valid input."""
        mock_read.return_value = [self.test_atoms]

        args = {
            'input': str(self.structure_file),
            'skfs_dir': str(self.skf_dir),
            'nproc': 1,
            'kspacing': 0.2,
            'mixer': 'Broyden',
            'd3': False,
            'pol_rep': False,
            'fermi_temp': 0.0001,
            'max_l': None,
            'outdir': None
        }

        result = get_args(args, 'scf')

        assert result['name'] == 'test_structure'
        assert len(result['structures']) == 1
        assert result['skfs_dir'] == self.skf_dir.resolve()
        assert result['mixer'] == 'Broyden'
        assert result['outdir'].name == 'scf_test_structure'
        assert 'dftb_params' in result

    def test_get_args_missing_input_file(self):
        """Test get_args() with missing input file."""
        args = {
            'input': str(self.temp_path / 'nonexistent.vasp'),
            'skfs_dir': str(self.skf_dir),
            'mixer': 'Broyden',
            'fermi_temp': 0.0001,
            'pol_rep': False,
            'd3': False,
            'max_l': None,
            'outdir': None
        }

        with pytest.raises(AssertionError, match='File .* does not exist'):
            get_args(args, 'scf')

    def test_get_args_missing_skf_directory(self):
        """Test get_args() with missing SKF directory."""
        args = {
            'input': str(self.structure_file),
            'skfs_dir': str(self.temp_path / 'nonexistent_skf'),
            'mixer': 'Broyden',
            'fermi_temp': 0.0001,
            'pol_rep': False,
            'd3': False,
            'max_l': None,
            'outdir': None
        }

        with patch('ase.io.read', return_value=[self.test_atoms]):
            with pytest.raises(AssertionError, match='Please, check carefully path to the directory with .skf files'):
                get_args(args, 'scf')

    def test_get_args_invalid_mixer(self):
        """Test get_args() with invalid mixer."""
        args = {
            'input': str(self.structure_file),
            'skfs_dir': str(self.skf_dir),
            'mixer': 'InvalidMixer',
            'fermi_temp': 0.0001,
            'pol_rep': False,
            'd3': False,
            'max_l': None,
            'outdir': None
        }

        with patch('ase.io.read', return_value=[self.test_atoms]):
            with pytest.raises(AssertionError, match='Unknown mixer InvalidMixer'):
                get_args(args, 'scf')

    def test_get_args_with_max_l_parameter(self):
        """Test get_args() with max angular momentum parameter."""
        args = {
            'input': str(self.structure_file),
            'skfs_dir': str(self.skf_dir),
            'mixer': 'Broyden',
            'max_l': '{"H": "s", "C": "p"}',
            'fermi_temp': 0.0001,
            'pol_rep': False,
            'd3': False,
            'outdir': None
        }

        with patch('ase.io.read', return_value=[self.test_atoms]):
            result = get_args(args, 'scf')

            # Check that max_l parameters were added to dftb_params
            assert 'Hamiltonian_MaxAngularMomentum_H' in result['dftb_params']
            assert result['dftb_params']['Hamiltonian_MaxAngularMomentum_H'] == 's'

    def test_get_args_invalid_max_l_parameter(self):
        """Test get_args() with invalid max angular momentum parameter."""
        args = {
            'input': str(self.structure_file),
            'skfs_dir': str(self.skf_dir),
            'mixer': 'Broyden',
            'max_l': '{"H": "invalid"}',
            'fermi_temp': 0.0001,
            'pol_rep': False,
            'd3': False,
            'outdir': None
        }

        with patch('ase.io.read', return_value=[self.test_atoms]):
            with pytest.raises(AssertionError, match='Unknown angular momentum invalid'):
                get_args(args, 'scf')

    def test_get_args_with_structures_list(self):
        """Test get_args() with pre-loaded structures list."""
        args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'skfs_dir': str(self.skf_dir),
            'mixer': 'Broyden',
            'fermi_temp': 0.0001,
            'pol_rep': False,
            'd3': False,
            'max_l': None,
            'outdir': None
        }

        result = get_args(args, 'scf')

        assert result['name'] == 'test_structure'
        assert len(result['structures']) == 1
        assert result['structures'][0] is self.test_atoms


class TestDFTBCalculationWorkflows:
    """Test cases for DFTB calculation workflows."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
        self.test_atoms.set_cell([10, 10, 10])

        self.test_args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'dftb_output',
            'skfs_dir': self.temp_path / 'skf',
            'kspacing': 0.2,
            'dftb_params': {
                'Hamiltonian_': 'DFTB',
                'Hamiltonian_SCC': 'Yes',
                'slako_dir': str(self.temp_path / 'skf') + '/'
            }
        }

        # Create output directory
        self.test_args['outdir'].mkdir(parents=True, exist_ok=True)
        self.test_args['skfs_dir'].mkdir(parents=True, exist_ok=True)

        # Create dummy SKF files for H-H interaction
        (self.test_args['skfs_dir'] / 'H-H.skf').write_text("Dummy SKF file content")

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('common.dftb.scf.get_args')
    @patch('ase.calculators.dftb.Dftb')
    @patch('ase.io.vasp.write_vasp')
    @patch('ase.io.trajectory.Trajectory')
    @patch('common.get_KPoints')
    @pytest.mark.skip(reason="Requires proper SKF file format - integration test")
    def test_run_dftb_scf_success(self, mock_get_kpoints, mock_traj, mock_write_vasp, mock_calc_class, mock_get_args):
        """Test successful DFTB SCF calculation."""
        mock_get_args.return_value = self.test_args
        mock_get_kpoints.return_value = [2, 2, 2]

        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -5.0
        mock_calculator.get_forces.return_value = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
        mock_calc_class.return_value = mock_calculator

        mock_traj_instance = Mock()
        mock_traj.return_value.__enter__.return_value = mock_traj_instance

        run_dftb_scf(self.test_args, 'scf')

        # Verify get_args was called
        mock_get_args.assert_called_once_with(self.test_args, calc_type='scf')

        # Verify DFTB calculator was initialized
        mock_calc_class.assert_called_once()

        # Verify structure had calculator attached
        assert self.test_atoms.calc is mock_calculator

        # Verify energy calculation
        mock_calculator.get_potential_energy.assert_called_once()

        # Verify output files were written
        mock_write_vasp.assert_called_once()
        mock_traj_instance.write.assert_called_once()

    def test_run_dftb_scf_invalid_calc_type(self):
        """Test run_dftb_scf() with invalid calculation type."""
        with pytest.raises(AssertionError, match="This function is only for SCF calculation"):
            run_dftb_scf(self.test_args, 'opt')

    @patch('common.dftb.scf.run_dftb_scf')
    def test_dftb_scf_wrapper(self, mock_run_scf):
        """Test dftb_scf() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'dftb'
        mock_args.subcommand = 'scf'

        with patch('builtins.vars', return_value=self.test_args):
            dftb_scf(mock_args)

        mock_run_scf.assert_called_once_with(self.test_args, calc_type='scf')

    def test_dftb_scf_wrapper_invalid_command(self):
        """Test dftb_scf() wrapper with invalid command."""
        mock_args = Mock()
        mock_args.command = 'invalid'

        with pytest.raises(AssertionError, match="This function is only for DFTB"):
            dftb_scf(mock_args)

    @patch('common.dftb.opt.run_dftb_opt')
    def test_dftb_opt_wrapper(self, mock_run_opt):
        """Test dftb_opt() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'dftb'

        dftb_opt(mock_args)

        mock_run_opt.assert_called_once()

    @patch('common.dftb.band.run_dftb_band')
    def test_dftb_band_wrapper(self, mock_run_band):
        """Test dftb_band() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'dftb'

        dftb_band(mock_args)

        mock_run_band.assert_called_once()

    @patch('common.dftb.eos.run_dftb_eos')
    def test_dftb_eos_wrapper(self, mock_run_eos):
        """Test dftb_eos() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'dftb'

        dftb_eos(mock_args)

        mock_run_eos.assert_called_once()

    @patch('common.dftb.neb.run_dftb_neb')
    def test_dftb_neb_wrapper(self, mock_run_neb):
        """Test dftb_neb() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'dftb'

        dftb_neb(mock_args)

        mock_run_neb.assert_called_once()


class TestDFTBParameterGeneration:
    """Test cases for DFTB parameter generation functions."""

    def test_get_calc_type_params_scf(self):
        """Test get_calc_type_params() for SCF calculation."""
        from common.dftb import get_calc_type_params

        params = get_calc_type_params('scf')

        assert 'Hamiltonian_MaxSCCIterations' in params
        assert params['Hamiltonian_ReadInitialCharges'] == 'Yes'
        assert 'Analysis_CalculateForces' in params

    def test_get_calc_type_params_opt(self):
        """Test get_calc_type_params() for optimization calculation."""
        from common.dftb import get_calc_type_params

        params = get_calc_type_params('opt')

        assert 'Hamiltonian_MaxSCCIterations' in params
        assert params['Hamiltonian_ReadInitialCharges'] == 'Yes'
        assert 'Analysis_CalculateForces' in params

    def test_get_calc_type_params_band(self):
        """Test get_calc_type_params() for band calculation."""
        from common.dftb import get_calc_type_params

        params = get_calc_type_params('band')

        assert params['Hamiltonian_MaxSCCIterations'] == 1
        assert params['Hamiltonian_ReadInitialCharges'] == 'Yes'
        assert params['Hamiltonian_SCCTolerance'] == 1e6

    def test_get_calc_type_params_invalid(self):
        """Test get_calc_type_params() with invalid calculation type."""
        from common.dftb import get_calc_type_params

        with pytest.raises(ValueError, match='type must be band or static or opt_geometry'):
            get_calc_type_params('invalid')

    def test_get_dispersion_params(self):
        """Test get_dispersion_params() function."""
        from common.dftb import get_dispersion_params

        params = get_dispersion_params()

        assert params['Hamiltonian_Dispersion_'] == 'DftD3'
        assert 'Hamiltonian_Dispersion_s6' in params
        assert 'Hamiltonian_Dispersion_Damping_' in params


class TestDFTBErrorHandling:
    """Test cases for DFTB error handling."""

    def test_dftb_wrapper_functions_validation(self):
        """Test that all DFTB wrapper functions validate command correctly."""
        mock_args = Mock()
        mock_args.command = 'invalid'

        wrapper_functions = [dftb_scf, dftb_opt, dftb_band, dftb_eos, dftb_neb]

        for wrapper_func in wrapper_functions:
            with pytest.raises(AssertionError, match="This function is only for DFTB"):
                wrapper_func(mock_args)

    def test_get_args_no_input_or_structures(self):
        """Test get_args() with neither input file nor structures provided."""
        args = {
            'skfs_dir': '/path/to/skf',
            'mixer': 'Broyden',
            'fermi_temp': 0.0001,
            'pol_rep': False,
            'd3': False,
            'max_l': None,
            'outdir': None
        }

        with pytest.raises(ValueError, match='Please, specify input file or trajectory'):
            get_args(args, 'scf')

    def test_mixer_validation(self):
        """Test that mixer validation works correctly."""
        from common.dftb import MIXERS

        # Test valid mixers
        assert 'Broyden' in MIXERS
        assert 'Anderson' in MIXERS

        # Test that the list contains expected mixers
        assert len(MIXERS) >= 2