#!/usr/bin/env python3
"""
Unit tests for VASP calculator functionality.

Tests the argument parsing, validation, and calculation workflows for VASP.
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
from ase import Atoms

from common.vasp import add_vasp_arguments, get_args, get_basic_params
from common.vasp.scf import vasp_scf, run_scf_vasp
from common.vasp.band import vasp_band
from common.vasp.pdos import vasp_pdos


class TestVASPArgumentParsing:
    """Test cases for VASP argument parsing functions."""
    
    def test_add_vasp_arguments_scf(self):
        """Test add_vasp_arguments() with SCF calculation type."""
        parser = argparse.ArgumentParser()
        result_parser = add_vasp_arguments(parser, 'scf')
        
        assert result_parser is parser
        assert 'ASE single point calculation with VASP' in parser.description
        
        # Test parsing with minimal required arguments
        args = parser.parse_args([
            '-i', 'test.vasp'
        ])
        
        assert args.input == 'test.vasp'
    
    def test_add_vasp_arguments_opt(self):
        """Test add_vasp_arguments() with optimization calculation type."""
        parser = argparse.ArgumentParser()
        add_vasp_arguments(parser, 'opt')
        
        assert 'ASE optimization with VASP' in parser.description
        
        # Test optimization-specific arguments
        args = parser.parse_args([
            '-i', 'test.vasp',
            '--fmax', '0.01',
            '--nsteps', '100',
            '--full'
        ])
        
        assert args.fmax == 0.01
        assert args.nsteps == 100
        assert args.full_opt is True
    
    def test_add_vasp_arguments_band(self):
        """Test add_vasp_arguments() with band calculation type."""
        parser = argparse.ArgumentParser()
        add_vasp_arguments(parser, 'band')
        
        assert 'ASE band structure calculation with VASP' in parser.description
        
        # Test band-specific arguments
        args = parser.parse_args([
            '-i', 'test.vasp',
            '--train'
        ])
        
        assert args.is_training is True
    
    def test_add_vasp_arguments_pdos(self):
        """Test add_vasp_arguments() with PDOS calculation type."""
        parser = argparse.ArgumentParser()
        add_vasp_arguments(parser, 'pdos')
        
        assert '(P)DOS calculation with VASP' in parser.description
        
        # Test PDOS arguments (same as SCF)
        args = parser.parse_args([
            '-i', 'test.vasp',
            '--smearing', '0.1'
        ])
        
        assert args.smearing == 0.1
    
    def test_add_vasp_arguments_invalid_calc_type(self):
        """Test add_vasp_arguments() with invalid calculation type."""
        parser = argparse.ArgumentParser()
        
        with pytest.raises(ValueError, match='Unknown type invalid'):
            add_vasp_arguments(parser, 'invalid')
    
    def test_add_vasp_arguments_default_values(self):
        """Test that default values are set correctly."""
        parser = argparse.ArgumentParser()
        add_vasp_arguments(parser, 'scf')
        
        args = parser.parse_args([
            '-i', 'test.vasp'
        ])
        
        assert args.kspacing == 0.04  # Default KSPACING
        assert args.setup is None
        assert args.smearing == 0.2  # Default DEF_SMEARING
        assert args.spin is False  # Default spin polarization
        assert args.config is None
        assert args.outdir is None
    
    def test_add_vasp_arguments_with_setups(self):
        """Test add_vasp_arguments() with atomic setups."""
        parser = argparse.ArgumentParser()
        add_vasp_arguments(parser, 'scf')
        
        args = parser.parse_args([
            '-i', 'test.vasp',
            '-s', "{'H': 'H.5', 'O': 'O'}",
            '--spin'
        ])
        
        assert args.setup == "{'H': 'H.5', 'O': 'O'}"
        assert args.spin is True


class TestVASPArgumentValidation:
    """Test cases for VASP argument validation."""
    
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
        
        # Create mock args object
        mock_args = Mock()
        mock_args.subcommand = 'scf'
        mock_args.config = None
        mock_args.input = str(self.structure_file)
        mock_args.outdir = None
        mock_args.kspacing = 0.2
        mock_args.setup = None
        mock_args.smearing = 0.1
        mock_args.spin = False
        
        result = get_args(mock_args)
        
        assert result['name'] == 'test_structure'
        assert len(result['structures']) == 1
        assert result['input'] == self.structure_file
        assert result['kspacing'] == 0.2
        assert result['smearing'] == 0.1
        assert result['outdir'].name == 'scf_test_structure'
    
    def test_get_args_missing_input_file(self):
        """Test get_args() with missing input file."""
        mock_args = Mock()
        mock_args.subcommand = 'scf'
        mock_args.config = None
        mock_args.input = str(self.temp_path / 'nonexistent.vasp')
        mock_args.outdir = None
        mock_args.kspacing = 0.2
        mock_args.setup = None
        mock_args.smearing = 0.1
        mock_args.spin = False
        
        with pytest.raises(AssertionError, match='Seems like path to the input file is wrong'):
            get_args(mock_args)
    
    def test_get_args_invalid_kspacing(self):
        """Test get_args() with invalid kspacing."""
        mock_args = Mock()
        mock_args.subcommand = 'scf'
        mock_args.config = None
        mock_args.input = str(self.structure_file)
        mock_args.outdir = None
        mock_args.kspacing = -0.1  # Invalid
        mock_args.setup = None
        mock_args.smearing = 0.1
        mock_args.spin = False
        
        with patch('ase.io.read', return_value=[self.test_atoms]):
            with pytest.raises(AssertionError, match='Seems like your value for KSPACING is not >0'):
                get_args(mock_args)
    
    def test_get_args_invalid_smearing(self):
        """Test get_args() with invalid smearing."""
        mock_args = Mock()
        mock_args.subcommand = 'scf'
        mock_args.config = None
        mock_args.input = str(self.structure_file)
        mock_args.outdir = None
        mock_args.kspacing = 0.2
        mock_args.setup = None
        mock_args.smearing = -0.1  # Invalid
        mock_args.spin = False
        
        with patch('ase.io.read', return_value=[self.test_atoms]):
            with pytest.raises(AssertionError, match='Seems like your value for smearing is not >0'):
                get_args(mock_args)
    
    def test_get_args_with_setups(self):
        """Test get_args() with atomic setups."""
        mock_args = Mock()
        mock_args.subcommand = 'scf'
        mock_args.config = None
        mock_args.input = str(self.structure_file)
        mock_args.outdir = None
        mock_args.kspacing = 0.2
        mock_args.setup = {'H': 'H.5'}
        mock_args.smearing = 0.1
        mock_args.spin = False
        
        with patch('ase.io.read', return_value=[self.test_atoms]):
            result = get_args(mock_args)
            
            assert result['setup'] == {'H': 'H.5'}
    
    def test_get_args_missing_setup_for_element(self):
        """Test get_args() with missing setup for an element."""
        # Create structure with carbon atom
        carbon_atoms = Atoms('CH4', positions=[[0, 0, 0], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
        
        mock_args = Mock()
        mock_args.subcommand = 'scf'
        mock_args.config = None
        mock_args.input = str(self.structure_file)
        mock_args.outdir = None
        mock_args.kspacing = 0.2
        mock_args.setup = {'H': 'H.5'}  # Missing C setup
        mock_args.smearing = 0.1
        mock_args.spin = False
        
        with patch('ase.io.read', return_value=[carbon_atoms]):
            with pytest.raises(AssertionError, match='No setup for C'):
                get_args(mock_args)
    
    def test_get_args_with_config_file(self):
        """Test get_args() with config file."""
        # Create mock config file
        config_file = self.temp_path / 'config.yaml'
        config_content = """
input: test_structure.vasp
kspacing: 0.15
smearing: 0.05
"""
        config_file.write_text(config_content)
        
        mock_args = Mock()
        mock_args.subcommand = 'scf'
        mock_args.config = str(config_file)
        
        with patch('ase.io.read', return_value=[self.test_atoms]):
            with patch('yaml.load') as mock_yaml_load:
                mock_yaml_load.return_value = {
                    'input': str(self.structure_file),
                    'kspacing': 0.15,
                    'smearing': 0.05,
                    'setup': None,
                    'spin': False,
                    'outdir': None
                }
                
                result = get_args(mock_args)
                
                assert result['kspacing'] == 0.15
                assert result['smearing'] == 0.05


class TestVASPParameterGeneration:
    """Test cases for VASP parameter generation functions."""
    
    def test_get_basic_params_default(self):
        """Test get_basic_params() with default parameters."""
        args = {
            'name': 'test_structure',
            'smearing': 0.1,
            'kspacing': 0.2,
            'spin': False,
            'setup': None
        }
        
        params = get_basic_params(args)
        
        assert params['system'] == 'test_structure'
        assert params['kspacing'] == 0.2
        assert params['sigma'] == 0.1
        assert params['xc'] == 'PBE'
        assert params['prec'] == 'Accurate'
        assert params['encut'] == 600
        assert 'ispin' not in params  # Should not be set for non-spin calculation
    
    def test_get_basic_params_with_spin(self):
        """Test get_basic_params() with spin polarization."""
        args = {
            'name': 'test_structure',
            'smearing': 0.1,
            'kspacing': 0.2,
            'spin': True,
            'setup': None
        }
        
        params = get_basic_params(args)
        
        assert params['ispin'] == 2  # Spin polarized
    
    def test_get_basic_params_with_setups(self):
        """Test get_basic_params() with atomic setups."""
        args = {
            'name': 'test_structure',
            'smearing': 0.1,
            'kspacing': 0.2,
            'spin': False,
            'setup': {'H': 'H.5', 'O': 'O'}
        }
        
        params = get_basic_params(args)
        
        assert params['setups'] == {'H': 'H.5', 'O': 'O'}
    
    def test_common_vasp_params(self):
        """Test that COMMON_VASP_PARAMS contains expected parameters."""
        from common.vasp import COMMON_VASP_PARAMS
        
        expected_params = ['xc', 'prec', 'encut', 'ediff', 'nsw', 'symprec', 
                          'ibrion', 'isif', 'nelm', 'ncore', 'ismear']
        
        for param in expected_params:
            assert param in COMMON_VASP_PARAMS


class TestVASPCalculationWorkflows:
    """Test cases for VASP calculation workflows."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
        self.test_atoms.set_cell([10, 10, 10])
        
        self.test_args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'vasp_output',
            'smearing': 0.1,
            'kspacing': 0.2,
            'spin': False,
            'setup': None
        }
        
        # Create output directory
        self.test_args['outdir'].mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('common.vasp.scf.get_args')
    @patch('common.vasp.scf.get_basic_params')
    @patch('ase.calculators.vasp.Vasp')
    @patch('ase.io.vasp.write_vasp')
    @patch('ase.io.trajectory.Trajectory')
    @patch('shutil.move')
    def test_run_scf_vasp_success(self, mock_move, mock_traj, mock_write_vasp, 
                                  mock_calc_class, mock_get_basic_params, mock_get_args):
        """Test successful VASP SCF calculation."""
        mock_get_args.return_value = self.test_args
        mock_get_basic_params.return_value = {
            'system': 'test_structure',
            'kspacing': 0.2,
            'sigma': 0.1,
            'directory': self.test_args['outdir']
        }
        
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -5.0
        mock_calculator.get_fermi_level.return_value = -2.5
        mock_calc_class.return_value = mock_calculator
        
        mock_traj_instance = Mock()
        mock_traj.return_value = mock_traj_instance
        
        run_scf_vasp(self.test_args)
        
        # Verify get_args was called
        mock_get_args.assert_called_once_with(self.test_args)
        
        # Verify get_basic_params was called
        mock_get_basic_params.assert_called_once_with(self.test_args)
        
        # Verify VASP calculator was initialized
        mock_calc_class.assert_called_once()
        
        # Verify structure had calculator attached
        assert self.test_atoms.calc is mock_calculator
        
        # Verify energy calculation
        mock_calculator.get_potential_energy.assert_called_once()
        mock_calculator.get_fermi_level.assert_called_once()
        
        # Verify output files were written
        mock_write_vasp.assert_called_once()
        mock_traj_instance.write.assert_called_once()
        mock_traj_instance.close.assert_called_once()
        
        # Verify VASP files were moved
        assert mock_move.call_count == 2  # INCAR and OUTCAR
    
    @patch('common.vasp.scf.run_scf_vasp')
    def test_vasp_scf_wrapper(self, mock_run_scf):
        """Test vasp_scf() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'vasp'
        
        vasp_scf(mock_args)
        
        mock_run_scf.assert_called_once()
    
    def test_vasp_scf_wrapper_invalid_command(self):
        """Test vasp_scf() wrapper with invalid command."""
        mock_args = Mock()
        mock_args.command = 'invalid'
        
        with pytest.raises(AssertionError, match="This function is only for VASP"):
            vasp_scf(mock_args)
    
    @patch('common.vasp.band.run_band_vasp')
    def test_vasp_band_wrapper(self, mock_run_band):
        """Test vasp_band() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'vasp'
        
        vasp_band(mock_args)
        
        mock_run_band.assert_called_once()
    
    @patch('common.vasp.pdos.run_pdos_vasp')
    def test_vasp_pdos_wrapper(self, mock_run_pdos):
        """Test vasp_pdos() CLI wrapper function."""
        mock_args = Mock()
        mock_args.command = 'vasp'
        
        vasp_pdos(mock_args)
        
        mock_run_pdos.assert_called_once()


class TestVASPErrorHandling:
    """Test cases for VASP error handling."""
    
    def test_vasp_wrapper_functions_validation(self):
        """Test that all VASP wrapper functions validate command correctly."""
        mock_args = Mock()
        mock_args.command = 'invalid'
        
        wrapper_functions = [vasp_scf, vasp_band, vasp_pdos]
        
        for wrapper_func in wrapper_functions:
            with pytest.raises(AssertionError, match="This function is only for VASP"):
                wrapper_func(mock_args)
    
    def test_get_total_n_val_e_missing_file(self):
        """Test get_total_N_val_e() with missing OUTCAR file."""
        from common.vasp import get_total_N_val_e
        
        nonexistent_file = Path('/nonexistent/OUTCAR')
        
        with pytest.raises(AssertionError, match='File .* does not exist'):
            get_total_N_val_e(nonexistent_file)
    
    def test_get_total_n_val_e_valid_file(self):
        """Test get_total_N_val_e() with valid OUTCAR file."""
        from common.vasp import get_total_N_val_e
        
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # Create mock OUTCAR file
            outcar_file = temp_path / 'OUTCAR'
            outcar_content = """
   NELECT =      8.0000    total number of electrons
   NKPTS =      1         number of k-points
"""
            outcar_file.write_text(outcar_content)
            
            result = get_total_N_val_e(outcar_file)
            assert result == 8
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_unknown_element_in_setups(self):
        """Test validation of unknown elements in setups."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        try:
            # Create structure file
            structure_file = temp_path / 'test.vasp'
            vasp_content = """Unknown element structure
1.0
10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0
Xx
1
Direct
0.0 0.0 0.0
"""
            structure_file.write_text(vasp_content)
            
            # Create structure with unknown element
            unknown_atoms = Atoms('Xx', positions=[[0, 0, 0]])
            
            mock_args = Mock()
            mock_args.subcommand = 'scf'
            mock_args.config = None
            mock_args.input = str(structure_file)
            mock_args.outdir = None
            mock_args.kspacing = 0.2
            mock_args.setup = {'Xx': 'Xx_setup'}  # Unknown element
            mock_args.smearing = 0.1
            mock_args.spin = False
            
            with patch('ase.io.read', return_value=[unknown_atoms]):
                with pytest.raises(AssertionError, match='Unknown element Xx'):
                    get_args(mock_args)
        
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)