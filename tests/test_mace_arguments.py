#!/usr/bin/env python3
"""
Unit tests for MACE argument parsing and validation functionality.

Tests the add_mace_arguments() and get_args() functions from common.mace module.
"""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
from ase import Atoms

from common.mace import add_mace_arguments, get_args


class TestAddMaceArguments:
    """Test cases for add_mace_arguments() function."""
    
    def test_add_mace_arguments_scf(self):
        """Test add_mace_arguments() with SCF calculation type."""
        parser = argparse.ArgumentParser()
        result_parser = add_mace_arguments(parser, 'scf')
        
        # Check that parser is returned
        assert result_parser is parser
        
        # Check description is set correctly
        assert 'ASE single point calculation with MACE' in parser.description
        
        # Parse empty args to check required arguments
        with pytest.raises(SystemExit):  # argparse exits on missing required args
            parser.parse_args([])
    
    def test_add_mace_arguments_opt(self):
        """Test add_mace_arguments() with optimization calculation type."""
        parser = argparse.ArgumentParser()
        add_mace_arguments(parser, 'opt')
        
        # Check description is set correctly
        assert 'ASE optimization with MACE' in parser.description
        
        # Test that optimization-specific arguments are added
        args = parser.parse_args([
            '-i', 'test.vasp', 
            '--model', 'test.model',
            '--fmax', '0.01',
            '--nsteps', '100',
            '--full'
        ])
        
        assert args.input == 'test.vasp'
        assert args.model == 'test.model'
        assert args.fmax == 0.01
        assert args.nsteps == 100
        assert args.full_opt is True
    
    def test_add_mace_arguments_eos(self):
        """Test add_mace_arguments() with EOS calculation type."""
        parser = argparse.ArgumentParser()
        add_mace_arguments(parser, 'eos')
        
        # Check description is set correctly
        assert 'ASE equation of state calculation with MACE' in parser.description
        
        # Test that EOS-specific arguments are added (same as opt)
        args = parser.parse_args([
            '-i', 'test.vasp', 
            '--model', 'test.model',
            '--fmax', '0.05',
            '--nsteps', '200'
        ])
        
        assert args.fmax == 0.05
        assert args.nsteps == 200
        assert args.full_opt is False  # Default value
    
    def test_add_mace_arguments_invalid_calc_type(self):
        """Test add_mace_arguments() with invalid calculation type."""
        parser = argparse.ArgumentParser()
        
        with pytest.raises(ValueError, match='Unknown calculation type: invalid'):
            add_mace_arguments(parser, 'invalid')
    
    def test_add_mace_arguments_default_values(self):
        """Test that default values are set correctly."""
        parser = argparse.ArgumentParser()
        add_mace_arguments(parser, 'scf')
        
        args = parser.parse_args(['-i', 'test.vasp', '--model', 'test.model'])
        
        assert args.device == 'cpu'
        assert args.default_dtype == 'float64'
        assert args.compile_mode is None
        assert args.outdir is None


class TestGetArgs:
    """Test cases for get_args() function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create a mock structure file in proper VASP format
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
        
        # Create a mock model file
        self.model_file = self.temp_path / 'test_model.model'
        self.model_file.write_text("Mock model content")
        
        # Create test structure
        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('common.mace.validate_mace_dependencies')
    @patch('common.mace.validate_mace_model_file')
    @patch('common.mace.check_device_availability')
    @patch('ase.io.read')
    def test_get_args_valid_input_scf(self, mock_read, mock_device, mock_model, mock_deps):
        """Test get_args() with valid input for SCF calculation."""
        # Setup mocks
        mock_read.return_value = [self.test_atoms]
        mock_device.return_value = 'cpu'
        
        args = {
            'input': str(self.structure_file),
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None
        }
        
        result = get_args(args, 'scf')
        
        # Check that dependencies were validated
        mock_deps.assert_called_once()
        mock_model.assert_called_once()
        mock_device.assert_called_once_with('cpu')
        
        # Check processed arguments
        assert result['name'] == 'test_structure'
        assert len(result['structures']) == 1
        assert result['model_path'] == self.model_file.resolve()
        assert result['device'] == 'cpu'
        assert result['default_dtype'] == 'float64'
        assert result['outdir'].name == 'scf_test_structure'
    
    @patch('common.mace.validate_mace_dependencies')
    @patch('common.mace.validate_mace_model_file')
    @patch('common.mace.check_device_availability')
    @patch('ase.io.read')
    def test_get_args_valid_input_opt(self, mock_read, mock_device, mock_model, mock_deps):
        """Test get_args() with valid input for optimization calculation."""
        mock_read.return_value = [self.test_atoms]
        mock_device.return_value = 'cpu'
        
        args = {
            'input': str(self.structure_file),
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None,
            'fmax': 0.01,
            'nsteps': 100,
            'full_opt': False
        }
        
        result = get_args(args, 'opt')
        
        # Check optimization-specific parameters
        assert result['fmax'] == 0.01
        assert result['nsteps'] == 100
        assert result['full_opt'] is False
    
    def test_get_args_missing_input_file(self):
        """Test get_args() with missing input file."""
        args = {
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None
        }
        
        with pytest.raises(ValueError, match='Please specify input structure file'):
            get_args(args, 'scf')
    
    def test_get_args_nonexistent_input_file(self):
        """Test get_args() with non-existent input file."""
        args = {
            'input': str(self.temp_path / 'nonexistent.vasp'),
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None
        }
        
        with pytest.raises(AssertionError, match='does not exist'):
            get_args(args, 'scf')
    
    def test_get_args_nonexistent_model_file(self):
        """Test get_args() with non-existent model file."""
        args = {
            'input': str(self.structure_file),
            'model': str(self.temp_path / 'nonexistent.model'),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None
        }
        
        with pytest.raises(AssertionError, match='does not exist'):
            get_args(args, 'scf')
    
    @patch('common.mace.validate_mace_dependencies')
    @patch('common.mace.validate_mace_model_file')
    @patch('common.mace.check_device_availability')
    @patch('ase.io.read')
    def test_get_args_invalid_device(self, mock_read, mock_device, mock_model, mock_deps):
        """Test get_args() with invalid device."""
        mock_read.return_value = [self.test_atoms]
        
        args = {
            'input': str(self.structure_file),
            'model': str(self.model_file),
            'device': 'invalid_device',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None
        }
        
        with pytest.raises(AssertionError, match='Device must be one of'):
            get_args(args, 'scf')
    
    @patch('common.mace.validate_mace_dependencies')
    @patch('common.mace.validate_mace_model_file')
    @patch('common.mace.check_device_availability')
    @patch('ase.io.read')
    def test_get_args_invalid_dtype(self, mock_read, mock_device, mock_model, mock_deps):
        """Test get_args() with invalid data type."""
        mock_read.return_value = [self.test_atoms]
        mock_device.return_value = 'cpu'
        
        args = {
            'input': str(self.structure_file),
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'invalid_dtype',
            'compile_mode': None,
            'outdir': None
        }
        
        with pytest.raises(AssertionError, match='Data type must be one of'):
            get_args(args, 'scf')
    
    def test_get_args_invalid_fmax(self):
        """Test get_args() with invalid fmax parameter."""
        with patch('common.mace.validate_mace_dependencies'), \
             patch('common.mace.validate_mace_model_file'), \
             patch('common.mace.check_device_availability', return_value='cpu'), \
             patch('ase.io.read', return_value=[self.test_atoms]):
            
            args = {
                'input': str(self.structure_file),
                'model': str(self.model_file),
                'device': 'cpu',
                'default_dtype': 'float64',
                'compile_mode': None,
                'outdir': None,
                'fmax': -0.01,  # Invalid negative value
                'nsteps': 100,
                'full_opt': False
            }
            
            with pytest.raises(ValueError, match='Force convergence criterion.*must be positive'):
                get_args(args, 'opt')
    
    def test_get_args_invalid_nsteps(self):
        """Test get_args() with invalid nsteps parameter."""
        with patch('common.mace.validate_mace_dependencies'), \
             patch('common.mace.validate_mace_model_file'), \
             patch('common.mace.check_device_availability', return_value='cpu'), \
             patch('ase.io.read', return_value=[self.test_atoms]):
            
            args = {
                'input': str(self.structure_file),
                'model': str(self.model_file),
                'device': 'cpu',
                'default_dtype': 'float64',
                'compile_mode': None,
                'outdir': None,
                'fmax': 0.01,
                'nsteps': 0,  # Invalid zero value
                'full_opt': False
            }
            
            with pytest.raises(ValueError, match='Number of steps.*must be positive'):
                get_args(args, 'opt')
    
    def test_get_args_invalid_nsteps_type(self):
        """Test get_args() with invalid nsteps type."""
        with patch('common.mace.validate_mace_dependencies'), \
             patch('common.mace.validate_mace_model_file'), \
             patch('common.mace.check_device_availability', return_value='cpu'), \
             patch('ase.io.read', return_value=[self.test_atoms]):
            
            args = {
                'input': str(self.structure_file),
                'model': str(self.model_file),
                'device': 'cpu',
                'default_dtype': 'float64',
                'compile_mode': None,
                'outdir': None,
                'fmax': 0.01,
                'nsteps': 'invalid',  # Invalid string value
                'full_opt': False
            }
            
            with pytest.raises(ValueError, match='Number of steps.*must be an integer'):
                get_args(args, 'opt')
    
    @patch('common.mace.validate_mace_dependencies')
    @patch('common.mace.validate_mace_model_file')
    @patch('common.mace.check_device_availability')
    @patch('ase.io.read')
    def test_get_args_with_structures_list(self, mock_read, mock_device, mock_model, mock_deps):
        """Test get_args() with pre-loaded structures list."""
        mock_device.return_value = 'cpu'
        
        args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None
        }
        
        result = get_args(args, 'scf')
        
        # Should not call ase.io.read since structures are provided
        mock_read.assert_not_called()
        
        assert result['name'] == 'test_structure'
        assert len(result['structures']) == 1
        assert result['structures'][0] is self.test_atoms
    
    @patch('common.mace.validate_mace_dependencies')
    @patch('common.mace.validate_mace_model_file')
    @patch('common.mace.check_device_availability')
    @patch('ase.io.read')
    def test_get_args_custom_outdir(self, mock_read, mock_device, mock_model, mock_deps):
        """Test get_args() with custom output directory."""
        mock_read.return_value = [self.test_atoms]
        mock_device.return_value = 'cpu'
        
        custom_outdir = self.temp_path / 'custom_output'
        
        args = {
            'input': str(self.structure_file),
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': str(custom_outdir)
        }
        
        result = get_args(args, 'scf')
        
        assert result['outdir'] == custom_outdir.resolve()
        assert custom_outdir.exists()  # Should be created
    
    @patch('common.mace.validate_mace_dependencies')
    def test_get_args_read_structure_error(self, mock_deps):
        """Test get_args() with structure file read error."""
        # Create an invalid structure file
        invalid_file = self.temp_path / 'invalid.vasp'
        invalid_file.write_text("Invalid content")
        
        args = {
            'input': str(invalid_file),
            'model': str(self.model_file),
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'outdir': None
        }
        
        with patch('ase.io.read', side_effect=Exception("Read error")):
            with pytest.raises(ValueError, match='Failed to read structure file'):
                get_args(args, 'scf')


class TestValidationFunctions:
    """Test cases for validation helper functions."""
    
    def test_validate_mace_dependencies_missing_torch(self):
        """Test validate_mace_dependencies() with missing PyTorch."""
        from common.mace import validate_mace_dependencies
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'torch'")):
            with pytest.raises(ImportError, match='PyTorch is required'):
                validate_mace_dependencies()
    
    def test_validate_mace_dependencies_missing_mace(self):
        """Test validate_mace_dependencies() with missing mace-torch."""
        from common.mace import validate_mace_dependencies
        
        # Mock torch as available but mace as missing
        def mock_import(name, *args, **kwargs):
            if name == 'torch':
                mock_torch = Mock()
                mock_torch.__version__ = '2.0.0'
                return mock_torch
            elif 'mace.calculators' in name:
                raise ImportError("No module named 'mace'")
            else:
                # Use the real import for other modules
                return __import__(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            with pytest.raises(ImportError, match='mace-torch package is required'):
                validate_mace_dependencies()
    
    def test_check_device_availability_cpu(self):
        """Test check_device_availability() with CPU device."""
        from common.mace import check_device_availability
        
        result = check_device_availability('cpu')
        assert result == 'cpu'
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_check_device_availability_cuda_unavailable(self, mock_cuda):
        """Test check_device_availability() with unavailable CUDA."""
        from common.mace import check_device_availability
        
        with pytest.raises(RuntimeError, match='CUDA device requested but CUDA is not available'):
            check_device_availability('cuda')
    
    @patch('torch.backends.mps.is_available', return_value=False)
    def test_check_device_availability_mps_unavailable(self, mock_mps):
        """Test check_device_availability() with unavailable MPS."""
        from common.mace import check_device_availability
        
        with pytest.raises(RuntimeError, match='MPS device requested but MPS is not available'):
            check_device_availability('mps')
    
    def test_check_device_availability_invalid_device(self):
        """Test check_device_availability() with invalid device."""
        from common.mace import check_device_availability
        
        with pytest.raises(ValueError, match='Unknown device'):
            check_device_availability('invalid')