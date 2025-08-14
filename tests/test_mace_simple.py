#!/usr/bin/env python3
"""
Simplified unit tests for MACE calculator functionality.

Tests core functionality with minimal mocking complexity.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
from ase import Atoms

from common.mace.scf import mace_scf
from common.mace.opt import mace_opt
from common.mace.eos import mace_eos


class TestMaceWrapperFunctions:
    """Test the CLI wrapper functions."""
    
    def test_mace_scf_wrapper_validation(self):
        """Test mace_scf wrapper validates command correctly."""
        mock_args = Mock()
        mock_args.command = 'invalid'
        
        with pytest.raises(AssertionError, match="This function is only for MACE"):
            mace_scf(mock_args)
    
    def test_mace_opt_wrapper_validation(self):
        """Test mace_opt wrapper validates command correctly."""
        mock_args = Mock()
        mock_args.command = 'invalid'
        
        with pytest.raises(AssertionError, match="This function is only for MACE"):
            mace_opt(mock_args)
    
    def test_mace_eos_wrapper_validation(self):
        """Test mace_eos wrapper validates command correctly."""
        mock_args = Mock()
        mock_args.command = 'invalid'
        
        with pytest.raises(AssertionError, match="This function is only for MACE"):
            mace_eos(mock_args)
    
    @patch('common.mace.scf.run_mace_scf')
    def test_mace_scf_wrapper_calls_run_function(self, mock_run_scf):
        """Test mace_scf wrapper calls run_mace_scf correctly."""
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'scf'
        
        test_args_dict = {'test': 'value'}
        
        with patch('builtins.vars', return_value=test_args_dict):
            mace_scf(mock_args)
        
        mock_run_scf.assert_called_once_with(test_args_dict, calc_type='scf')
    
    @patch('common.mace.opt.run_mace_opt')
    def test_mace_opt_wrapper_calls_run_function(self, mock_run_opt):
        """Test mace_opt wrapper calls run_mace_opt correctly."""
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'opt'
        
        test_args_dict = {'test': 'value'}
        
        with patch('builtins.vars', return_value=test_args_dict):
            mace_opt(mock_args)
        
        mock_run_opt.assert_called_once_with(test_args_dict, calc_type='opt')
    
    @patch('common.mace.eos.run_mace_eos')
    def test_mace_eos_wrapper_calls_run_function(self, mock_run_eos):
        """Test mace_eos wrapper calls run_mace_eos correctly."""
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'eos'
        
        test_args_dict = {'test': 'value'}
        
        with patch('builtins.vars', return_value=test_args_dict):
            mace_eos(mock_args)
        
        mock_run_eos.assert_called_once_with(test_args_dict, calc_type='eos')


class TestMaceCalculationValidation:
    """Test calculation type validation."""
    
    def test_scf_calculation_type_validation(self):
        """Test SCF function validates calculation type."""
        from common.mace.scf import run_mace_scf
        
        with pytest.raises(AssertionError, match="This function is only for SCF calculation"):
            run_mace_scf({}, 'opt')
    
    def test_opt_calculation_type_validation(self):
        """Test optimization function validates calculation type."""
        from common.mace.opt import run_mace_opt
        
        with pytest.raises(AssertionError, match="This function is only for optimization calculation"):
            run_mace_opt({}, 'scf')
    
    def test_eos_calculation_type_validation(self):
        """Test EOS function validates calculation type."""
        from common.mace.eos import run_mace_eos
        
        with pytest.raises(AssertionError, match="This function is only for EOS calculation"):
            run_mace_eos({}, 'scf')


class TestMaceImportHandling:
    """Test MACE import error handling."""
    
    @patch('common.mace.scf.get_args')
    def test_scf_mace_import_error(self, mock_get_args):
        """Test SCF handles MACE import error correctly."""
        from common.mace.scf import run_mace_scf
        
        mock_get_args.return_value = {'test': 'args'}
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'mace'")):
            with pytest.raises(ImportError, match="mace-torch package is required"):
                run_mace_scf({}, 'scf')
    
    @patch('common.mace.opt.get_args')
    def test_opt_mace_import_error(self, mock_get_args):
        """Test optimization handles MACE import error correctly."""
        from common.mace.opt import run_mace_opt
        
        mock_get_args.return_value = {'test': 'args'}
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'mace'")):
            with pytest.raises(ImportError, match="mace-torch package is required"):
                run_mace_opt({}, 'opt')
    
    @patch('common.mace.eos.get_args')
    def test_eos_mace_import_error(self, mock_get_args):
        """Test EOS handles MACE import error correctly."""
        from common.mace.eos import run_mace_eos
        
        mock_get_args.return_value = {'test': 'args'}
        
        with patch('builtins.__import__', side_effect=ImportError("No module named 'mace'")):
            with pytest.raises(ImportError, match="mace-torch package is required"):
                run_mace_eos({}, 'eos')


class TestMaceParameterValidation:
    """Test parameter validation in optimization functions."""
    
    @patch('common.mace.opt.get_args')
    def test_opt_fmax_validation(self, mock_get_args):
        """Test optimization validates fmax parameter."""
        # Mock get_args to return the invalid args directly without processing
        def mock_get_args_side_effect(args, calc_type):
            # Return args with invalid fmax to trigger validation
            return {
                'name': 'test',
                'structures': [Atoms('H', positions=[[0, 0, 0]])],
                'outdir': Path('/tmp/test'),
                'model_path': Path('/tmp/model.model'),
                'device': 'cpu',
                'default_dtype': 'float64',
                'compile_mode': None,
                'fmax': -0.01,  # Invalid
                'nsteps': 100,
                'full_opt': False
            }
        
        mock_get_args.side_effect = mock_get_args_side_effect
        
        # Mock the MACE calculator import
        with patch.dict('sys.modules', {'mace': Mock(), 'mace.calculators': Mock()}):
            from common.mace.opt import run_mace_opt
            
            with pytest.raises(AssertionError, match="fmax should be positive"):
                run_mace_opt({}, 'opt')
    
    @patch('common.mace.opt.get_args')
    def test_opt_nsteps_validation(self, mock_get_args):
        """Test optimization validates nsteps parameter."""
        # Mock get_args to return the invalid args directly without processing
        def mock_get_args_side_effect(args, calc_type):
            # Return args with invalid nsteps to trigger validation
            return {
                'name': 'test',
                'structures': [Atoms('H', positions=[[0, 0, 0]])],
                'outdir': Path('/tmp/test'),
                'model_path': Path('/tmp/model.model'),
                'device': 'cpu',
                'default_dtype': 'float64',
                'compile_mode': None,
                'fmax': 0.01,
                'nsteps': 0,  # Invalid
                'full_opt': False
            }
        
        mock_get_args.side_effect = mock_get_args_side_effect
        
        # Mock the MACE calculator import
        with patch.dict('sys.modules', {'mace': Mock(), 'mace.calculators': Mock()}):
            from common.mace.opt import run_mace_opt
            
            with pytest.raises(AssertionError, match="nsteps should be positive"):
                run_mace_opt({}, 'opt')


class TestMaceErrorHandling:
    """Test error handling in wrapper functions."""
    
    @patch('common.mace.scf.run_mace_scf')
    def test_scf_keyboard_interrupt_handling(self, mock_run_scf):
        """Test SCF wrapper handles keyboard interrupt."""
        mock_run_scf.side_effect = KeyboardInterrupt()
        
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'scf'
        
        with patch('builtins.vars', return_value={}):
            with patch('builtins.print') as mock_print:
                with pytest.raises(KeyboardInterrupt):
                    mace_scf(mock_args)
                
                mock_print.assert_any_call("\nMACE SCF calculation interrupted by user")
    
    @patch('common.mace.opt.run_mace_opt')
    def test_opt_general_exception_handling(self, mock_run_opt):
        """Test optimization wrapper handles general exceptions."""
        mock_run_opt.side_effect = RuntimeError("Test error")
        
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'opt'
        
        with patch('builtins.vars', return_value={}):
            with patch('builtins.print') as mock_print:
                with pytest.raises(RuntimeError):
                    mace_opt(mock_args)
                
                mock_print.assert_any_call("\nMACE optimization failed: Test error")
    
    @patch('common.mace.eos.run_mace_eos')
    def test_eos_general_exception_handling(self, mock_run_eos):
        """Test EOS wrapper handles general exceptions."""
        mock_run_eos.side_effect = RuntimeError("Test error")
        
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'eos'
        
        with patch('builtins.vars', return_value={}):
            with patch('builtins.print') as mock_print:
                with pytest.raises(RuntimeError):
                    mace_eos(mock_args)
                
                mock_print.assert_any_call("\nMACE EOS calculation failed: Test error")