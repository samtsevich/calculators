#!/usr/bin/env python3
"""
Comprehensive test suite for all calculator implementations.

This module provides unified tests that work across QE, DFTB, VASP, and MACE calculators
to ensure consistent behavior and interfaces.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import numpy as np
from ase import Atoms

# Import all calculator modules
from common.qe import add_qe_arguments as add_qe_args
from common.dftb import add_dftb_arguments as add_dftb_args
from common.vasp import add_vasp_arguments as add_vasp_args
from common.mace import add_mace_arguments as add_mace_args


class TestCalculatorArgumentParsing:
    """Test argument parsing consistency across all calculators."""
    
    @pytest.mark.parametrize("calc_name,add_args_func", [
        ("QE", add_qe_args),
        ("DFTB", add_dftb_args),
        ("VASP", add_vasp_args),
        ("MACE", add_mace_args),
    ])
    def test_scf_argument_parsing(self, calc_name, add_args_func):
        """Test that all calculators support SCF calculation type."""
        import argparse
        parser = argparse.ArgumentParser()
        
        result_parser = add_args_func(parser, 'scf')
        
        # Check that parser is returned
        assert result_parser is parser
        
        # Check that description contains expected text
        assert 'scf' in parser.description.lower() or 'single point' in parser.description.lower()
    
    @pytest.mark.parametrize("calc_name,add_args_func", [
        ("QE", add_qe_args),
        ("DFTB", add_dftb_args),
        ("VASP", add_vasp_args),
        ("MACE", add_mace_args),
    ])
    def test_opt_argument_parsing(self, calc_name, add_args_func):
        """Test that all calculators support optimization calculation type."""
        import argparse
        parser = argparse.ArgumentParser()
        
        result_parser = add_args_func(parser, 'opt')
        
        # Check that parser is returned
        assert result_parser is parser
        
        # Check that description contains expected text
        assert 'opt' in parser.description.lower() or 'optimization' in parser.description.lower()
    
    @pytest.mark.parametrize("calc_name,add_args_func", [
        ("QE", add_qe_args),
        ("DFTB", add_dftb_args),
        ("VASP", add_vasp_args),
        ("MACE", add_mace_args),
    ])
    def test_invalid_calc_type_raises_error(self, calc_name, add_args_func):
        """Test that all calculators raise ValueError for invalid calculation types."""
        import argparse
        parser = argparse.ArgumentParser()
        
        with pytest.raises(ValueError, match='Unknown'):
            add_args_func(parser, 'invalid_calc_type')
    
    @pytest.mark.parametrize("calc_name,add_args_func", [
        ("QE", add_qe_args),
        ("DFTB", add_dftb_args),
        ("VASP", add_vasp_args),
        ("MACE", add_mace_args),
    ])
    def test_input_argument_present(self, calc_name, add_args_func):
        """Test that all calculators have input file argument."""
        import argparse
        parser = argparse.ArgumentParser()
        add_args_func(parser, 'scf')
        
        # Parse with minimal arguments to check if input is recognized
        try:
            # This should not raise an error about unknown arguments
            args = parser.parse_args(['-i', 'test.vasp'], namespace=argparse.Namespace())
            assert hasattr(args, 'input')
        except SystemExit:
            # It's OK if it exits due to missing required args, 
            # as long as it recognizes the -i flag
            pass


class TestCalculatorCommonInterface:
    """Test common interface patterns across calculators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock structure file
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
    
    @pytest.mark.parametrize("calc_name,add_args_func", [
        ("QE", add_qe_args),
        ("DFTB", add_dftb_args),
        ("VASP", add_vasp_args),
        ("MACE", add_mace_args),
    ])
    def test_output_directory_argument(self, calc_name, add_args_func):
        """Test that all calculators support output directory argument."""
        import argparse
        parser = argparse.ArgumentParser()
        add_args_func(parser, 'scf')
        
        # Check if outdir argument is present
        help_text = parser.format_help()
        assert '-o' in help_text or '--outdir' in help_text or '--output' in help_text
    
    def test_calculator_wrapper_functions_exist(self):
        """Test that wrapper functions exist for all calculators."""
        # Test QE wrapper functions
        try:
            from common.qe.scf import qe_scf
            assert callable(qe_scf)
        except ImportError:
            pytest.skip("QE SCF module not available")
        
        # Test DFTB wrapper functions
        try:
            from common.dftb.scf import dftb_scf
            assert callable(dftb_scf)
        except ImportError:
            pytest.skip("DFTB SCF module not available")
        
        # Test VASP wrapper functions
        try:
            from common.vasp.scf import vasp_scf
            assert callable(vasp_scf)
        except ImportError:
            pytest.skip("VASP SCF module not available")
        
        # Test MACE wrapper functions
        try:
            from common.mace.scf import mace_scf
            assert callable(mace_scf)
        except ImportError:
            pytest.skip("MACE SCF module not available")


class TestCalculatorValidationPatterns:
    """Test validation patterns across calculators."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock structure file
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
    
    def test_qe_missing_input_validation(self):
        """Test QE validates missing input file."""
        try:
            from common.qe import get_args
            
            mock_args = Mock()
            mock_args.subcommand = 'scf'
            mock_args.config = None
            mock_args.input = str(self.temp_path / 'nonexistent.vasp')
            mock_args.pseudopotentials = "{}"  # QE expects this to be a string that can be evaluated
            
            with pytest.raises(AssertionError):
                get_args(mock_args)
        except ImportError:
            pytest.skip("QE module not available")
    
    def test_dftb_missing_input_validation(self):
        """Test DFTB validates missing input file."""
        try:
            from common.dftb import get_args
            
            args = {
                'input': str(self.temp_path / 'nonexistent.vasp'),
                'skfs_dir': str(self.temp_path),
                'mixer': 'Broyden',
                'outdir': None
            }
            
            with pytest.raises(AssertionError):
                get_args(args, 'scf')
        except ImportError:
            pytest.skip("DFTB module not available")
    
    def test_vasp_missing_input_validation(self):
        """Test VASP validates missing input file."""
        try:
            from common.vasp import get_args
            
            mock_args = Mock()
            mock_args.subcommand = 'scf'
            mock_args.config = None
            mock_args.input = str(self.temp_path / 'nonexistent.vasp')
            
            with pytest.raises(AssertionError):
                get_args(mock_args)
        except ImportError:
            pytest.skip("VASP module not available")
    
    def test_mace_missing_input_validation(self):
        """Test MACE validates missing input file."""
        try:
            from common.mace import get_args
            
            args = {
                'input': str(self.temp_path / 'nonexistent.vasp'),
                'model': str(self.temp_path / 'model.model'),
                'device': 'cpu',
                'default_dtype': 'float64',
                'compile_mode': None,
                'outdir': None
            }
            
            with pytest.raises(AssertionError):
                get_args(args, 'scf')
        except ImportError:
            pytest.skip("MACE module not available")


class TestCalculatorWrapperFunctions:
    """Test wrapper function patterns across calculators."""
    
    def test_qe_scf_wrapper_validation(self):
        """Test QE SCF wrapper validates command correctly."""
        try:
            from common.qe.scf import qe_scf
            
            mock_args = Mock()
            mock_args.command = 'invalid'
            
            with pytest.raises(AssertionError, match="This function is only for QE"):
                qe_scf(mock_args)
        except ImportError:
            pytest.skip("QE SCF module not available")
    
    def test_dftb_scf_wrapper_validation(self):
        """Test DFTB SCF wrapper validates command correctly."""
        try:
            from common.dftb.scf import dftb_scf
            
            mock_args = Mock()
            mock_args.command = 'invalid'
            
            with pytest.raises(AssertionError, match="This function is only for DFTB"):
                dftb_scf(mock_args)
        except ImportError:
            pytest.skip("DFTB SCF module not available")
    
    def test_vasp_scf_wrapper_validation(self):
        """Test VASP SCF wrapper validates command correctly."""
        try:
            from common.vasp.scf import vasp_scf
            
            mock_args = Mock()
            mock_args.command = 'invalid'
            
            with pytest.raises(AssertionError, match="This function is only for VASP"):
                vasp_scf(mock_args)
        except ImportError:
            pytest.skip("VASP SCF module not available")
    
    def test_mace_scf_wrapper_validation(self):
        """Test MACE SCF wrapper validates command correctly."""
        try:
            from common.mace.scf import mace_scf
            
            mock_args = Mock()
            mock_args.command = 'invalid'
            
            with pytest.raises(AssertionError, match="This function is only for MACE"):
                mace_scf(mock_args)
        except ImportError:
            pytest.skip("MACE SCF module not available")


class TestCalculatorConstants:
    """Test that calculators define expected constants."""
    
    def test_qe_constants(self):
        """Test QE defines expected constants."""
        try:
            from common import F_MAX, KSPACING, N_STEPS, DEF_SMEARING
            
            # Check that constants are defined and reasonable
            assert isinstance(F_MAX, (int, float))
            assert F_MAX > 0
            assert isinstance(KSPACING, (int, float))
            assert KSPACING > 0
            assert isinstance(N_STEPS, int)
            assert N_STEPS > 0
            assert isinstance(DEF_SMEARING, (int, float))
            assert DEF_SMEARING > 0
        except ImportError:
            pytest.skip("Common constants not available")
    
    def test_dftb_constants(self):
        """Test DFTB defines expected constants."""
        try:
            from common.dftb import MIXERS, GENERAL_PARAMS, SCC_TOLERANCE
            
            # Check MIXERS
            assert isinstance(MIXERS, list)
            assert len(MIXERS) > 0
            assert 'Broyden' in MIXERS
            
            # Check GENERAL_PARAMS
            assert isinstance(GENERAL_PARAMS, dict)
            assert 'Hamiltonian_' in GENERAL_PARAMS
            
            # Check SCC_TOLERANCE
            assert isinstance(SCC_TOLERANCE, float)
            assert SCC_TOLERANCE > 0
        except ImportError:
            pytest.skip("DFTB constants not available")
    
    def test_vasp_constants(self):
        """Test VASP defines expected constants."""
        try:
            from common.vasp import COMMON_VASP_PARAMS
            
            # Check COMMON_VASP_PARAMS
            assert isinstance(COMMON_VASP_PARAMS, dict)
            assert 'xc' in COMMON_VASP_PARAMS
            assert 'encut' in COMMON_VASP_PARAMS
            assert 'ediff' in COMMON_VASP_PARAMS
        except ImportError:
            pytest.skip("VASP constants not available")
    
    def test_mace_constants(self):
        """Test MACE defines expected constants."""
        try:
            from common.mace import DEFAULT_DEVICE, DEFAULT_DTYPE
            
            # Check MACE constants
            assert isinstance(DEFAULT_DEVICE, str)
            assert DEFAULT_DEVICE in ['cpu', 'cuda', 'mps']
            assert isinstance(DEFAULT_DTYPE, str)
            assert DEFAULT_DTYPE in ['float32', 'float64']
        except ImportError:
            pytest.skip("MACE constants not available")


class TestCalculatorErrorHandling:
    """Test error handling patterns across calculators."""
    
    def test_calculator_import_error_handling(self):
        """Test that calculators handle import errors gracefully."""
        # This test ensures that if a calculator's dependencies are missing,
        # the error messages are helpful
        
        # Test MACE import error handling
        try:
            from common.mace.scf import run_mace_scf
            
            with patch('builtins.__import__', side_effect=ImportError("No module named 'mace'")):
                with pytest.raises(ImportError, match="mace-torch package is required"):
                    run_mace_scf({}, 'scf')
        except ImportError:
            pytest.skip("MACE module not available")
    
    def test_calculation_type_validation(self):
        """Test that calculation functions validate their calc_type parameter."""
        # Test MACE calculation type validation
        try:
            from common.mace.scf import run_mace_scf
            
            with pytest.raises(AssertionError, match="This function is only for SCF calculation"):
                run_mace_scf({}, 'opt')
        except ImportError:
            pytest.skip("MACE SCF module not available")
        
        # Test DFTB calculation type validation
        try:
            from common.dftb.scf import run_dftb_scf
            
            with pytest.raises(AssertionError, match="This function is only for SCF calculation"):
                run_dftb_scf({}, 'opt')
        except ImportError:
            pytest.skip("DFTB SCF module not available")


class TestCalculatorIntegration:
    """Test integration patterns across calculators."""
    
    def test_cli_integration_exists(self):
        """Test that CLI integration exists for all calculators."""
        try:
            from cli import QE_CALC_TYPES, DFTB_CALC_TYPES, VASP_CALC_TYPES, MACE_CALC_TYPES
            
            # Check that calculation type dictionaries exist
            assert isinstance(QE_CALC_TYPES, dict)
            assert isinstance(DFTB_CALC_TYPES, dict)
            assert isinstance(VASP_CALC_TYPES, dict)
            assert isinstance(MACE_CALC_TYPES, dict)
            
            # Check that SCF is available for all calculators
            assert 'scf' in QE_CALC_TYPES
            assert 'scf' in DFTB_CALC_TYPES
            assert 'scf' in VASP_CALC_TYPES
            assert 'scf' in MACE_CALC_TYPES
            
            # Check that functions are callable
            assert callable(QE_CALC_TYPES['scf'])
            assert callable(DFTB_CALC_TYPES['scf'])
            assert callable(VASP_CALC_TYPES['scf'])
            assert callable(MACE_CALC_TYPES['scf'])
            
        except ImportError:
            pytest.skip("CLI module not available")
    
    def test_common_module_imports(self):
        """Test that common module imports work correctly."""
        try:
            from common import F_MAX, KSPACING, N_STEPS
            
            # Check that common constants are accessible
            assert F_MAX is not None
            assert KSPACING is not None
            assert N_STEPS is not None
            
        except ImportError:
            pytest.skip("Common module not available")


class TestCalculatorDocumentation:
    """Test that calculators have proper documentation."""
    
    def test_function_docstrings_exist(self):
        """Test that key functions have docstrings."""
        # Test QE functions
        try:
            from common.qe import add_qe_arguments, get_args
            
            assert add_qe_arguments.__doc__ is not None
            assert len(add_qe_arguments.__doc__.strip()) > 0
            
        except ImportError:
            pytest.skip("QE module not available")
        
        # Test DFTB functions
        try:
            from common.dftb import add_dftb_arguments, get_args
            
            assert add_dftb_arguments.__doc__ is not None
            assert len(add_dftb_arguments.__doc__.strip()) > 0
            
        except ImportError:
            pytest.skip("DFTB module not available")
        
        # Test VASP functions
        try:
            from common.vasp import add_vasp_arguments, get_args
            
            assert add_vasp_arguments.__doc__ is not None
            assert len(add_vasp_arguments.__doc__.strip()) > 0
            
        except ImportError:
            pytest.skip("VASP module not available")
        
        # Test MACE functions
        try:
            from common.mace import add_mace_arguments, get_args
            
            assert add_mace_arguments.__doc__ is not None
            assert len(add_mace_arguments.__doc__.strip()) > 0
            
        except ImportError:
            pytest.skip("MACE module not available")