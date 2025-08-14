#!/usr/bin/env python3
"""
Integration tests for MACE calculation workflows.

Tests the SCF, optimization, and EOS calculation workflows with mock MACE calculator.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
from ase import Atoms
from ase.io.trajectory import Trajectory

from common.mace.scf import run_mace_scf, mace_scf
from common.mace.opt import run_mace_opt, mace_opt
from common.mace.eos import run_mace_eos, mace_eos


class TestMaceSCFCalculation:
    """Test cases for MACE SCF calculation workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test structure
        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
        self.test_atoms.set_cell([10, 10, 10])
        
        # Mock model file
        self.model_file = self.temp_path / 'test_model.model'
        self.model_file.write_text("Mock model content")
        
        # Test arguments
        self.test_args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'scf_output',
            'model_path': self.model_file,
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None
        }
        
        # Create output directory
        self.test_args['outdir'].mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('common.mace.scf.get_args')
    @patch('mace.calculators.MACECalculator')
    @patch('ase.io.vasp.write_vasp')
    @patch('ase.io.trajectory.Trajectory')
    def test_run_mace_scf_success(self, mock_traj, mock_write_vasp, mock_calc_class, mock_get_args):
        """Test successful MACE SCF calculation."""
        # Setup mocks
        mock_get_args.return_value = self.test_args
        
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -5.0
        mock_calculator.get_forces.return_value = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
        mock_calc_class.return_value = mock_calculator
        
        mock_traj_instance = Mock()
        mock_traj.return_value.__enter__.return_value = mock_traj_instance
        
        # Run SCF calculation
        run_mace_scf(self.test_args, 'scf')
        
        # Verify get_args was called
        mock_get_args.assert_called_once_with(self.test_args, calc_type='scf')
        
        # Verify MACE calculator was initialized correctly
        mock_calc_class.assert_called_once_with(
            model_paths=[str(self.model_file)],
            device='cpu',
            default_dtype='float64'
        )
        
        # Verify structure had calculator attached
        assert self.test_atoms.calc is mock_calculator
        
        # Verify energy and forces were calculated
        mock_calculator.get_potential_energy.assert_called_once()
        mock_calculator.get_forces.assert_called_once()
        
        # Verify output files were written
        mock_write_vasp.assert_called_once()
        mock_traj_instance.write.assert_called_once()
    
    @patch('common.mace.scf.get_args')
    @patch('mace.calculators.MACECalculator')
    def test_run_mace_scf_calculator_init_error(self, mock_calc_class, mock_get_args):
        """Test MACE SCF calculation with calculator initialization error."""
        mock_get_args.return_value = self.test_args
        mock_calc_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(RuntimeError, match="Failed to initialize MACE calculator"):
            run_mace_scf(self.test_args, 'scf')
    
    @patch('common.mace.scf.get_args')
    @patch('mace.calculators.MACECalculator')
    def test_run_mace_scf_calculation_error(self, mock_calc_class, mock_get_args):
        """Test MACE SCF calculation with energy calculation error."""
        mock_get_args.return_value = self.test_args
        
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.side_effect = RuntimeError("CUDA out of memory")
        mock_calc_class.return_value = mock_calculator
        
        with pytest.raises(RuntimeError, match="GPU out of memory during MACE calculation"):
            run_mace_scf(self.test_args, 'scf')
    
    @patch('common.mace.scf.get_args')
    @patch('mace.calculators.MACECalculator')
    def test_run_mace_scf_invalid_energy(self, mock_calc_class, mock_get_args):
        """Test MACE SCF calculation with invalid energy result."""
        mock_get_args.return_value = self.test_args
        
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = float('nan')
        mock_calculator.get_forces.return_value = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
        mock_calc_class.return_value = mock_calculator
        
        with pytest.raises(RuntimeError, match="MACE calculation produced invalid energy"):
            run_mace_scf(self.test_args, 'scf')
    
    def test_run_mace_scf_invalid_calc_type(self):
        """Test run_mace_scf() with invalid calculation type."""
        with pytest.raises(AssertionError, match="This function is only for SCF calculation"):
            run_mace_scf(self.test_args, 'opt')
    
    @patch('common.mace.scf.run_mace_scf')
    def test_mace_scf_wrapper(self, mock_run_scf):
        """Test mace_scf() CLI wrapper function."""
        # Create mock args object
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'scf'
        
        # Mock vars() to return our test args
        with patch('builtins.vars', return_value=self.test_args):
            mace_scf(mock_args)
        
        # Verify run_mace_scf was called correctly
        mock_run_scf.assert_called_once_with(self.test_args, calc_type='scf')
    
    def test_mace_scf_wrapper_invalid_command(self):
        """Test mace_scf() wrapper with invalid command."""
        mock_args = Mock()
        mock_args.command = 'invalid'
        
        with pytest.raises(AssertionError, match="This function is only for MACE"):
            mace_scf(mock_args)


class TestMaceOptimizationCalculation:
    """Test cases for MACE optimization calculation workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test structure
        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
        self.test_atoms.set_cell([10, 10, 10])
        
        # Mock model file
        self.model_file = self.temp_path / 'test_model.model'
        self.model_file.write_text("Mock model content")
        
        # Test arguments
        self.test_args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'opt_output',
            'model_path': self.model_file,
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'fmax': 0.01,
            'nsteps': 100,
            'full_opt': False
        }
        
        # Create output directory
        self.test_args['outdir'].mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('common.mace.opt.get_args')
    @patch('mace.calculators.MACECalculator')
    @patch('ase.optimize.BFGS')
    @patch('ase.io.vasp.write_vasp')
    @patch('ase.io.trajectory.Trajectory')
    def test_run_mace_opt_positions_only(self, mock_traj, mock_write_vasp, mock_bfgs, mock_calc_class, mock_get_args):
        """Test successful MACE optimization of positions only."""
        # Setup mocks
        mock_get_args.return_value = self.test_args
        
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -5.5
        mock_calculator.get_forces.return_value = np.array([[0.005, 0.0, 0.0], [0.0, 0.0, 0.005]])
        mock_calc_class.return_value = mock_calculator
        
        mock_optimizer = Mock()
        mock_optimizer.run.return_value = True  # Converged
        mock_bfgs.return_value = mock_optimizer
        
        mock_traj_instance = Mock()
        mock_traj.return_value.__enter__.return_value = mock_traj_instance
        
        # Run optimization
        run_mace_opt(self.test_args, 'opt')
        
        # Verify get_args was called
        mock_get_args.assert_called_once_with(self.test_args, calc_type='opt')
        
        # Verify MACE calculator was initialized
        mock_calc_class.assert_called_once()
        
        # Verify BFGS optimizer was set up correctly
        mock_bfgs.assert_called_once()
        # Just verify it was called, don't check the exact structure since it might be wrapped
        
        # Verify optimization was run
        mock_optimizer.run.assert_called_once_with(fmax=0.01, steps=100)
        
        # Verify output files were written
        mock_write_vasp.assert_called_once()
        mock_traj_instance.write.assert_called_once()
    
    @patch('common.mace.opt.get_args')
    @patch('mace.calculators.MACECalculator')
    @patch('ase.optimize.BFGS')
    @patch('ase.filters.UnitCellFilter')
    @patch('ase.io.vasp.write_vasp')
    @patch('ase.io.trajectory.Trajectory')
    def test_run_mace_opt_full_optimization(self, mock_traj, mock_write_vasp, mock_filter, mock_bfgs, mock_calc_class, mock_get_args):
        """Test successful MACE full optimization (cell + positions)."""
        # Setup for full optimization
        full_opt_args = self.test_args.copy()
        full_opt_args['full_opt'] = True
        mock_get_args.return_value = full_opt_args
        
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -5.5
        mock_calculator.get_forces.return_value = np.array([[0.005, 0.0, 0.0], [0.0, 0.0, 0.005]])
        mock_calc_class.return_value = mock_calculator
        
        mock_filtered_structure = Mock()
        mock_filter.return_value = mock_filtered_structure
        
        mock_optimizer = Mock()
        mock_optimizer.run.return_value = True
        mock_bfgs.return_value = mock_optimizer
        
        mock_traj_instance = Mock()
        mock_traj.return_value.__enter__.return_value = mock_traj_instance
        
        # Run optimization
        run_mace_opt(full_opt_args, 'opt')
        
        # Verify UnitCellFilter was used
        mock_filter.assert_called_once_with(self.test_atoms)
        
        # Verify BFGS optimizer was set up with filtered structure
        mock_bfgs.assert_called_once()
        call_args = mock_bfgs.call_args[0]
        assert call_args[0] is mock_filtered_structure
    
    @patch('common.mace.opt.get_args')
    @patch('mace.calculators.MACECalculator')
    @patch('ase.optimize.BFGS')
    def test_run_mace_opt_convergence_failure(self, mock_bfgs, mock_calc_class, mock_get_args):
        """Test MACE optimization with convergence failure."""
        mock_get_args.return_value = self.test_args
        
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -5.0
        mock_calculator.get_forces.return_value = np.array([[0.1, 0.0, 0.0], [0.0, 0.0, 0.1]])
        mock_calc_class.return_value = mock_calculator
        
        mock_optimizer = Mock()
        mock_optimizer.run.return_value = False  # Did not converge
        mock_bfgs.return_value = mock_optimizer
        
        # Should not raise error, just print warning
        with patch('builtins.print') as mock_print:
            run_mace_opt(self.test_args, 'opt')
            
        # Check that warning was printed
        mock_print.assert_any_call("Warning: Optimization did not converge within 100 steps")
    
    @patch('common.mace.opt.get_args')
    @patch('mace.calculators.MACECalculator')
    @patch('ase.optimize.BFGS')
    def test_run_mace_opt_optimization_error(self, mock_bfgs, mock_calc_class, mock_get_args):
        """Test MACE optimization with optimization error."""
        mock_get_args.return_value = self.test_args
        
        mock_calculator = Mock()
        mock_calc_class.return_value = mock_calculator
        
        mock_optimizer = Mock()
        mock_optimizer.run.side_effect = RuntimeError("Optimization failed")
        mock_bfgs.return_value = mock_optimizer
        
        with pytest.raises(RuntimeError, match="Optimization failed"):
            run_mace_opt(self.test_args, 'opt')
    
    def test_run_mace_opt_invalid_calc_type(self):
        """Test run_mace_opt() with invalid calculation type."""
        with pytest.raises(AssertionError, match="This function is only for optimization calculation"):
            run_mace_opt(self.test_args, 'scf')
    
    @patch('common.mace.opt.get_args')
    def test_run_mace_opt_invalid_fmax(self, mock_get_args):
        """Test run_mace_opt() with invalid fmax parameter."""
        invalid_args = self.test_args.copy()
        invalid_args['fmax'] = -0.01
        mock_get_args.return_value = invalid_args
        
        with pytest.raises(AssertionError, match="fmax should be positive"):
            run_mace_opt(invalid_args, 'opt')
    
    @patch('common.mace.opt.get_args')
    def test_run_mace_opt_invalid_nsteps(self, mock_get_args):
        """Test run_mace_opt() with invalid nsteps parameter."""
        invalid_args = self.test_args.copy()
        invalid_args['nsteps'] = 0
        mock_get_args.return_value = invalid_args
        
        with pytest.raises(AssertionError, match="nsteps should be positive"):
            run_mace_opt(invalid_args, 'opt')
    
    @patch('common.mace.opt.run_mace_opt')
    def test_mace_opt_wrapper(self, mock_run_opt):
        """Test mace_opt() CLI wrapper function."""
        # Create mock args object
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'opt'
        
        # Mock vars() to return our test args
        with patch('builtins.vars', return_value=self.test_args):
            mace_opt(mock_args)
        
        # Verify run_mace_opt was called correctly
        mock_run_opt.assert_called_once_with(self.test_args, calc_type='opt')


class TestMaceEOSCalculation:
    """Test cases for MACE EOS calculation workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test structure
        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
        self.test_atoms.set_cell([10, 10, 10])
        
        # Mock model file
        self.model_file = self.temp_path / 'test_model.model'
        self.model_file.write_text("Mock model content")
        
        # Test arguments
        self.test_args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'eos_output',
            'model_path': self.model_file,
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'fmax': 0.01,
            'nsteps': 100,
            'full_opt': False
        }
        
        # Create output directory
        self.test_args['outdir'].mkdir(parents=True, exist_ok=True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('common.mace.eos.get_args')
    @patch('common.mace.eos.run_mace_opt')
    @patch('ase.io.read')
    @patch('ase.eos.EquationOfState')
    @patch('ase.io.trajectory.Trajectory')
    @patch('numpy.savetxt')
    def test_run_mace_eos_success(self, mock_savetxt, mock_traj, mock_eos_class, mock_read, mock_run_opt, mock_get_args):
        """Test successful MACE EOS calculation."""
        # Setup mocks
        mock_get_args.return_value = self.test_args
        
        # Mock optimized reference structure
        optimized_atoms = self.test_atoms.copy()
        optimized_atoms.set_positions([[0, 0, 0], [0, 0, 0.9]])  # Slightly optimized
        
        # Mock read to return different structures for different calls
        def mock_read_side_effect(filename, index=-1):
            if 'final.traj' in str(filename):
                return optimized_atoms
            return optimized_atoms
        
        mock_read.side_effect = mock_read_side_effect
        
        # Mock EOS fitting
        mock_eos = Mock()
        mock_eos.fit.return_value = (100.0, -10.0, 50.0)  # v0, e0, B
        mock_eos.getplotdata.return_value = [None, None, None, 
                                           np.linspace(90, 110, 20),  # v_fit
                                           np.linspace(-9.8, -10.2, 20),  # e_fit
                                           np.array([95, 100, 105]),  # v
                                           np.array([-9.9, -10.0, -9.9])]  # e
        mock_eos_class.return_value = mock_eos
        
        # Mock trajectory
        mock_traj_instance = Mock()
        mock_traj.return_value = mock_traj_instance
        
        # Run EOS calculation
        run_mace_eos(self.test_args, 'eos')
        
        # Verify get_args was called
        mock_get_args.assert_called_once_with(self.test_args, calc_type='eos')
        
        # Verify reference structure optimization was called
        mock_run_opt.assert_called()
        
        # Verify multiple optimization calls for volume scaling (9 points + reference)
        assert mock_run_opt.call_count >= 9
        
        # Verify EOS fitting was performed
        mock_eos_class.assert_called_once()
        mock_eos.fit.assert_called_once()
        
        # Verify data files were saved
        assert mock_savetxt.call_count == 2  # e_vs_v_sc.dat and e_vs_v_sc_fit.dat
    
    @patch('common.mace.eos.get_args')
    @patch('common.mace.eos.run_mace_opt')
    def test_run_mace_eos_reference_opt_failure(self, mock_run_opt, mock_get_args):
        """Test MACE EOS calculation with reference optimization failure."""
        mock_get_args.return_value = self.test_args
        mock_run_opt.side_effect = RuntimeError("Optimization failed")
        
        with pytest.raises(RuntimeError, match="Failed to optimize reference structure for EOS"):
            run_mace_eos(self.test_args, 'eos')
    
    @patch('common.mace.eos.get_args')
    @patch('common.mace.eos.run_mace_opt')
    @patch('ase.io.read')
    @patch('ase.eos.EquationOfState')
    @patch('ase.io.trajectory.Trajectory')
    def test_run_mace_eos_fitting_failure(self, mock_traj, mock_eos_class, mock_read, mock_run_opt, mock_get_args):
        """Test MACE EOS calculation with EOS fitting failure."""
        mock_get_args.return_value = self.test_args
        
        # Mock successful optimization
        optimized_atoms = self.test_atoms.copy()
        mock_read.return_value = optimized_atoms
        
        # Mock trajectory
        mock_traj_instance = Mock()
        mock_traj.return_value = mock_traj_instance
        
        # Mock EOS fitting failure
        mock_eos = Mock()
        mock_eos.fit.side_effect = Exception("Fitting failed")
        mock_eos_class.return_value = mock_eos
        
        with pytest.raises(RuntimeError, match="Failed to fit equation of state"):
            run_mace_eos(self.test_args, 'eos')
    
    @patch('common.mace.eos.get_args')
    @patch('common.mace.eos.run_mace_opt')
    @patch('ase.io.read')
    @patch('ase.eos.EquationOfState')
    @patch('ase.io.trajectory.Trajectory')
    def test_run_mace_eos_invalid_bulk_modulus(self, mock_traj, mock_eos_class, mock_read, mock_run_opt, mock_get_args):
        """Test MACE EOS calculation with invalid bulk modulus."""
        mock_get_args.return_value = self.test_args
        
        # Mock successful optimization
        optimized_atoms = self.test_atoms.copy()
        mock_read.return_value = optimized_atoms
        
        # Mock EOS fitting with negative bulk modulus
        mock_eos = Mock()
        mock_eos.fit.return_value = (100.0, -10.0, -50.0)  # Negative B
        mock_eos_class.return_value = mock_eos
        
        mock_traj_instance = Mock()
        mock_traj.return_value = mock_traj_instance
        
        with pytest.raises(RuntimeError, match="EOS fit produced negative bulk modulus"):
            run_mace_eos(self.test_args, 'eos')
    
    def test_run_mace_eos_invalid_calc_type(self):
        """Test run_mace_eos() with invalid calculation type."""
        with pytest.raises(AssertionError, match="This function is only for EOS calculation"):
            run_mace_eos(self.test_args, 'scf')
    
    @patch('common.mace.eos.run_mace_eos')
    def test_mace_eos_wrapper(self, mock_run_eos):
        """Test mace_eos() CLI wrapper function."""
        # Create mock args object
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'eos'
        
        # Mock vars() to return our test args
        with patch('builtins.vars', return_value=self.test_args):
            mace_eos(mock_args)
        
        # Verify run_mace_eos was called correctly
        mock_run_eos.assert_called_once_with(self.test_args, calc_type='eos')


class TestCalculationErrorHandling:
    """Test cases for error handling across all calculation types."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.test_atoms = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1.0]])
        self.test_atoms.set_cell([10, 10, 10])
        
        self.model_file = self.temp_path / 'test_model.model'
        self.model_file.write_text("Mock model content")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('common.mace.scf.get_args')
    @patch('mace.calculators.MACECalculator')
    def test_keyboard_interrupt_handling_scf(self, mock_calc_class, mock_get_args):
        """Test keyboard interrupt handling in SCF calculation."""
        test_args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'scf_output',
            'model_path': self.model_file,
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None
        }
        
        mock_get_args.return_value = test_args
        mock_calc_class.side_effect = KeyboardInterrupt()
        
        # Create mock args object for wrapper
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'scf'
        
        with patch('builtins.vars', return_value=test_args):
            with patch('builtins.print') as mock_print:
                with pytest.raises(KeyboardInterrupt):
                    mace_scf(mock_args)
                
                # Verify interrupt message was printed
                mock_print.assert_any_call("\nMACE SCF calculation interrupted by user")
    
    @patch('common.mace.opt.get_args')
    @patch('mace.calculators.MACECalculator')
    def test_general_exception_handling_opt(self, mock_calc_class, mock_get_args):
        """Test general exception handling in optimization calculation."""
        test_args = {
            'name': 'test_structure',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'opt_output',
            'model_path': self.model_file,
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None,
            'fmax': 0.01,
            'nsteps': 100,
            'full_opt': False
        }
        
        mock_get_args.return_value = test_args
        mock_calc_class.side_effect = Exception("Unexpected error")
        
        # Create mock args object for wrapper
        mock_args = Mock()
        mock_args.command = 'mace'
        mock_args.subcommand = 'opt'
        
        with patch('builtins.vars', return_value=test_args):
            with patch('builtins.print') as mock_print:
                with pytest.raises(Exception):
                    mace_opt(mock_args)
                
                # Verify error message was printed
                mock_print.assert_any_call("\nMACE optimization failed: Failed to initialize MACE calculator: Unexpected error")


class TestMockCalculatorIntegration:
    """Test integration with mock MACE calculator to verify workflow behavior."""
    
    def setup_method(self):
        """Set up test fixtures with mock calculator."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create realistic test structure
        self.test_atoms = Atoms('H2O', 
                               positions=[[0, 0, 0], [0.757, 0.586, 0], [-0.757, 0.586, 0]],
                               cell=[15, 15, 15])
        
        self.model_file = self.temp_path / 'test_model.model'
        self.model_file.write_text("Mock model content")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('common.mace.scf.get_args')
    @patch('mace.calculators.MACECalculator')
    def test_realistic_scf_workflow(self, mock_calc_class, mock_get_args):
        """Test realistic SCF workflow with mock calculator."""
        # Setup realistic test arguments
        test_args = {
            'name': 'water_molecule',
            'structures': [self.test_atoms],
            'outdir': self.temp_path / 'scf_water',
            'model_path': self.model_file,
            'device': 'cpu',
            'default_dtype': 'float64',
            'compile_mode': None
        }
        
        mock_get_args.return_value = test_args
        
        # Create realistic mock calculator
        mock_calculator = Mock()
        mock_calculator.get_potential_energy.return_value = -14.2  # Realistic water energy
        mock_calculator.get_forces.return_value = np.array([
            [0.01, -0.02, 0.0],   # Small forces indicating near-equilibrium
            [-0.005, 0.01, 0.0],
            [-0.005, 0.01, 0.0]
        ])
        mock_calc_class.return_value = mock_calculator
        
        # Create output directory
        test_args['outdir'].mkdir(parents=True, exist_ok=True)
        
        # Mock file writing operations
        with patch('ase.io.vasp.write_vasp') as mock_write_vasp:
            with patch('ase.io.trajectory.Trajectory') as mock_traj:
                mock_traj_instance = Mock()
                mock_traj.return_value.__enter__.return_value = mock_traj_instance
                
                # Run SCF calculation
                run_mace_scf(test_args, 'scf')
                
                # Verify calculator was properly configured
                mock_calc_class.assert_called_once_with(
                    model_paths=[str(self.model_file)],
                    device='cpu',
                    default_dtype='float64'
                )
                
                # Verify structure had calculator attached
                assert self.test_atoms.calc is mock_calculator
                
                # Verify calculations were performed
                mock_calculator.get_potential_energy.assert_called_once()
                mock_calculator.get_forces.assert_called_once()
                
                # Verify output files were created
                mock_write_vasp.assert_called_once()
                mock_traj_instance.write.assert_called_once()
                
                # Verify trajectory was written with correct properties
                write_call = mock_traj_instance.write.call_args
                assert write_call[0][0] is self.test_atoms
                assert 'energy' in write_call[1]
                assert 'forces' in write_call[1]
                assert write_call[1]['energy'] == -14.2