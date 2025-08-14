#!/usr/bin/env python3
"""
Test runner script for all calculator tests.

This script runs tests for MACE, QE, DFTB, and VASP calculators.
"""

import subprocess
import sys


def run_tests():
    """Run all calculator tests."""
    print("Running Calculator Unit Tests")
    print("=" * 50)
    
    # Test categories to run
    test_categories = [
        ("MACE Argument Parsing", "tests/test_mace_arguments.py::TestAddMaceArguments"),
        ("MACE Core Functionality", "tests/test_mace_simple.py"),
        ("QE Calculator Tests", "tests/test_qe_calculator.py::TestQEArgumentParsing"),
        ("DFTB Calculator Tests", "tests/test_dftb_calculator.py::TestDFTBArgumentParsing"),
        ("VASP Calculator Tests", "tests/test_vasp_calculator.py::TestVASPArgumentParsing"),
        ("Cross-Calculator Tests", "tests/test_all_calculators.py::TestCalculatorArgumentParsing"),
    ]
    
    total_passed = 0
    total_failed = 0
    
    for category_name, test_path in test_categories:
        print(f"\n{category_name}:")
        print("-" * len(category_name))
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_path, "-v", "--tb=short"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # Count passed tests from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'passed' in line and 'warning' in line:
                        # Extract number of passed tests
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'passed,':
                                passed_count = int(parts[i-1])
                                total_passed += passed_count
                                print(f"✅ {passed_count} tests passed")
                                break
                        break
            else:
                print(f"❌ Some tests failed")
                print(result.stdout)
                total_failed += 1
                
        except subprocess.TimeoutExpired:
            print(f"❌ Tests timed out")
            total_failed += 1
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            total_failed += 1
    
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"✅ Total tests passed: {total_passed}")
    if total_failed > 0:
        print(f"❌ Test categories with failures: {total_failed}")
    else:
        print("🎉 All calculator test categories passed!")
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)