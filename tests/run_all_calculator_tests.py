#!/usr/bin/env python3
"""
Comprehensive test runner for all calculator tests.

This script runs tests for MACE, QE, DFTB, and VASP calculators.
"""

import subprocess
import sys


def run_tests():
    """Run all calculator tests."""
    print("Running All Calculator Unit Tests")
    print("=" * 60)
    
    # Test categories to run
    test_categories = [
        ("MACE Argument Parsing", "tests/test_mace_arguments.py::TestAddMaceArguments"),
        ("MACE Core Functionality", "tests/test_mace_simple.py"),
        ("QE Argument Parsing", "tests/test_qe_calculator.py::TestQEArgumentParsing"),
        ("QE Validation", "tests/test_qe_calculator.py::TestQEArgumentValidation"),
        ("DFTB Argument Parsing", "tests/test_dftb_calculator.py::TestDFTBArgumentParsing"),
        ("DFTB Validation", "tests/test_dftb_calculator.py::TestDFTBArgumentValidation"),
        ("VASP Argument Parsing", "tests/test_vasp_calculator.py::TestVASPArgumentParsing"),
        ("VASP Validation", "tests/test_vasp_calculator.py::TestVASPArgumentValidation"),
    ]
    
    total_passed = 0
    total_failed = 0
    failed_categories = []
    
    for category_name, test_path in test_categories:
        print(f"\n{category_name}:")
        print("-" * len(category_name))
        
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "-q"
            ], capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                # Count passed tests from output
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'passed' in line and ('warning' in line or line.strip().endswith('passed')):
                        # Extract number of passed tests
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'passed' or part == 'passed,':
                                try:
                                    passed_count = int(parts[i-1])
                                    total_passed += passed_count
                                    print(f"✅ {passed_count} tests passed")
                                    break
                                except (ValueError, IndexError):
                                    print(f"✅ Tests passed")
                                    total_passed += 1
                                    break
                        break
                else:
                    # Fallback if we can't parse the count
                    print(f"✅ Tests passed")
                    total_passed += 1
            else:
                print(f"❌ Some tests failed")
                if result.stdout:
                    print("STDOUT:", result.stdout[-500:])  # Last 500 chars
                if result.stderr:
                    print("STDERR:", result.stderr[-500:])  # Last 500 chars
                total_failed += 1
                failed_categories.append(category_name)
                
        except subprocess.TimeoutExpired:
            print(f"❌ Tests timed out")
            total_failed += 1
            failed_categories.append(category_name)
        except Exception as e:
            print(f"❌ Error running tests: {e}")
            total_failed += 1
            failed_categories.append(category_name)
    
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"✅ Total tests passed: {total_passed}")
    if total_failed > 0:
        print(f"❌ Failed categories: {total_failed}")
        print(f"   Failed: {', '.join(failed_categories)}")
    else:
        print("🎉 All test categories passed!")
    
    return total_failed == 0


def run_quick_test():
    """Run a quick subset of tests to verify basic functionality."""
    print("Running Quick Calculator Tests")
    print("=" * 40)
    
    quick_tests = [
        ("MACE Core", "tests/test_mace_simple.py::TestMaceWrapperFunctions::test_mace_scf_wrapper_validation"),
        ("QE Args", "tests/test_qe_calculator.py::TestQEArgumentParsing::test_add_qe_arguments_scf"),
        ("DFTB Args", "tests/test_dftb_calculator.py::TestDFTBArgumentParsing::test_add_dftb_arguments_scf"),
        ("VASP Args", "tests/test_vasp_calculator.py::TestVASPArgumentParsing::test_add_vasp_arguments_scf"),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_path in quick_tests:
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", test_path, "-v", "-q"
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print(f"✅ {name}")
                passed += 1
            else:
                print(f"❌ {name}")
                failed += 1
        except Exception as e:
            print(f"❌ {name} - Error: {e}")
            failed += 1
    
    print(f"\nQuick Test Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        success = run_quick_test()
    else:
        success = run_tests()
    
    sys.exit(0 if success else 1)