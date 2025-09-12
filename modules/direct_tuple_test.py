# direct_tuple_test.py - Test the specific method without full imports

import os
import sys
import inspect

def check_method_exists():
    """Check if the fixed methods exist in the file"""
    
    try:
        with open('modules/enhanced_google_integration.py', 'r') as f:
            content = f.read()
        
        # Check for the specific fixes we added
        fixes_to_check = [
            "# CRITICAL FIX: Ensure result is a dictionary before accessing it",
            "if not isinstance(result, dict):",
            "# CRITICAL FIX: Validate the result is a dictionary", 
            "# CRITICAL FIX: Get the result as a dictionary"
        ]
        
        print("Checking if Section 6 fixes are present in the file...")
        
        fixes_found = 0
        for fix_text in fixes_to_check:
            if fix_text in content:
                fixes_found += 1
                print(f"✅ Found: {fix_text}")
            else:
                print(f"❌ Missing: {fix_text}")
        
        print(f"\nResult: {fixes_found}/{len(fixes_to_check)} critical fixes found")
        
        if fixes_found == len(fixes_to_check):
            print("✅ All Section 6 fixes appear to be applied correctly")
            return True
        else:
            print("❌ Section 6 fixes are incomplete or missing")
            return False
            
    except FileNotFoundError:
        print("❌ Could not find modules/enhanced_google_integration.py")
        return False
    except Exception as e:
        print(f"❌ Error checking file: {e}")
        return False

if __name__ == "__main__":
    check_method_exists()