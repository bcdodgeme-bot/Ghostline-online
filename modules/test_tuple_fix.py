# test_tuple_fix.py - Direct test to verify the tuple fix is working

import os
import sys

# Add current directory to Python path
sys.path.insert(0, '.')

def test_tuple_fix():
    print("Testing if Section 6 tuple fix is working...")
    
    try:
        from modules.enhanced_google_integration import GoogleIntegration
        print("✅ GoogleIntegration imported successfully")
        
        # Initialize the integration
        google_integration = GoogleIntegration()
        print("✅ GoogleIntegration initialized")
        
        # Test the specific method that was causing tuple errors
        print("\n🔍 Testing get_search_console_data_for_site method...")
        
        # This should return a dictionary, not a tuple
        result = google_integration.get_search_console_data_for_site(
            "nonexistent_site_test", 
            start_date='2024-01-01', 
            end_date='2024-01-07'
        )
        
        print(f"Return type: {type(result)}")
        print(f"Is dictionary: {isinstance(result, dict)}")
        print(f"Is tuple: {isinstance(result, tuple)}")
        
        if isinstance(result, tuple):
            print("🚨 TUPLE ERROR STILL EXISTS - Section 6 fix not applied correctly")
            print(f"Tuple contents: {result}")
            return False
        elif isinstance(result, dict):
            print("✅ Method returns dictionary correctly")
            print(f"Result keys: {list(result.keys())}")
            print(f"Success: {result.get('success')}")
            print(f"Error: {result.get('error')}")
            return True
        else:
            print(f"⚠️ Unexpected return type: {type(result)}")
            return False
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_tuple_fix()