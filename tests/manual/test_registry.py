"""
Test Calibration Registry.
"""
import sys
import os

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from thermo.calibration.registry import CalibrationRegistry, get_calibration

def test_registry():
    reg = CalibrationRegistry()
    print("Loaded Maps:", reg.maps.keys())
    
    # Test Friction Prediction
    # Test Friction Prediction
    print("\n--- Test Friction Map ---")
    if "friction" in reg.maps: # Cleaned key
        # Features: p_max_bar, rpm
        inputs = {"p_max_bar": 100.0, "rpm": 3000.0}
        fmep = reg.predict("friction", inputs)
        print(f"Inputs: {inputs}")
        print(f"Predicted FMEP: {fmep}")
        
        # Sanity Check
        val = fmep.get("value", 0.0)
        assert val > 0.0, "FMEP should be positive"
        assert val < 5.0, "FMEP should be reasonable"
    else:
        print("FAIL: Friction Map not loaded.")
        
    # Test Combustion Prediction
    print("\n--- Test Combustion Map ---")
    if "combustion" in reg.maps:
        # Features: jet_intensity
        # Jet = (V_pre / A_nozzle) * (RPM / 1000)
        # e.g. (1e-6 / 3e-6) * (3000/1000) = 0.33 * 3 = 1.0
        inputs = {"jet_intensity": 1.0}
        wiebe = reg.predict("combustion", inputs)
        print(f"Inputs: {inputs}")
        print(f"Predicted Wiebe: {wiebe}")
        
    else:
        print("FAIL: Combustion Map not loaded.")

if __name__ == "__main__":
    test_registry()
