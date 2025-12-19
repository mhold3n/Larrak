"""
Surrogate Module: Neural Network Surrogate for Engine Performance Prediction.
Translates Thermo Calibration results into Gear Kinematics inputs.

Previously named 'interpreter' - that name is deprecated.
"""
import warnings
import sys

# Deprecation alias: allow "from interpreter.X import Y" to work temporarily
def __getattr__(name):
    if name in ("model", "dataset", "breathing", "kinematics", "litvin", 
                "conjugate_nlp", "conjugate_optimizer", "gear_config", 
                "fea_check", "breathing_adapter", "validator"):
        warnings.warn(
            f"The 'interpreter' module name is deprecated. Use 'surrogate' instead. "
            f"Accessing interpreter.{name} via surrogate.",
            DeprecationWarning,
            stacklevel=2
        )
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module 'surrogate' has no attribute '{name}'")

# Also create a shim for the old 'interpreter' package name
# This allows "import interpreter" to redirect to "surrogate" with a warning
def _create_interpreter_shim():
    """Create a shim module that redirects 'interpreter' to 'surrogate'."""
    import types
    
    class InterpreterShim(types.ModuleType):
        def __getattr__(self, name):
            warnings.warn(
                f"The 'interpreter' module is deprecated. Use 'surrogate' instead.",
                DeprecationWarning,
                stacklevel=2
            )
            import importlib
            try:
                return importlib.import_module(f"surrogate.{name}")
            except ImportError:
                raise AttributeError(f"module 'interpreter' has no attribute '{name}'")
    
    shim = InterpreterShim("interpreter")
    shim.__path__ = []
    shim.__package__ = "interpreter"
    return shim

# Install the shim if 'interpreter' hasn't been imported yet
if "interpreter" not in sys.modules:
    sys.modules["interpreter"] = _create_interpreter_shim()
