"""
Phase 2 Interpreter Layer.
Bridges Thermo (Phase 1) and Campro (Phase 3) via Parametric NLP.
"""

from .interface import Interpreter
from .inversion import inverse_slider_crank, calculate_ideal_ratio
from .fitting import RatioCurveFitter
