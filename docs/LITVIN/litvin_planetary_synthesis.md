# Litvin Planetary Synthesis — Design Draft

This document describes the 2D conjugate synthesis for an internal ring gear driving a Litvin planet with center motion constrained by a fixed radial slot. It captures rationale and invariants; once implemented and tested, it will be archived.

Key points:
- Motion law: θ_p(θ_r)=2θ_r, center distance d(θ_r)=R0+c(θ_r), TDC at θ_r∈{0,π}.
- Envelope condition in planet frame: normal velocity component of the transformed ring flank vanishes.
- Predictor–corrector tracks the contact parameter along θ_r.
- Optimization proceeds in staged orders (0..3) from evaluation to micro-geometry collocation and optional motion-law co-optimization.


