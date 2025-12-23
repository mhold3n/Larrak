import numpy as np

# Geometry
B = 0.1
S = 0.2
Vd = np.pi * (B / 2) ** 2 * S
print(f"Displacement: {Vd * 1000:.2f} Liters")

# Target
# at P=1 bar (1e5), T=300K
rho = 1e5 / (287 * 300)
m_air = Vd * rho
m_fuel = m_air / 14.7
Q_stoich = m_fuel * 44e6
print(f"Stoich Q at 1 bar: {Q_stoich:.2f} J")

# at P=2 bar
rho2 = 2e5 / (287 * 300)
m_air2 = Vd * rho2
Q_stoich2 = (m_air2 / 14.7) * 44e6
print(f"Stoich Q at 2 bar: {Q_stoich2:.2f} J")

# Current Logic Check
# Q_range = 3000 to 5000
test_q = 5000
# Code: p_bar = 2.0 * (test_q / 1000.0)
p_bar_code = 2.0 * (test_q / 1000.0)
print(f"\nCode Logic for Q={test_q}:")
print(f"  -> Generated Boost: {p_bar_code} bar")
# Check actual stoichiometry at this boost
rho_code = (p_bar_code * 1e5) / (287 * 300)
m_air_code = Vd * rho_code
Q_capacity = (m_air_code / 14.7) * 44e6
phi_result = test_q / Q_capacity
print(f"  -> Actual Capacity: {Q_capacity:.2f} J")
print(f"  -> Resulting Phi: {phi_result:.3f}")

# Proposed Fix
# Want Phi ~ 1.0 (or input phi)
# P_req = (Q / 44e6 * 14.7) / Vd * R * T
p_req = (test_q / 44e6 * 14.7) / Vd * 287 * 300
print(f"\nCorrect Pressure for Q={test_q} (Phi=1):")
print(f"  -> {p_req / 1e5:.3f} bar")
