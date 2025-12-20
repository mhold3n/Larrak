# Relating \(R(\theta)\), Cam Rotation, and Linear Follower Displacement

## Overview

This document provides a complete, self-contained framework to relate:
- the **ring / radial follower** instantaneous radius \(R(\psi)\) (2nd follower),
- the **cam** rotation \(\theta\) and its contacting curve geometry, and
- the **linear follower** displacement \(x\) (1st follower).

We separate **geometry**, **rolling kinematics**, and **time-law/ODEs**, then give a minimal algorithm and design checks.

---

## 0) Notation

| Symbol | Meaning |
|---|---|
| \(\theta\) | Cam angle about its center (CCW positive) |
| \(\psi\) | Ring (2nd follower) rotation about its center (CCW positive) |
| \(x(\theta)\) | Linear follower (1st) displacement law vs cam angle |
| \(r_p(\theta)\) | Cam **pitch curve** radius (locus of the 1st follower’s roller center) |
| \(r_n(\theta)\) | Cam **profile** radius (working surface) |
| \(\kappa_c(\theta)\) | Curvature of the cam’s **contacting** curve at the contact point |
| \(\rho_c(\theta)=1/\kappa_c(\theta)\) | Cam’s local osculating radius at contact |
| \(R(\psi)\) | Ring’s instantaneous radius; \(\kappa_s(\psi)=1/R(\psi)\) |
| \(\omega=d\theta/dt,\; \alpha=d\omega/dt\) | Cam angular speed/acceleration |
| \(\Omega=d\psi/dt\) | Ring angular speed |

---

## 1) Geometry: from the 1st follower’s motion law to the cam curve

Assume the 1st follower is a **translating roller** of radius \(R_1\) and its lift law is \(x(\theta)\). Then the cam’s pitch curve (roller-center locus) and working profile are:
\[
r_p(\theta) = r_b + x(\theta), \qquad
r_n(\theta) = r_p(\theta) - R_1 .
\]

If the ring contacts an **offset** of this curve (e.g., due to a ring-side roller of radius \(d\) with sign), transform curvature via the parallel-curve law
\[
\kappa_{\text{offset}}(\theta) \;=\; \frac{\kappa_c(\theta)}{1 + d\,\kappa_c(\theta)}.
\]

For a curve given in **polar form** \(r(\theta)\), the curvature is
\[
\kappa_c(\theta) \;=\; \frac{r^2 + 2\,(r')^2 - r\,r''}{\bigl(r^2 + (r')^2\bigr)^{3/2}},
\quad r'=\frac{dr}{d\theta},\; r''=\frac{d^2r}{d\theta^2},
\]
hence \(\rho_c(\theta) = 1/\kappa_c(\theta)\).

> **Which curve’s curvature?** Use the curvature of the **contacting** curve at the cam–ring contact (the actual profile if knife-edge contact, or the appropriate offset if a roller is present).

---

## 2) Rolling kinematics at the cam–ring contact (no slip)

Pure rolling implies equal tangential speeds at the contact. Using Frenet kinematics (“spin rate = curvature × tangent speed”), eliminating the common contact arclength rate yields the **pitch-curve meshing law**:
\[
\boxed{\;\rho_c(\theta)\,d\theta \;=\; R(\psi)\,d\psi\;}
\quad\Longleftrightarrow\quad
\boxed{\;\frac{d\psi}{d\theta} \;=\; \frac{\rho_c(\theta)}{R(\psi)}\;}
\]
(Sign changes encode internal vs external contact; magnitudes are shown here.)

**Sanity checks.**
- If both are circular (\(\rho_c = r\) constant, \(R(\psi)=R\) constant), then \(d\psi/d\theta = r/R\) (constant gear ratio).
- If the ring is "flat" (\(R\to\infty \Rightarrow \kappa_s=0\)), the constraint reduces to the rack limit with \(v = \omega\,\rho_c\).

**Implementation Note (2024):** Due to numerical instability in the ODE solver for complex cam geometries, the current implementation uses a simplified approach that always generates complete 360° profiles. When the meshing law ODE fails to converge, the system falls back to linear approximation to ensure complete profile coverage. This pragmatic approach prioritizes functionality over perfect physics accuracy.

---

## 3) Time-law chaining to the linear follower \(x(t)\)

The linear follower’s **displacement vs time** is
\[
x(t) \;=\; x\!\bigl(\theta(t)\bigr),
\]
with chain-rule kinematics
\[
\dot x = x'(\theta)\,\omega, \qquad
\ddot x = x''(\theta)\,\omega^2 + x'(\theta)\,\alpha.
\]

The rolling constraint couples \(\theta\) and \(\psi\) in time:
\[
\frac{d\psi}{dt} \;=\; \frac{\rho_c(\theta)}{R(\psi)}\,\omega(t),
\qquad
\frac{d\theta}{dt} \;=\; \frac{R(\psi)}{\rho_c(\theta)}\,\Omega(t).
\]

Thus, specifying **any one** time-law (e.g., \(\omega(t)\) or \(\Omega(t)\)) determines the other via the rolling relation, after which \(x(t)\) follows directly.

---

## 4) Mapping “input surface” → “output surface”

You design the **input** via the linear follower excitation, i.e., you choose \(x(\theta)\). The **output** is the ring rotation. The synthesis pipeline is:

1. **Lift → cam curve.** Choose \(x(\theta)\) (with desired continuity), compute \(r_p(\theta)\), select the contacting curve (profile or offset), and compute \(\kappa_c(\theta)\) (hence \(\rho_c(\theta)\)).  
2. **Pick/design \(R(\psi)\).** This is the ring’s instantaneous radius (its “pitch” locus).  
3. **Relate angles.** Integrate the meshing law \(\rho_c(\theta)\,d\theta = R(\psi)\,d\psi\) to obtain \(\psi(\theta)\) (or \(\theta(\psi)\)).  
4. **Time-law (optional).** If you want a particular time behavior (e.g., constant \(\Omega\) or constant \(\omega\)), solve the corresponding first-order ODE (next section), then evaluate \(x(t)=x(\theta(t))\).

**Turns-per-turn (global ratio).**
\[
N_{\text{cam per ring}}
= \frac{1}{2\pi}\,\int_{0}^{2\pi} \frac{d\theta}{d\psi}\,d\psi
= \frac{1}{2\pi} \int_{0}^{2\pi} \frac{\rho_c\!\bigl(\theta(\psi)\bigr)}{R(\psi)}\,d\psi.
\]
For circular elements this reduces to \(N = R/r\).

---

## 5) Design targets and implied ODEs

### A) Ring at constant angular speed \(\Omega(t)\equiv\Omega_0\)

Then \(\psi(t)=\psi_0+\Omega_0 t\) and \(\theta(t)\) satisfies
\[
\frac{d\theta}{dt} \;=\; \frac{R\!\bigl(\psi_0+\Omega_0 t\bigr)}{\rho_c\!\bigl(\theta(t)\bigr)}\,\Omega_0.
\]
From \(\theta(t)\) obtain \(x(t), \dot x(t), \ddot x(t)\).

### B) Cam at constant angular speed \(\omega(t)\equiv\omega_0\)

Then \(\theta(t)=\theta_0+\omega_0 t\) and \(\psi(t)\) satisfies
\[
\frac{d\psi}{dt} \;=\; \frac{\rho_c\!\bigl(\theta_0+\omega_0 t\bigr)}{R\!\bigl(\psi(t)\bigr)}\,\omega_0.
\]

---

## 6) Practical checks

1. **Use the contacting curve.** If a roller/clearance offset exists on either element, transform curvature with \(\kappa_d=\kappa/(1+d\kappa)\). Avoid \(|d\,\kappa|\ge 1\) (cusps/undercuts).  
2. **Contact side and signs.** Internal vs external contact flips the sign in the differential law; use magnitudes in planning, then apply the correct sign convention.  
3. **Smoothness/jerk.** Select \(x(\theta)\) and \(R(\psi)\) to avoid large curvature spikes that would force \(\omega\) or \(\Omega\) to vary excessively and violate dynamic limits.  
4. **Consistency of contact point.** Evaluate \(\kappa_c(\theta)\) at the instantaneous contact; for cams this is naturally parameterized by \(\theta\).

---

## 7) Minimal algorithm (ready to implement)

```python
# Inputs:
#   x(theta)      -> 1st follower lift law (symbolic/numeric)
#   R(psi)        -> ring instantaneous radius
#   driver        -> 'cam' with omega(t) or 'ring' with Omega(t)

# Step 1: Build contacting cam curve r_contact(theta) (profile or offset)
# Step 2: Compute kappa_c(theta) from polar curvature; rho_c(theta) = 1/kappa_c
# Step 3: Define and integrate the ODEs:
#   if driver == 'ring':
#       psi(t)    = psi0 + ∫ Omega(t) dt
#       dtheta/dt = [ R(psi(t)) / rho_c(theta) ] * Omega(t)
#   if driver == 'cam':
#       theta(t)  = theta0 + ∫ omega(t) dt
#       dpsi/dt   = [ rho_c(theta(t)) / R(psi) ] * omega(t)
# Step 4: Evaluate x(t) = x(theta(t))
# Step 5: Optionally compute xdot, xddot via chain rule
```

---

## References (links include pages with LaTeX/BibTeX citation options)

- **Curvature of a polar curve** — Wolfram MathWorld:  
  https://mathworld.wolfram.com/Curvature.html

- **Parallel (offset) curve; curvature transform** — Wikipedia “Parallel curve” (has “Cite this page”):  
  https://en.wikipedia.org/wiki/Parallel_curve

- **Cams: pitch curve, working profile, roller-center locus** — CMU, *Chapter 6: Cams* (downloadable notes with citations):  
  https://www.cs.cmu.edu/~rapidproto/mechanisms/chpt6.html

- **Gears and equal tangential speeds at the pitch point** — CMU, *Chapter 7: Gears*:  
  https://www.cs.cmu.edu/~rapidproto/mechanisms/chpt7.html

- **Noncircular gear transmission on pitch curves (synthesis from a desired function)** — Bäsel, *Determining the geometry of noncircular gears for given transmission function*, arXiv:1905.02642 (BibTeX on arXiv):  
  https://arxiv.org/abs/1905.02642

- **Design and kinematics of noncircular gears** — Maláková et al., *Applied Sciences* 2021 (MDPI provides BibTeX):  
  https://www.mdpi.com/2076-3417/11/14/6424
