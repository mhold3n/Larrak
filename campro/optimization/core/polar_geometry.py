import numpy as np
from scipy.interpolate import BPoly


class PolarCamGeometry:
    """
    Planet-Ring Polar Geometry parameterization.

    Defines a radial profile r(theta) based on:
    - Stroke (Mean Planet Diameter)
    - Outer Diameter limit
    - BDC angle (theta_bdc)
    - Inflection point locations (relative lambda)
    - Desired gear ratios (radial velocities) at inflection points
    """

    def __init__(
        self,
        stroke: float,
        outer_diameter: float,
        theta_bdc: float,
        ratios: tuple[float, float],
        inflections: tuple[float, float] = (0.5, 0.5),
        center_offset: float = 0.0,
        gen_mode: str = "spline",  # "spline" or "hypocycloid"
        r_drive: float | None = None,  # Used for hypocycloid mode
    ):
        """
        Initialize the geometry.

        Args:
            stroke: Total travel distance (r_max - r_min).
            outer_diameter: Maximum diameter constraint (defines r_max = OD/2).
            theta_bdc: Angle of Bottom Dead Center (radians).
            ratios: Tuple of (ratio_down, ratio_up).
            inflections: Tuple of (lambda_1, lambda_2).
            center_offset: Shift in r_mean (unused here).
            gen_mode: "spline" (heuristic Hermite) or "hypocycloid" (analytical golden).
            r_drive: Radius of the drive point from planet center.
                     Required if gen_mode="hypocycloid".
        """
        self.stroke = stroke
        self.r_max = outer_diameter / 2.0
        self.r_min = self.r_max - stroke
        self.theta_bdc = theta_bdc
        self.gen_mode = gen_mode
        self.r_drive = r_drive

        # Spline parameters (used if gen_mode="spline")
        self.ratio_1 = -abs(ratios[0])
        self.lam_1 = inflections[0]
        self.ratio_2 = abs(ratios[1])
        self.lam_2 = inflections[1]

        self._poly = None
        self._poly_d1 = None
        self._poly_d2 = None

        if self.gen_mode == "hypocycloid":
            if self.r_drive is None:
                raise ValueError("r_drive must be provided for hypocycloid generation.")
            self._build_hypocycloid()
        else:
            self._build_spline()

    def _build_hypocycloid(self):
        """
        Construct analytical profile based on 2:1 Hypocycloid (Ellipse).

        Geometry:
        - Ring Radius R = r_max (Outer Limit) implies the ring surface.
          Actually, r_max in our context refers to the MAXIMUM PISTON EXTENSION.
        - In a Planet-Ring setup:
          R_ring = 2 * R_planet (for 2:1 ratio).
          Center Distance = R_planet.
          Drive Point Radius = r_drive (from planet center).

        - Path Equations (centered on Ring):
          x(phi) = (R_ring - R_planet)*cos(phi) + r_drive*cos(phi*(R_ring-R_planet)/R_planet)
          Since R_ring = 2*R_planet:
          (R-r)/r = 1.
          So terms sum up:
          x(phi) = (R_planet + r_drive) * cos(phi)
          y(phi) = (R_planet - r_drive) * sin(phi)

        - This forms an Ellipse with semi-axes a = R_p + r_d, b = |R_p - r_d|.
        - We map this to r(theta) by defining r = sqrt(x^2 + y^2).
          However, theta in our engine cycle is the CRANK ANGLE (phi).
          r(theta) is the radial distance from center.

        - Constraints:
          Max Radius (r_max) = a = R_planet + r_drive
          Min Radius (r_min) = b = |R_planet - r_drive|
          Stroke = r_max - r_min = 2 * r_min (if going to center) or just a - b.
          a - b = (Rp + rd) - (Rp - rd) = 2*rd (if Rp > rd).

        - Solving for R_planet and r_drive from User Constraints (Stroke, OD):
          r_drive = Stroke / 2.0
          R_planet = r_max - r_drive = (OD/2) - (Stroke/2).

        - Note: The user provided r_drive EXPLICITLY.
          If provided, it overrides the derived value?
          Or should we derive the profile strictly from r_drive and R_planet?
          Let's use the USER r_drive and derived R_planet to match r_max.

          R_planet = self.r_max - self.r_drive

          If this conflicts with Stroke, we warn?
          Actually, the Hypocycloid defines the stroke EXACTLY as 2*r_drive.
          So if r_drive is provided, Stroke is fixed by it.
        """
        R_planet = self.r_max - self.r_drive
        rd = self.r_drive

        # Ellipse parameters
        A = R_planet + rd  # Semi-major axis (x direction, at theta=0)
        B = abs(R_planet - rd)  # Semi-minor axis (y direction, at theta=90)

        # For analytical evaluation, we don't need a spline.
        # But to match the API (which might expect .evaluate), we can:
        # A) Implement a custom method that computes analytical values.
        # B) Sample and spline it (easier for integration with existing BPoly logic).
        # Let's do B (Sample & Spline) for robustness with the existing Solver tools
        # that expect polynomial chunks. Or better, just store the ellipse params
        # and override evaluate().

        self._ellipse_params = (A, B)

    def _evaluate_hypocycloid(self, theta, d=0):
        """Analytical derivative of ellipse radius r(theta)."""
        # r(theta) = sqrt( (A cos t)^2 + (B sin t)^2 )
        # This assumes the 'theta' passed in corresponds to the 'phi' parameter of the ellipse.
        # In a 2:1 gear, the output angle theta IS the planet roll angle phi.

        A, B = self._ellipse_params
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        r_sq = (A * cos_t) ** 2 + (B * sin_t) ** 2
        r = np.sqrt(r_sq)

        if d == 0:
            return r

        # First derivative: dr/dtheta
        # dr/dt = (1/2r) * d(r^2)/dt
        # d(r^2)/dt = 2(A cos)(-A sin) + 2(B sin)(B cos) = 2(B^2 - A^2) sin cos
        #           = (B^2 - A^2) sin(2t)
        dr = ((B**2 - A**2) * sin_t * cos_t) / r

        if d == 1:
            return dr

        # Second derivative
        # d2r = d(dr)/dt
        # complex quotient rule...
        # Let N = (B^2 - A^2) sin t cos t
        # dr = N / r
        # d2r = (r dN - N dr) / r^2
        # dN = (B^2 - A^2) (cos^2 t - sin^2 t) = (B^2 - A^2) cos(2t)

        term = B**2 - A**2
        dN = term * (cos_t**2 - sin_t**2)
        N = term * sin_t * cos_t

        d2r = (r * dN - N * dr) / (r**2)
        return d2r

    def _build_spline(self):
        """Construct the piecewise cubic Hermite spline."""
        # Nodes
        # 0: TDC (0)
        # 1: Inflection 1
        # 2: BDC (theta_bdc)
        # 3: Inflection 2
        # 4: TDC_next (2*pi)

        t_tdc1 = 0.0
        t_bdc = self.theta_bdc
        t_tdc2 = 2 * np.pi

        # Angular positions of inflection points
        t_inf1 = t_tdc1 + self.lam_1 * (t_bdc - t_tdc1)
        t_inf2 = t_bdc + self.lam_2 * (t_tdc2 - t_bdc)

        points = [t_tdc1, t_inf1, t_bdc, t_inf2, t_tdc2]

        # Radial positions (r)
        # We assume inflection points are roughly at mid-stroke for the 'roughing' pass.
        # In a full optimization, r_inf might be a free variable.
        # Here we fix it to mean arithmetic radius for simplicity in initialization.
        r_mid = (self.r_max + self.r_min) / 2.0

        # Boundary conditions (val, derivative)
        # TDC: r_max, 0
        # Inf1: r_mid, ratio_1
        # BDC: r_min, 0
        # Inf2: r_mid, ratio_2
        # TDC2: r_max, 0

        # Note: BPoly.from_derivatives expects xi (sites) and yi (values/derivatives)
        # yi shape: (n_points, n_derivatives)
        # We provide r and r' (1st derivative)

        # To ensure smoothness, we might want r'' continuity, but standard Hermite
        # only guarantees C1 if we specify derivatives.
        # For "inflection" implies r''=0 usually, but here we just mean "midpoint".
        # Let's enforcing value & slope.

        r_vals = np.array(
            [
                [self.r_max, 0.0],  # TDC
                [r_mid, self.ratio_1],  # Inf1
                [self.r_min, 0.0],  # BDC
                [r_mid, self.ratio_2],  # Inf2
                [self.r_max, 0.0],  # TDC_next
            ]
        )

        self._poly = BPoly.from_derivatives(points, r_vals)
        self._poly_d1 = self._poly.derivative(1)
        self._poly_d2 = self._poly.derivative(2)

    def evaluate(self, theta: float | np.ndarray) -> tuple[float, float, float]:
        """
        Evaluate r, r', r'' at given angle theta.
        """
        # Wrap theta to [0, 2pi)
        t_wrapped = np.mod(theta, 2 * np.pi)

        if self.gen_mode == "hypocycloid":
            # Handle array inputs correctly for analytical func
            if np.isscalar(t_wrapped):
                r = self._evaluate_hypocycloid(t_wrapped, d=0)
                dr = self._evaluate_hypocycloid(t_wrapped, d=1)
                d2r = self._evaluate_hypocycloid(t_wrapped, d=2)
            else:
                # Vectorized
                r = self._evaluate_hypocycloid(t_wrapped, d=0)
                dr = self._evaluate_hypocycloid(t_wrapped, d=1)
                d2r = self._evaluate_hypocycloid(t_wrapped, d=2)
            return r, dr, d2r

        r = self._poly(t_wrapped)
        dr = self._poly_d1(t_wrapped)
        d2r = self._poly_d2(t_wrapped)

        return r, dr, d2r

    def get_kinematics(self, t: float, omega: float = 1.0):
        """
        Get time-domain kinematics given a rotation speed.

        Args:
            t: Time (s) - used to derive theta = omega * t
            omega: Angular velocity (rad/s)

        Returns:
            x (m), v (m/s), a (m/s^2)
        """
        theta = (omega * t) % (2 * np.pi)
        r, dr, d2r = self.evaluate(theta)

        # x = r
        # v = dr/dt = dr/dtheta * dtheta/dt = dr * omega
        # a = dv/dt = d(dr*omega)/dt = d2r/dtheta * omega * omega

        x = r
        v = dr * omega
        a = d2r * (omega**2)

        return x, v, a
