import casadi as ca
import numpy as np


class SymbolicPolarCam:
    """
    CasADi-compatible implementation of the Planet-Ring Polar Geometry.

    This allows the inflection points (lambda) to be symbolic variables,
    enabling the optimizer to adjust the cam shape.
    """

    def __init__(
        self,
        stroke: float,
        outer_diameter: float,
        theta_bdc: float,
        ratios: tuple[float, float],
    ):
        self.stroke = stroke
        self.r_max = outer_diameter / 2.0
        self.r_min = self.r_max - stroke
        self.theta_bdc = theta_bdc

        # Fixed Gear Ratios
        self.ratio_1 = -abs(ratios[0])  # Downstroke
        self.ratio_2 = abs(ratios[1])  # Upstroke

        # Midpoint radius (fixed for now as per roughing strategy)
        self.r_mid = (self.r_max + self.r_min) / 2.0

        # Generator Mode
        self.gen_mode = "spline"  # default

    def set_mode(self, mode: str):
        self.gen_mode = mode

    def evaluate(self, theta, lam1, lam2):
        """
        Evaluate r(theta) symbolically.

        Args:
            theta: CasADi expression or float for angle (0 to 2pi).
            lam1: Normalized position of inflection 1 (0 to 1). (Ignored in hypocycloid)
            lam2: Normalized position of inflection 2 (0 to 1). (Ignored in hypocycloid)

        Returns:
            r: Radial position
        """
        if self.gen_mode == "hypocycloid":
            # Analytical Ellipse: A=r_max, B=r_min (assuming stroke = 2*rd)
            # r(theta) = sqrt( (rmax cos t)^2 + (rmin sin t)^2 )
            A = self.r_max
            B = self.r_min

            # Ensure theta is respected as rotational angle
            # For 2:1 planet ring, theta (crank) maps to ellipse parameter directly?
            # PolarGeometry says: x = (Rp+rd)cos(t), y=(Rp-rd)sin(t) -> Ellipse.
            # Yes.

            cos_t = ca.cos(theta)
            sin_t = ca.sin(theta)

            r_sq = (A * cos_t) ** 2 + (B * sin_t) ** 2
            r = ca.sqrt(r_sq)
            return r

        # Node locations
        # Node locations
        t_tdc1 = 0.0
        t_bdc = self.theta_bdc
        t_tdc2 = 2 * np.pi

        # Inflection points depend on lambda
        t_inf1 = t_tdc1 + lam1 * (t_bdc - t_tdc1)
        t_inf2 = t_bdc + lam2 * (t_tdc2 - t_bdc)

        # Values and Slopes at nodes
        # Nodes: [TDC1, Inf1, BDC, Inf2, TDC2]
        # x: [0, t_inf1, t_bdc, t_inf2, 2pi]
        # y: [r_max, r_mid, r_min, r_mid, r_max]
        # m: [0, ratio_1, 0, ratio_2, 0]

        # We need to dispatch based on which interval theta is in.
        # Intervals:
        # 1. 0 -> Inf1
        # 2. Inf1 -> BDC
        # 3. BDC -> Inf2
        # 4. Inf2 -> 2pi

        r1 = self._hermite_cubic(
            theta, t_tdc1, t_inf1, self.r_max, self.r_mid, 0.0, self.ratio_1
        )
        r2 = self._hermite_cubic(
            theta, t_inf1, t_bdc, self.r_mid, self.r_min, self.ratio_1, 0.0
        )
        r3 = self._hermite_cubic(
            theta, t_bdc, t_inf2, self.r_min, self.r_mid, 0.0, self.ratio_2
        )
        r4 = self._hermite_cubic(
            theta, t_inf2, t_tdc2, self.r_mid, self.r_max, self.ratio_2, 0.0
        )

        # Select using if_else
        # Note: Order matters for efficiency, likely nested if_else

        # Segment 3/4 (theta > BDC)
        #   Segment 4 (theta > Inf2) -> r4
        #   Segment 3 (theta <= Inf2) -> r3
        upper_half = ca.if_else(theta > t_inf2, r4, r3)

        # Segment 1/2 (theta <= BDC)
        #   Segment 2 (theta > Inf1) -> r2
        #   Segment 1 (theta <= Inf1) -> r1
        lower_half = ca.if_else(theta > t_inf1, r2, r1)

        return ca.if_else(theta > t_bdc, upper_half, lower_half)

    def _hermite_cubic(self, t, x0, x1, y0, y1, m0, m1):
        """
        Evaluate cubic Hermite spline on interval [x0, x1].
        """
        h = x1 - x0
        # Prevent division by zero if x0=x1 (though lambdas should prevent this)
        h = ca.fmax(h, 1e-6)

        t_norm = (t - x0) / h

        # Basis functions
        # h00 = 2t^3 - 3t^2 + 1
        # h10 = t^3 - 2t^2 + t
        # h01 = -2t^3 + 3t^2
        # h11 = t^3 - t^2

        t2 = t_norm * t_norm
        t3 = t2 * t_norm

        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t_norm
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2

        return y0 * h00 + h * m0 * h10 + y1 * h01 + h * m1 * h11
