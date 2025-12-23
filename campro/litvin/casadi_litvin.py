from __future__ import annotations

from typing import Any

import casadi as cs


def casadi_involute_xy(rb: Any, phi: Any) -> tuple[Any, Any]:
    """
    Parametric involute of a circle of radius rb.
    Symbolic CasADi implementation.
    """
    c = cs.cos(phi)
    s = cs.sin(phi)
    x = rb * (c + phi * s)
    y = rb * (s - phi * c)
    return x, y


def casadi_involute_tangent(rb: Any, phi: Any) -> tuple[Any, Any]:
    """
    Derivative of involute curve d/dphi.
    T = [-sin(phi) + sin(phi) + phi*cos(phi), cos(phi) - cos(phi) + phi*sin(phi)] * rb
      = [phi*cos(phi), phi*sin(phi)] * rb
    """
    c = cs.cos(phi)
    s = cs.sin(phi)
    tx = rb * phi * c
    ty = rb * phi * s
    return tx, ty


def casadi_rotate(theta: Any, x: Any, y: Any) -> tuple[Any, Any]:
    """Rotate point (x,y) by theta."""
    c = cs.cos(theta)
    s = cs.sin(theta)
    nx = c * x - s * y
    ny = s * x + c * y
    return nx, ny


def casadi_planet_transform(
    x_ring: Any, y_ring: Any, theta_r: Any, d: Any, theta_p: Any
) -> tuple[Any, Any]:
    """
    Transform a point from Ring Frame to Planet Frame.

    Args:
        x_ring, y_ring: Point in Ring Frame (e.g. Involute).
        theta_r: Ring Rotation Angle (World Frame).
        d: Center distance (distance from Ring center to Planet center).
        theta_p: Planet Rotation Angle (relative to World? Or local?).

    Matches logic in planetary_synthesis._planet_coords:
    1. Rotate Ring -> World by theta_r
    2. Translate World -> Relative by (-d, 0) [assuming planet center is at (d, 0) in rotated frame?]
       Wait, in _planet_coords:
       d = center_distance(theta_r)
       x_rel = x_world - d
       So planet center is at (d, 0) in the WORLD frame?
       No, usually 'd' is radial distance.
       Let's verify _planet_coords:
         x_world, y_world = _rotate(theta_r, x_ring, y_ring)
         d = kin.center_distance(theta_r)
         x_rel = x_world - d  <-- Implies planet center is on X-axis of World?
                             Only true if the "World" frame rotates with the slot center?
                             Or if theta_r is measured relative to the slot axis.

    Ref: `kinematics.py` -> `center_distance`
    """
    # 1. Ring -> World (Rotation by theta_r)
    x_w, y_w = casadi_rotate(theta_r, x_ring, y_ring)

    # 2. World -> Relative (Translation by -d)
    # This assumes the planet center is always on the X-axis of the reference frame
    # used for theta_r.
    x_rel = x_w - d
    y_rel = y_w

    # 3. Relative -> Planet (Rotation by -theta_p)
    x_p, y_p = casadi_rotate(-theta_p, x_rel, y_rel)

    return x_p, y_p


def casadi_conjugacy_residual(
    rb: Any,  # Base radius (can be variable)
    theta_r: Any,  # Ring angle (driver)
    d: Any,  # Center distance (function of theta_r, or variable)
    theta_p: Any,  # Planet angle (function of theta_r, or variable)
    phi: Any,  # Contact roll angle (variable)
    d_prime: Any,  # dd/dtheta_r
    theta_p_prime: Any,  # dtheta_p/dtheta_r
) -> Any:
    """
    Compute the conjugacy residual (Mesh Condition).
    Condition: Relative velocity at contact point must be tangent to the surface.

    Mathematically: (v_rel . n) = 0  => (v_rel . normal) = 0
    Or equivalently: (v_rel x tangent) = 0 (in 2D z-component)

    v_rel is the velocity of the potential contact point (defined by phi)
    as seen in the Planet Frame, due to the kinematic motion (changing theta_r).

    We compute d(Point_Planet)/d(theta_r) holding phi constant.
    """

    # Ring Point (Fixed in Ring Frame for a given phi)
    xr, yr = casadi_involute_xy(rb, phi)

    # Tangent in Ring Frame
    txr, tyr = casadi_involute_tangent(rb, phi)

    # Transform Point to Planet Frame
    xp, yp = casadi_planet_transform(xr, yr, theta_r, d, theta_p)

    # Transform Tangent to Planet Frame (Rotation only)
    # The translation 'd' does not affect the vector.
    # But the rotations do.
    # T_world = Rotate(theta_r) * T_ring
    tx_w, ty_w = casadi_rotate(theta_r, txr, tyr)
    # T_planet = Rotate(-theta_p) * T_world
    tx_p, ty_p = casadi_rotate(-theta_p, tx_w, ty_w)

    # Compute Kinematic Velocity v_k = d(Point_Planet)/d(theta_r)
    # We use the chain rule with the provided derivatives d' and theta_p'
    # point_P = Rotate(-theta_p) * (Rotate(theta_r)*P_ring - [d, 0])

    # Let's verify specific derivatives:
    # P_w = [c_r*xr - s_r*yr, s_r*xr + c_r*yr]
    # dP_w/dtheta_r = [-s_r*xr - c_r*yr, c_r*xr - s_r*yr] = [-yw, xw]
    # So v_w_rot = [-yw, xw] (Velocity of ring point in world frame due to rotation)

    # P_rel = P_w - [d, 0]
    # dP_rel/dtheta_r = v_w_rot - [d_prime, 0]

    # P_p = Rotate(-theta_p) * P_rel
    # dP_p/dtheta_r = d(Rotate(-theta_p))/dtheta_r * P_rel + Rotate(-theta_p) * dP_rel/dtheta_r
    #               = -theta_p_prime * [-sin(-tp), -cos(-tp); cos(-tp), -sin(-tp)] * P_rel  + ...
    #               Basically velocity of rotation -theta_p + transformed linear velocity.

    # Implementation using exact derivatives (or AD if we passed expressions)
    # But since d, theta_p are scalar inputs here (expressions), CasADi will handle it
    # IF we build the graph of xp, yp in terms of theta_r directly.
    # BUT, d and theta_p are separate inputs. The dependency on theta_r is implicit
    # or passed via d_prime.

    # Let's do it explicitly with terms:

    c_r = cs.cos(theta_r)
    s_r = cs.sin(theta_r)
    xr_w = c_r * xr - s_r * yr
    yr_w = s_r * xr + c_r * yr

    # Derivative of World Pos w.r.t theta_r
    dxr_w = -yr_w  # standard rotation derivative
    dyr_w = xr_w

    # Relative Pos derivatives
    # d_prime is dd/dtheta_r
    dx_rel_dt = dxr_w - d_prime
    dy_rel_dt = dyr_w

    # Planet Point P_p
    # Using casadi_rotate for x_rel, y_rel by -theta_p
    # let alpha = -theta_p
    alpha = -theta_p
    c_a = cs.cos(alpha)
    s_a = cs.sin(alpha)

    xp_val = c_a * (xr_w - d) - s_a * yr_w
    yp_val = s_a * (xr_w - d) + c_a * yr_w

    # Derivative w.r.t theta_r
    # term 1: d/dalpha * dalpha/dtheta_r * P_rel
    # dalpha = -theta_p_prime
    # d(Rotate(alpha))/dalpha * P = [-s_a, -c_a; c_a, -s_a] * P = [-yp_val, xp_val] ? No
    # Rotation matrix R(a) = [[c, -s], [s, c]]
    # R'(a) = [[-s, -c], [c, -s]] = R(a + pi/2)
    # R'(a) * v = R(a) * [[0, -1], [1, 0]] * v = R(a) * [-y, x]
    # So term 1 is: -theta_p_prime * (Velocity due to planet rotation)
    # The vector in planet frame is (xp_val, yp_val).
    # Rotation of planet frame induces velocity: -theta_p_prime * [-yp_val, xp_val]?
    # Wait, carefully:
    # P_p = R(alpha) * P_rel
    # dP_p/dt = R'(alpha)*alpha' * P_rel + R(alpha) * dP_rel/dt
    #         = alpha' * R(alpha) * [[0, -1], [1, 0]] * P_rel + R(alpha) * V_rel

    # P_rel vector (unrotated)
    xrel = xr_w - d
    yrel = yr_w

    # Cross term (velocity from rotating frame)
    # R(alpha) * [-yrel, xrel]  <-- This is [-yp_val, xp_val]
    # So v_rot = alpha' * [-yp_val, xp_val]
    v_rot_x = -theta_p_prime * (-yp_val)
    v_rot_y = -theta_p_prime * (xp_val)

    # Linear term (transform of world velocity)
    # R(alpha) * [dx_rel_dt, dy_rel_dt]
    v_lin_x, v_lin_y = casadi_rotate(alpha, dx_rel_dt, dy_rel_dt)

    # Total relative velocity dPoint/dtheta_r
    vx = v_rot_x + v_lin_x
    vy = v_rot_y + v_lin_y

    # Conjugacy condition: Cross product of Tangent and Velocity
    # Tangent in Planet Frame
    tx_p, ty_p = casadi_rotate(alpha, tx_w, ty_w)  # Should match previous calculation

    # Cross product (2D): ax*by - ay*bx
    residual = tx_p * vy - ty_p * vx

    return residual


def casadi_undercut_check() -> float:
    """Placeholder for undercut check."""
    return 0.0
