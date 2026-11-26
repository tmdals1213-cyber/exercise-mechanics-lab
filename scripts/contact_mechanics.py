"""Explicit contact force contracts, independent of the OHP movement driver.

World axes are the model axes, Z up. Wrenches here are [force, moment].
This is an external-force interface; it is NOT a contact trajectory solver.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Contact:
    name: str
    body_id: int
    local_point: np.ndarray
    kind: str = 'unilateral_point'
    frame_world: np.ndarray = field(default_factory=lambda: np.eye(3))
    friction: float = .8
    active: bool = True

    def inequalities(self, faces=16):
        """Return A,lo,hi for a WORLD force (N).

        Inscribed regular polygon: every feasible tangential force satisfies
        ||Ft|| <= mu*Fn, unlike independent axis bounds (a circumscribed square).
        A grasp is an ideal bilateral point constraint, with NO free moment.
        """
        R = np.asarray(self.frame_world, float)
        if R.shape != (3, 3) or not np.allclose(R.T @ R, np.eye(3), atol=1e-10) or np.linalg.det(R) < .999999:
            raise ValueError('Contact frame must be a right-handed orthonormal basis')
        if self.friction < 0 or faces < 4:
            raise ValueError('Invalid contact friction or polygon size')
        if not self.active:
            return np.eye(3), np.zeros(3), np.zeros(3)
        if self.kind == 'bilateral_point':
            return np.empty((0, 3)), np.empty(0), np.empty(0)
        if self.kind != 'unilateral_point':
            raise ValueError('Unsupported contact kind: ' + self.kind)
        rows = [[0., 0., -1.]]
        for phi in np.linspace(0, 2*np.pi, faces, endpoint=False):
            rows.append([np.cos(phi), np.sin(phi), -self.friction*np.cos(np.pi/faces)])
        A = np.asarray(rows) @ R.T
        return A, np.full(len(A), -np.inf), np.zeros(len(A))


def force_to_wrench(point_world, origin_world):
    """Map a point force into world [F, M_about_origin]."""
    x, y, z = np.asarray(point_world) - np.asarray(origin_world)
    return np.vstack([np.eye(3), [[0, -z, y], [z, 0, -x], [-y, x, 0]]])


def generalized_force_map(model, data, contacts, reduction=None):
    import mujoco
    columns, points = [], []
    for contact in contacts:
        bid = contact.body_id
        point = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ contact.local_point
        J = np.zeros((3, model.nv)); Jr = np.zeros_like(J)
        mujoco.mj_jac(model, data, J, Jr, point, bid)
        columns.append(J.T); points.append(point)
    A = np.concatenate(columns, axis=1) if columns else np.empty((model.nv, 0))
    return (reduction.T @ A if reduction is not None else A), np.asarray(points).reshape(-1, 3)


def balance_residual(mass_kg, acceleration, gravity, forces):
    return mass_kg*(np.asarray(acceleration)-np.asarray(gravity))-np.asarray(forces).reshape(-1, 3).sum(0)


def verify_contracts():
    """Manufactured mechanics checks, not simulated pull-ups or jumps."""
    ground = Contact('ground', 0, np.zeros(3))
    A, lo, hi = ground.inequalities()
    # Old box admitted (.8,.8,1), exceeding a circular cone by sqrt(2).
    assert np.max(A @ [.8, .8, 1]) > .1
    phi = np.linspace(0, 2*np.pi, 10001)
    radii = .8*np.cos(np.pi/16)/np.max(np.cos(phi[:, None]-np.arange(16)*2*np.pi/16), axis=1)
    assert radii.max() <= .8 + 1e-10
    assert np.max(A @ [0, 0, -1]) > 0
    mass = 90.; gravity = np.array([0, 0, -9.80665])
    assert np.linalg.norm(balance_residual(mass, np.zeros(3), gravity, [[0,0,mass*9.80665]])) < 1e-10
    # Same weight held by two ideal hand constraints; no foot reaction.
    assert np.linalg.norm(balance_residual(mass, np.zeros(3), gravity, [[0,0,mass*9.80665/2]]*2)) < 1e-10
    assert np.linalg.norm(balance_residual(mass, gravity, gravity, [])) < 1e-10
    inactive = Contact('foot in flight', 0, np.zeros(3), active=False)
    Ai, li, ui = inactive.inequalities(); assert np.all(Ai == np.eye(3)) and np.all(li == ui)
    grasp = Contact('ideal hand grasp', 0, np.zeros(3), kind='bilateral_point')
    assert grasp.inequalities()[0].shape == (0,3)
    f=np.array([3.,4.,5.]); p=np.array([.3,.2,.1]); o=np.array([-.1,.2,.5])
    assert np.allclose(force_to_wrench(p,o)@f, np.r_[f,np.cross(p-o,f)])
    # Landing impulse identity: constant support chosen to stop downward COM.
    T=.2; incoming=-3.; support=mass*(0-incoming)/T+mass*9.80665
    assert abs((support-mass*9.80665)*T-mass*(0-incoming)) < 1e-10
    return {'status':'passed','checks':['circular friction bound','unilateral push only','standing force balance','ideal two-hand hanging force balance','ballistic flight','inactive contact zero force','point-wrench transform','landing impulse identity'],
            'scope':'Manufactured contact/algebra checks only. No human pull-up, muscle-up, jump or landing trajectory has been solved.'}
