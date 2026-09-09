"""Relativistic 4-vector and 4-tensor algebra in metric signature (+,-,-,-)."""
import numpy as np


def minkowski_dot(a, b):
    r"""Compute 4-vector Minkowski inner product a^\mu b_\mu = a^0 b^0 - a^1 b^1 - a^2 b^2 - a^3 b^3.

    Accepts arrays of 4-vectors with final dimension of size 4.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError('Input 4-vectors must have final dimension of size 4')
    return a[..., 0] * b[..., 0] - np.sum(a[..., 1:] * b[..., 1:], axis=-1)


def minkowski_norm_sq(a):
    r"""Compute Minkowski squared norm a \cdot a = (a^0)^2 - |a|^2."""
    return minkowski_dot(a, a)


def lower_index(a):
    r"""Lower index using metric g_{\mu\nu} = diag(1, -1, -1, -1): a_\mu = (a^0, -a^1, -a^2, -a^3)."""
    a = np.asarray(a, dtype=float)
    if a.shape[-1] != 4:
        raise ValueError('Input 4-vector must have final dimension of size 4')
    a_lower = np.empty_like(a)
    a_lower[..., 0] = a[..., 0]
    a_lower[..., 1:] = -a[..., 1:]
    return a_lower


def raise_index(a_lower):
    r"""Raise index using metric g^{\mu\nu} = diag(1, -1, -1, -1): a^\mu = (a_0, -a_1, -a_2, -a_3)."""
    return lower_index(a_lower)


def wedge_4d(a, b):
    r"""Compute anti-symmetric outer product (a \wedge b)^{\mu\nu} = a^\mu b^\nu - a^\nu b^\mu.

    Returns array of shape (..., 4, 4).
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape[-1] != 4 or b.shape[-1] != 4:
        raise ValueError('Input 4-vectors must have final dimension of size 4')
    return a[..., :, None] * b[..., None, :] - a[..., None, :] * b[..., :, None]


def four_position(ct, r):
    r"""Construct 4-position x^\mu = (ct, x, y, z)."""
    ct = np.asarray(ct, dtype=float)
    r = np.asarray(r, dtype=float)
    if r.shape[-1] != 3:
        raise ValueError('Spatial position r must have final dimension of size 3')
    ct_expanded = np.expand_dims(ct, axis=-1)
    return np.concatenate([ct_expanded, r], axis=-1)


def four_momentum(p_spatial, m, c):
    r"""Construct 4-momentum p^\mu = (p^0, p) on the mass shell p^0 = sqrt(m^2 c^2 + |p|^2)."""
    p_spatial = np.asarray(p_spatial, dtype=float)
    if p_spatial.shape[-1] != 3:
        raise ValueError('p_spatial must have final dimension of size 3')
    m, c = float(m), float(c)
    if m <= 0 or c <= 0:
        raise ValueError('m and c must be positive')
    p_sq = np.sum(np.square(p_spatial), axis=-1)
    p0 = np.sqrt(m * m * c * c + p_sq)
    p0_expanded = np.expand_dims(p0, axis=-1)
    return np.concatenate([p0_expanded, p_spatial], axis=-1)


def four_velocity(p, m):
    r"""Construct 4-velocity u^\mu = p^\mu / m = (\gamma c, \gamma v)."""
    p = np.asarray(p, dtype=float)
    if p.shape[-1] != 4:
        raise ValueError('p must have final dimension of size 4')
    m = float(m)
    if m <= 0:
        raise ValueError('m must be positive')
    return p / m


def four_acceleration(F, u, q, m):
    r"""Compute 4-acceleration w^\mu = du^\mu/d\tau = (q/m^2) F^{\mu\nu} u_\nu."""
    F, u = np.asarray(F, dtype=float), np.asarray(u, dtype=float)
    if F.shape[-2:] != (4, 4) or u.shape[-1] != 4:
        raise ValueError('F must have final shape (4,4) and u must have final dimension 4')
    q, m = float(q), float(m)
    if m <= 0:
        raise ValueError('m must be positive')
    u_lower = lower_index(u)
    fu = np.einsum('...ij,...j->...i', F, u_lower)
    return (q / (m * m)) * fu
