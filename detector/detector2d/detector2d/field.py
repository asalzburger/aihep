"""Turn a physical transverse momentum + charge + field into a signed bend radius.

This is the only place physical units (GeV, Tesla, meters) enter the
package; everything else (:mod:`detector2d.geometry`,
:mod:`detector2d.intersect`) works in an arbitrary length unit and a signed
``radius`` directly. Callers who already know the radius they want (e.g. a
radius fit straight out of a picture, in pixels) can skip this module
entirely and construct a :class:`~detector2d.geometry.Trajectory` directly.
"""

from __future__ import annotations

import math

#: Standard R[m] = pt[GeV] / (k * |q| * B[T]) constant (k = 0.3 * c-derived).
DEFAULT_K = 0.2998


def signed_radius(pt: float, charge: float, bz: float, k: float = DEFAULT_K) -> float:
    """Signed bend radius of a particle with transverse momentum ``pt`` (>=0),
    charge ``charge`` (in units of e, sign included), in a field ``bz`` out of
    the 2D plane. Returns ``math.inf`` for a neutral particle or zero field
    (straight track) -- pass the result straight into ``Trajectory(radius=...)``.
    """
    if charge == 0 or bz == 0:
        return math.inf
    return pt / (k * charge * bz)
