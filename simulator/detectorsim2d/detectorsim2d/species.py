"""Particle species: what a particle *is*, and therefore how the detector
sees it.

Two properties drive everything downstream:

- ``charge`` decides whether the particle bends in a magnetic field and
  whether it leaves position hits in a tracking layer at all (a neutral
  particle passes through silicon without ionizing it).
- ``interaction`` decides where it stops and what it deposits:

  ==========  =====================================================
  ``em``      showers in the EM calorimeter and is absorbed there
  ``hadron``  punches through the ECAL (a minimum-ionizing trickle if
              charged) and showers in the hadronic calorimeter
  ``muon``    ionizes minimally all the way through both calorimeters
              and is the only thing that reaches the muon system
  ==========  =====================================================

That table *is* the physics content of this toy: the reason a detector can
tell an electron from a pion from a muon at all is that the three stop in
three different places.

``pi0`` is deliberately treated as a single neutral EM object rather than its
two decay photons -- the two-photon opening angle is a 3D effect this 2D toy
has nothing useful to say about, and for calorimeter purposes it deposits the
same way a photon does.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Interaction classes -- see the module docstring.
EM = "em"
HADRON = "hadron"
MUON = "muon"


@dataclass(frozen=True)
class Species:
    name: str
    pdg: int
    charge: float
    interaction: str

    @property
    def is_charged(self) -> bool:
        return self.charge != 0.0


SPECIES: dict[str, Species] = {
    species.name: species
    for species in (
        Species("electron", 11, -1.0, EM),
        Species("positron", -11, +1.0, EM),
        Species("photon", 22, 0.0, EM),
        Species("pi0", 111, 0.0, EM),
        Species("pi+", 211, +1.0, HADRON),
        Species("pi-", -211, -1.0, HADRON),
        Species("neutron", 2112, 0.0, HADRON),
        Species("mu-", 13, -1.0, MUON),
        Species("mu+", -13, +1.0, MUON),
    )
}

#: Every species this package knows, in a stable order.
SPECIES_NAMES: tuple[str, ...] = tuple(SPECIES)

_BY_PDG: dict[int, Species] = {species.pdg: species for species in SPECIES.values()}


def get(name: str) -> Species:
    """Look a species up by name, with a listing of the valid ones on error."""
    try:
        return SPECIES[name]
    except KeyError:
        raise KeyError(f"unknown species {name!r}; known: {', '.join(SPECIES_NAMES)}") from None


def from_pdg(pdg: int) -> Species:
    """Look a species up by PDG code."""
    try:
        return _BY_PDG[int(pdg)]
    except KeyError:
        raise KeyError(f"unknown PDG code {pdg!r}; known: {sorted(_BY_PDG)}") from None


def for_row(row) -> Species | None:
    """The :class:`Species` of a ``particles`` table row, or ``None`` if the
    row predates species (e.g. a hand-built table from ``tracking/denby``,
    which only knows ``charge`` and ``radius``). Callers treat ``None`` as
    "a bare charged stub": it bends and leaves hits, but has no calorimeter
    response."""
    name = row.get("species") if hasattr(row, "get") else None
    if name is None or (isinstance(name, float) and name != name):  # NaN
        return None
    return get(str(name))
