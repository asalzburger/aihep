"""One test case per particle type: what the detector sees, and why.

The whole point of the calorimeter/muon extension is that the three
interaction classes stop in three different places, so each species gets its
own signature. These tests pin those signatures.
"""

import math

import numpy as np
import pandas as pd
import pytest
from detector2d.calorimeter import build_calo_stack
from detector2d.field import FieldRegion, FieldRegions
from detector2d.geometry import CircleLayer
from detector2d.polygon import build_muon_system

from tracksim2d.config import FieldConfig, ParticleGunConfig, SimConfig
from tracksim2d.edm import DEPOSITS_COLUMNS, PARTICLES_COLUMNS
from tracksim2d.response import ResponseConfig, ShowerProfile, sum_cells
from tracksim2d.simulate import path_for_row, propagate_particles, simulate_events
from tracksim2d.species import SPECIES

TRACKER_OUTER = 210.0
ECAL_INNER, ECAL_THICKNESS, ECAL_N_PHI = 210.0, 30.0, 256
HCAL_INNER, HCAL_THICKNESS, HCAL_N_PHI = 300.0, 80.0, 64
WORLD_RADIUS = 800.0
MIP_ECAL, MIP_HCAL = 0.3, 0.8
EM_FRACTIONS = (0.60, 0.28, 0.12)
HADRON_FRACTIONS = (0.65, 0.35)


def _layers():
    """A compact stand-in for configs/full_detector.yaml: 4 bare tracker
    circles, a 3-layer staggered ECAL, a 2-layer HCAL, and 3 octagonal muon
    triplet stations."""
    tracker = [
        CircleLayer(layer_id=i, center=(0.0, 0.0), radius=r, system="tracker")
        for i, r in enumerate((50.0, 100.0, 150.0, 200.0))
    ]
    ecal = build_calo_stack(
        100, ECAL_INNER, 3, ECAL_THICKNESS, ECAL_N_PHI, system="ecal", phi_stagger=[0.0, 0.5, 0.0]
    )
    hcal = build_calo_stack(200, HCAL_INNER, 2, HCAL_THICKNESS, HCAL_N_PHI, system="hcal")
    muon = build_muon_system(300, apothem_inner=520.0, station_spacing=100.0, n_stations=3, n_planes=3)
    return tracker + ecal + hcal + muon


def _field():
    """Strong in the tracker, none in the calorimeters, half and reversed in
    the muon system."""
    return FieldRegions(
        regions=(
            FieldRegion(r_max=TRACKER_OUTER, bz=2.0),
            FieldRegion(r_max=480.0, bz=0.0),
            FieldRegion(r_max=None, bz=-1.0),
        ),
        k=0.2998,
    )


def _response():
    """No stochastic smearing, so energies come out exact and assertions can
    be tight."""
    return ResponseConfig(
        em=ShowerProfile(EM_FRACTIONS, sigma_cells=1.5, sigma_growth=0.5),
        hadron=ShowerProfile(HADRON_FRACTIONS, sigma_cells=1.0, sigma_growth=0.4),
        mip_energy={"ecal": MIP_ECAL, "hcal": MIP_HCAL},
        stochastic={},
    )


def run_particle(species_name: str, energy: float = 200.0, phi0: float = 0.3):
    """Shoot one particle of the given species through the detector."""
    species = SPECIES[species_name]
    particle = dict(
        event_id=0,
        particle_id=0,
        species=species.name,
        pdg=species.pdg,
        x0=0.0,
        y0=0.0,
        phi0=phi0,
        charge=species.charge,
        energy=energy,
        radius=math.nan,
    )
    particles = pd.DataFrame([particle], columns=PARTICLES_COLUMNS)
    hits, deposits = propagate_particles(
        particles,
        _layers(),
        field=_field(),
        world_radius=WORLD_RADIUS,
        max_path_length=4000.0,
        response_config=_response(),
    )
    return particles, hits, deposits


def energy_in(deposits, system) -> float:
    return float(deposits[deposits["system"] == system]["energy"].sum())


def layer_energies(deposits, system) -> list[float]:
    subset = deposits[deposits["system"] == system]
    return [float(g["energy"].sum()) for _, g in subset.groupby("layer_id", sort=True)]


def phi_rms(deposits) -> float:
    """Energy-weighted RMS spread in azimuth -- the shower's lateral width."""
    phi = np.arctan2(deposits["y"], deposits["x"])
    weights = deposits["energy"]
    mean = np.average(phi, weights=weights)
    return float(math.sqrt(np.average((phi - mean) ** 2, weights=weights)))


# --- EM particles: electron, positron, photon, pi0 ---------------------------


@pytest.mark.parametrize("name", ["electron", "positron", "photon", "pi0"])
def test_em_particles_shower_in_the_ecal_and_stop_there(name):
    _particles, hits, deposits = run_particle(name)

    assert set(deposits["system"]) == {"ecal"}
    assert len(layer_energies(deposits, "ecal")) == 3
    # fully contained: the ECAL measures the particle's whole energy
    assert energy_in(deposits, "ecal") == pytest.approx(200.0)
    # absorbed, so nothing beyond -- this is why an electron is not a muon
    assert "muon" not in set(hits["system"])
    assert energy_in(deposits, "hcal") == 0.0


@pytest.mark.parametrize("name", ["electron", "positron", "photon", "pi0"])
def test_em_shower_deposits_most_in_the_first_layer_then_decreasing(name):
    _particles, _hits, deposits = run_particle(name)
    per_layer = layer_energies(deposits, "ecal")
    assert per_layer[0] > per_layer[1] > per_layer[2]
    assert [e / sum(per_layer) for e in per_layer] == pytest.approx(list(EM_FRACTIONS))


@pytest.mark.parametrize("name", ["electron", "positron", "photon", "pi0"])
def test_em_shower_widens_with_depth(name):
    _particles, _hits, deposits = run_particle(name)
    widths = [
        phi_rms(deposits[deposits["layer_id"] == layer_id]) for layer_id in (100, 101, 102)
    ]
    assert widths[0] < widths[1] < widths[2]


@pytest.mark.parametrize("name,charged", [("electron", True), ("positron", True),
                                          ("photon", False), ("pi0", False)])
def test_only_charged_em_particles_leave_tracker_hits(name, charged):
    """A neutral particle does not ionize: a photon crosses the silicon
    invisibly and is seen for the first time when it showers."""
    _particles, hits, _deposits = run_particle(name)
    tracker_hits = hits[hits["system"] == "tracker"]
    assert (len(tracker_hits) > 0) is charged


def test_electron_and_positron_bend_in_opposite_directions():
    """Same shower, mirrored track -- the only thing charge changes upstream
    of the calorimeter."""
    _p, electron_hits, electron_deposits = run_particle("electron")
    _p, positron_hits, positron_deposits = run_particle("positron")

    def drift(hits):
        outermost = hits.sort_values("path_length").iloc[-1]
        return math.remainder(math.atan2(outermost["y"], outermost["x"]) - 0.3, 2 * math.pi)

    assert drift(electron_hits) * drift(positron_hits) < 0
    # identical energy response regardless of the sign of the charge
    assert energy_in(electron_deposits, "ecal") == pytest.approx(energy_in(positron_deposits, "ecal"))


def test_a_neutral_em_particle_deposits_exactly_where_it_was_aimed():
    """A photon is straight, so its shower centroid must sit on phi0 -- a
    check that the impact point comes from the real intersection."""
    _particles, _hits, deposits = run_particle("photon", phi0=0.3)
    first_layer = deposits[deposits["layer_id"] == 100]
    centroid = np.average(np.arctan2(first_layer["y"], first_layer["x"]), weights=first_layer["energy"])
    assert centroid == pytest.approx(0.3, abs=first_layer.iloc[0]["s_local"] * 0 + 1e-3)


# --- hadrons: pi+, pi-, neutron ---------------------------------------------


@pytest.mark.parametrize("name", ["pi+", "pi-", "neutron"])
def test_hadrons_punch_through_the_ecal_and_shower_in_the_hcal(name):
    _particles, hits, deposits = run_particle(name)
    assert len(layer_energies(deposits, "hcal")) == 2
    assert energy_in(deposits, "hcal") == pytest.approx(200.0)
    assert "muon" not in set(hits["system"])  # absorbed in the HCAL


@pytest.mark.parametrize("name", ["pi+", "pi-", "neutron"])
def test_hadronic_shower_deposits_most_in_the_first_hcal_layer(name):
    _particles, _hits, deposits = run_particle(name)
    per_layer = layer_energies(deposits, "hcal")
    assert per_layer[0] > per_layer[1]
    assert [e / sum(per_layer) for e in per_layer] == pytest.approx(list(HADRON_FRACTIONS))


@pytest.mark.parametrize("name,charged", [("pi+", True), ("pi-", True), ("neutron", False)])
def test_only_a_charged_hadron_leaves_a_mip_trail_in_the_ecal(name, charged):
    """A charged hadron ionizes its way through the ECAL -- a fixed, tiny
    amount per layer that does not scale with its energy. A neutron leaves
    nothing there at all."""
    _particles, hits, deposits = run_particle(name, energy=200.0)
    ecal_energy = energy_in(deposits, "ecal")
    if charged:
        assert ecal_energy == pytest.approx(3 * MIP_ECAL)
        assert ecal_energy < 0.01 * 200.0  # unmistakably not a shower
        assert len(hits[hits["system"] == "tracker"]) > 0
    else:
        assert ecal_energy == 0.0
        assert len(hits[hits["system"] == "tracker"]) == 0


def test_a_charged_hadrons_ecal_trail_is_one_cell_per_layer():
    _particles, _hits, deposits = run_particle("pi+")
    ecal = deposits[deposits["system"] == "ecal"]
    assert len(ecal) == 3  # one cell in each of the 3 layers
    assert set(ecal["energy"]) == {MIP_ECAL}


def test_a_charged_hadrons_mip_trail_does_not_scale_with_its_energy():
    """Ionization is a fixed toll per layer; only the shower scales. (Both
    energies are kept above ~65, the point below which R = pt/(k*q*B) is too
    small for the track to escape the tracker at all.)"""
    _p, _h, low = run_particle("pi+", energy=150.0)
    _p, _h, high = run_particle("pi+", energy=300.0)
    assert energy_in(low, "ecal") == pytest.approx(energy_in(high, "ecal")) == pytest.approx(3 * MIP_ECAL)
    assert energy_in(high, "hcal") == pytest.approx(2 * energy_in(low, "hcal"))


def test_a_hadronic_shower_is_broader_than_an_em_one():
    _p, _h, em = run_particle("photon")
    _p, _h, hadronic = run_particle("neutron")
    assert phi_rms(hadronic[hadronic["layer_id"] == 200]) > 2 * phi_rms(em[em["layer_id"] == 100])


# --- muons -------------------------------------------------------------------


@pytest.mark.parametrize("name", ["mu+", "mu-"])
def test_only_muons_reach_the_muon_system(name):
    """3 stations x 3 planes = 9 hits: the triplet structure the muon system
    is built for."""
    _particles, hits, _deposits = run_particle(name)
    muon_hits = hits[hits["system"] == "muon"]
    assert len(muon_hits) == 9
    assert sorted(muon_hits["layer_id"]) == [300, 301, 302, 310, 311, 312, 320, 321, 322]
    assert len(hits[hits["system"] == "tracker"]) > 0


@pytest.mark.parametrize("name", ["mu+", "mu-"])
def test_a_muon_leaves_only_a_mip_trail_in_both_calorimeters(name):
    """The signature that separates a muon from everything else: it crosses
    both calorimeters without showering."""
    _particles, _hits, deposits = run_particle(name, energy=200.0)
    assert energy_in(deposits, "ecal") == pytest.approx(3 * MIP_ECAL)
    assert energy_in(deposits, "hcal") == pytest.approx(2 * MIP_HCAL)
    assert float(deposits["energy"].sum()) < 0.02 * 200.0
    assert len(deposits) == 5  # one cell per sampling layer, nothing more


@pytest.mark.parametrize("name", ["mu+", "mu-"])
def test_a_muon_bends_the_other_way_in_the_muon_system(name):
    """The flux return reverses the field outside the solenoid, so the track
    curls back the opposite way -- and at half the field, so twice as gently."""
    particles, _hits, _deposits = run_particle(name)
    path = path_for_row(particles.iloc[0], _field(), WORLD_RADIUS, 4000.0)
    tracker, calo, muon = path.segments[0], path.segments[1], path.segments[2]

    assert calo.trajectory.is_straight  # no field in the calorimeters
    assert tracker.trajectory.radius * muon.trajectory.radius < 0
    assert abs(muon.trajectory.radius) == pytest.approx(2 * abs(tracker.trajectory.radius))

    def turn(segment):
        span = min(segment.length, 50.0)
        return math.remainder(
            segment.trajectory.direction_at(span) - segment.trajectory.direction_at(0.0), 2 * math.pi
        )

    assert turn(tracker) * turn(muon) < 0


def test_mu_plus_and_mu_minus_curl_opposite_ways():
    plus, _h, _d = run_particle("mu+")
    minus, _h, _d = run_particle("mu-")
    r_plus = path_for_row(plus.iloc[0], _field(), WORLD_RADIUS, 4000.0).segments[0].trajectory.radius
    r_minus = path_for_row(minus.iloc[0], _field(), WORLD_RADIUS, 4000.0).segments[0].trajectory.radius
    assert r_plus * r_minus < 0


# --- the ECAL stagger, at simulation level -----------------------------------


def test_the_staggered_ecal_layer_samples_a_shower_on_a_half_cell_offset_grid():
    """The physics reason layer 1 is shifted: its cells are offset by half a
    cell from layers 0 and 2, so a shower is sampled on two interleaved grids
    rather than one -- and a shower landing on a cell boundary in layers 0/2
    lands mid-cell in layer 1."""
    layers = _layers()
    rings = {ring.layer_id: ring for ring in layers if ring.layer_id in (100, 101, 102)}

    # aim exactly at a layer-0 cell boundary: the worst case for layers 0 and 2
    boundary_phi = rings[100].cell_edges(20)[1]
    _particles, _hits, deposits = run_particle("photon", phi0=boundary_phi)

    def peak_cell(layer_id):
        subset = deposits[deposits["layer_id"] == layer_id]
        return int(subset.loc[subset["energy"].idxmax()]["cell_id"])

    # layers 0 and 2 share a grid, so their peak cells sit at the same azimuth
    assert rings[100].cell_center_phi(peak_cell(100)) == pytest.approx(
        rings[102].cell_center_phi(peak_cell(102))
    )
    # layer 1's grid is offset by exactly half a cell from theirs
    offset = rings[101].cell_center_phi(peak_cell(101)) - rings[100].cell_center_phi(peak_cell(100))
    assert abs(math.remainder(offset, 2 * math.pi)) == pytest.approx(0.5 * rings[100].dphi)

    # ...and that is what buys the measurement something. Aimed at a cell
    # boundary, layers 0 and 2 split the shower *exactly* evenly between two
    # cells -- they cannot say which side it landed on. The same shower sits
    # inside a single layer-1 cell, which resolves it. (Comparing peak
    # *fractions* between layers would prove nothing here: layer 1's shower is
    # also wider, which lowers its peak share independently of the stagger.)
    def top_two_ratio(layer_id):
        energies = sorted(deposits[deposits["layer_id"] == layer_id]["energy"], reverse=True)
        return energies[1] / energies[0]

    assert top_two_ratio(100) == pytest.approx(1.0, abs=1e-9)
    assert top_two_ratio(102) == pytest.approx(1.0, abs=1e-9)
    assert top_two_ratio(101) < 0.95


# --- cross-cutting -----------------------------------------------------------


@pytest.mark.parametrize("name", sorted(SPECIES))
def test_every_species_produces_a_well_formed_deposits_table(name):
    _particles, _hits, deposits = run_particle(name)
    assert list(deposits.columns) == DEPOSITS_COLUMNS
    assert (deposits["energy"] > 0).all()
    assert set(deposits["system"]) <= {"ecal", "hcal"}
    for _, row in deposits.iterrows():
        n_phi = ECAL_N_PHI if row["system"] == "ecal" else HCAL_N_PHI
        assert 0 <= row["cell_id"] < n_phi
        # the cell's recorded position really is on its own ring
        expected = ECAL_INNER if row["system"] == "ecal" else HCAL_INNER
        assert math.hypot(row["x"], row["y"]) > expected


def test_a_low_pt_particle_curls_up_inside_the_tracker_and_never_reaches_the_calorimeter():
    """R = pt / (k*q*B) = 20/(0.2998*2) ~ 33, so the track's whole circle fits
    well inside the tracker -- physically real, and a case downstream code
    has to cope with."""
    _particles, hits, deposits = run_particle("pi+", energy=20.0)
    assert len(deposits) == 0
    assert set(hits["system"]) <= {"tracker"}


def test_deposits_are_truth_level_and_sum_cells_gives_the_reconstructed_view():
    """Two particles aimed at the same place contribute separate rows, which
    is what makes the table usable as ground truth; sum_cells collapses them
    into what a real readout would report."""
    species = SPECIES["photon"]
    particles = pd.DataFrame(
        [
            dict(event_id=0, particle_id=i, species=species.name, pdg=species.pdg,
                 x0=0.0, y0=0.0, phi0=0.3, charge=0.0, energy=100.0, radius=math.nan)
            for i in range(2)
        ],
        columns=PARTICLES_COLUMNS,
    )
    _hits, deposits = propagate_particles(
        particles, _layers(), field=_field(), world_radius=WORLD_RADIUS,
        max_path_length=4000.0, response_config=_response(),
    )
    assert set(deposits["particle_id"]) == {0, 1}

    summed = sum_cells(deposits)
    assert "particle_id" not in summed.columns
    assert len(summed) == len(deposits) // 2  # both particles hit the same cells
    assert float(summed["energy"].sum()) == pytest.approx(float(deposits["energy"].sum()))
    assert float(summed["energy"].sum()) == pytest.approx(200.0)


def test_a_species_less_particle_still_tracks_but_has_no_calorimeter_response():
    """Backwards compatibility: a hand-built kinematic table (as
    tracking/denby produces) has no species column, so it is a bare charged
    stub -- it bends and leaves hits, but the calorimeter has nothing to
    model."""
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.3, charge=1.0, radius=math.nan)],
        columns=PARTICLES_COLUMNS,
    )
    hits, deposits = propagate_particles(
        particles, _layers(), response_config=_response(), world_radius=WORLD_RADIUS
    )
    assert len(deposits) == 0
    assert len(hits[hits["system"] == "tracker"]) == 4


def test_stochastic_smearing_perturbs_energies_without_changing_the_structure():
    """Turned off everywhere else in these tests so assertions can be exact;
    here we check it actually does something, and stays physical."""
    config = _response()
    config.stochastic = {"ecal": 0.3}
    species = SPECIES["photon"]
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, species=species.name, pdg=species.pdg, x0=0.0, y0=0.0,
              phi0=0.3, charge=0.0, energy=200.0, radius=math.nan)],
        columns=PARTICLES_COLUMNS,
    )
    _hits, deposits = propagate_particles(
        particles, _layers(), field=_field(), world_radius=WORLD_RADIUS,
        max_path_length=4000.0, response_config=config, rng=np.random.default_rng(3),
    )
    total = float(deposits["energy"].sum())
    assert total != pytest.approx(200.0)  # smeared
    assert 150.0 < total < 260.0  # but not wildly
    assert (deposits["energy"] > 0).all()


def test_simulate_events_with_a_mixed_species_gun_is_reproducible():
    config = SimConfig(
        layers=_layers(),
        magnetic_field=FieldConfig(bz=2.0, k=0.2998),
        field_regions=_field(),
        gun=ParticleGunConfig(
            n_particles=6,
            phi_min=-3.14,
            phi_max=3.14,
            species=("electron", "photon", "pi+", "neutron", "mu+", "mu-"),
            pt_min=150.0,
            pt_max=300.0,
        ),
        response=_response(),
        n_events=2,
        seed=11,
        world_radius=WORLD_RADIUS,
        max_path_length=4000.0,
    )
    particles_a, hits_a, deposits_a = simulate_events(config)
    particles_b, hits_b, deposits_b = simulate_events(config)

    pd.testing.assert_frame_equal(particles_a, particles_b)
    pd.testing.assert_frame_equal(hits_a, hits_b)
    pd.testing.assert_frame_equal(deposits_a, deposits_b)

    assert len(particles_a) == 12
    assert set(particles_a["species"]) <= set(SPECIES)
    assert (particles_a["charge"] == [SPECIES[s].charge for s in particles_a["species"]]).all()
    assert len(deposits_a) > 0


def test_species_weights_can_bias_the_gun():
    config = SimConfig(
        layers=[],
        magnetic_field=FieldConfig(bz=2.0),
        gun=ParticleGunConfig(
            n_particles=200, species=("mu+", "electron"), species_weights=(9.0, 1.0)
        ),
        n_events=1,
        seed=5,
    )
    particles, _hits, _deposits = simulate_events(config)
    fraction_muons = (particles["species"] == "mu+").mean()
    assert 0.8 < fraction_muons < 0.98


def test_mismatched_species_weights_are_rejected():
    gun = ParticleGunConfig(species=("mu+", "electron"), species_weights=(1.0,))
    with pytest.raises(ValueError, match="line up"):
        _ = gun.species_probabilities
