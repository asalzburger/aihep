import math

import numpy as np
import pandas as pd
import pytest
from detector2d.barrel import build_barrel_modules, module_reach
from detector2d.geometry import CircleLayer, LineLayer, Trajectory

from detectorsim2d.config import FieldConfig, ParticleGunConfig, SimConfig
from detectorsim2d.edm import PARTICLES_COLUMNS
from detectorsim2d.simulate import boundary_crossing_s, hits_for_particles, sample_particles, simulate_events
from detectorsim2d.species import get as get_species


def _particle_row(**overrides):
    row = dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan)
    row.update(overrides)
    return row


def test_straight_particle_hits_expected_line_layers():
    particles = pd.DataFrame([_particle_row()], columns=PARTICLES_COLUMNS)
    layers = [
        LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0)),
        LineLayer(layer_id=1, p1=(20.0, -5.0), p2=(20.0, 5.0)),
    ]
    hits = hits_for_particles(particles, layers)
    assert len(hits) == 2
    assert list(hits.sort_values("path_length")["layer_id"]) == [0, 1]
    assert list(hits.sort_values("path_length")["x"]) == pytest.approx([10.0, 20.0])


def test_curved_particle_hits_circle_layer():
    # trajectory circle: center (0,0) r=5 (particle starts at (0,-5) heading +x).
    # layer: circle center (6,0) r=5 -- classic 3-4-5 intersection at (3, +-4).
    particles = pd.DataFrame([_particle_row(x0=0.0, y0=-5.0, phi0=0.0, radius=5.0)], columns=PARTICLES_COLUMNS)
    layers = [CircleLayer(layer_id=0, center=(6.0, 0.0), radius=5.0)]
    hits = hits_for_particles(particles, layers)
    assert len(hits) == 1  # only the earliest crossing is kept
    x, y = hits.iloc[0]["x"], hits.iloc[0]["y"]
    assert (round(x, 6), round(y, 6)) in {(3.0, -4.0), (3.0, 4.0)}
    assert math.hypot(x - 6.0, y - 0.0) == pytest.approx(5.0)


def test_particle_with_nan_radius_is_straight():
    particles = pd.DataFrame([_particle_row(radius=math.nan)], columns=PARTICLES_COLUMNS)
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0))]
    hits = hits_for_particles(particles, layers)
    assert len(hits) == 1
    assert hits.iloc[0]["x"] == pytest.approx(10.0)
    assert hits.iloc[0]["y"] == pytest.approx(0.0)


def test_detailed_barrel_overlap_produces_a_genuine_double_hit():
    """A straight particle aimed into the angular overlap between two
    neighboring barrel modules crosses both -- hits_for_particles needs no
    special-casing for this: it just loops over every module object (both
    happen to share layer_id=0) and keeps whichever ones the trajectory
    actually crosses."""
    radius, half_length, tilt_deg, overlap_fraction = 68.0, 4.0, 10.0, 0.15
    tilt = math.radians(tilt_deg)
    modules = build_barrel_modules(
        layer_id=0, radius=radius, half_length=half_length, tilt=tilt, overlap_fraction=overlap_fraction
    )
    n = len(modules)
    delta = 2.0 * math.pi / n
    reach = module_reach(radius, 2 * half_length, tilt)
    phi_in_overlap = (delta + reach) / 2.0  # inside module 0's overlap with module 1

    particles = pd.DataFrame([_particle_row(phi0=phi_in_overlap)], columns=PARTICLES_COLUMNS)
    hits = hits_for_particles(particles, modules)

    assert len(hits) == 2
    assert set(hits["layer_id"]) == {0}
    # both hits belong to the same physical crossing, not two unrelated ones
    assert (hits["path_length"].max() - hits["path_length"].min()) < 2 * half_length


def test_boundary_crossing_s_is_none_without_a_boundary():
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=None)
    assert boundary_crossing_s(trajectory, None) is None


def test_boundary_crossing_s_finds_straight_track_crossing():
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=None)
    assert boundary_crossing_s(trajectory, 25.0) == pytest.approx(25.0)


def test_boundary_crossing_s_none_when_trajectory_never_reaches_it():
    # a tight loop of radius 5 (max reach 2*5=10 from the origin) never
    # reaches a boundary of radius 50
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=5.0)
    assert boundary_crossing_s(trajectory, 50.0) is None


def test_hits_for_particles_drops_the_loop_back_crossing_beyond_the_boundary():
    # a curved trajectory starting at the origin always crosses a concentric
    # circular layer twice (see detector2d.barrel's module docstring): once
    # on the way out, and again after looping back through on the way in.
    # radius=10 -> max reach 20, so a layer at radius=18 is only reachable
    # via that second, loop-back crossing (its "first" and only crossing
    # happens late, close to the half-loop point) -- with no boundary, that
    # crossing is still kept; with a boundary tighter than it, it must be
    # dropped instead of drawing the particle back in from outside the
    # tracker volume.
    particles = pd.DataFrame([_particle_row(radius=10.0)], columns=PARTICLES_COLUMNS)
    layers = [CircleLayer(layer_id=0, center=(0.0, 0.0), radius=18.0)]

    hits_unbounded = hits_for_particles(particles, layers, tracker_boundary=None)
    assert len(hits_unbounded) == 1

    hits_bounded = hits_for_particles(particles, layers, tracker_boundary=15.0)
    assert len(hits_bounded) == 0


def test_hits_for_particles_keeps_hits_within_the_boundary():
    particles = pd.DataFrame([_particle_row()], columns=PARTICLES_COLUMNS)  # straight, along +x
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0))]
    hits = hits_for_particles(particles, layers, tracker_boundary=50.0)
    assert len(hits) == 1
    assert hits.iloc[0]["x"] == pytest.approx(10.0)


def test_simulate_events_passes_tracker_boundary_through_config():
    # pt=1.0, k=0.2998 (default), bz chosen so the resolved radius is
    # exactly 10.0 -- same loop-back setup as the test above, but exercised
    # through the full simulate_events(config) path rather than calling
    # hits_for_particles directly.
    k = FieldConfig().k
    bz = 1.0 / (k * 10.0)
    layers = [CircleLayer(layer_id=0, center=(0.0, 0.0), radius=18.0)]
    config = SimConfig(
        layers=layers,
        magnetic_field=FieldConfig(bz=bz, k=k),
        gun=ParticleGunConfig(n_particles=1, phi_min=0.0, phi_max=0.0, charges=(1.0,), pt_min=1.0, pt_max=1.0),
        n_events=1,
        seed=1,
        tracker_boundary=15.0,
    )
    particles, hits, _deposits = simulate_events(config)
    assert particles.iloc[0]["radius"] == pytest.approx(10.0)
    assert len(hits) == 0  # the only crossing is the loop-back one, dropped by the boundary


def test_sample_particles_reproducible_with_same_seed():
    config = SimConfig(
        layers=[],
        magnetic_field=FieldConfig(bz=1.0),
        gun=ParticleGunConfig(n_particles=4, pt_min=1.0, pt_max=5.0),
        n_events=1,
        seed=7,
    )
    rows_a = sample_particles(np.random.default_rng(config.seed), config, event_id=0)
    rows_b = sample_particles(np.random.default_rng(config.seed), config, event_id=0)
    assert rows_a == rows_b
    assert len(rows_a) == 4


def test_simulate_events_end_to_end():
    layers = [LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0), pitch=1.0)]
    config = SimConfig(
        layers=layers,
        magnetic_field=FieldConfig(bz=0.0),  # no field -> straight tracks, guaranteed to hit
        gun=ParticleGunConfig(n_particles=5, phi_min=-0.1, phi_max=0.1),
        n_events=3,
        seed=42,
    )
    particles, hits, _deposits = simulate_events(config)
    assert len(particles) == 15  # 3 events * 5 particles
    assert np.isinf(particles["radius"]).all()  # bz=0 -> straight (signed_radius returns inf)
    assert len(hits) == 15  # every straight, shallow-angle particle crosses the one layer
    assert set(hits["event_id"]) == {0, 1, 2}


# --- gun modes ----------------------------------------------------------------


def test_unknown_gun_mode_is_rejected():
    with pytest.raises(ValueError):
        ParticleGunConfig(mode="bogus")


def test_standard_mode_is_the_default():
    assert ParticleGunConfig().mode == "standard"


def test_jets_mode_keeps_the_standard_particle_multiplicity():
    config = SimConfig(gun=ParticleGunConfig(n_particles=17, mode="jets"))
    rows = sample_particles(np.random.default_rng(0), config, event_id=0)
    assert len(rows) == 17


def test_jets_mode_groups_particles_into_a_handful_of_collimated_axes():
    # a wide phi range and a tiny cone_sigma make each jet's particles land
    # in a tight cluster well separated from the others, so sorting phi0 and
    # looking for the gaps recovers exactly jet_count_min/_max clusters.
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=200,
            mode="jets",
            jet_count_min=3,
            jet_count_max=3,
            jet_cone_sigma=0.01,
            phi_min=-3.0,
            phi_max=3.0,
            pt_min=1.0,
            pt_max=1.0,
        )
    )
    rows = sample_particles(np.random.default_rng(0), config, event_id=0)
    phis = sorted(row["phi0"] for row in rows)
    n_clusters = 1 + sum(1 for a, b in zip(phis, phis[1:]) if b - a > 0.1)
    assert n_clusters == 3


def test_jets_mode_particles_stay_tight_around_their_jet_axis():
    # a single jet (jet_count_min == jet_count_max == 1): every particle in
    # the event shares one axis, so their phi0 spread should be of order
    # jet_cone_sigma, not of order the gun's full phi_min/phi_max range.
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=100,
            mode="jets",
            jet_count_min=1,
            jet_count_max=1,
            jet_cone_sigma=0.05,
            phi_min=-3.0,
            phi_max=3.0,
        )
    )
    rows = sample_particles(np.random.default_rng(3), config, event_id=0)
    phis = np.array([row["phi0"] for row in rows])
    assert phis.std() < 0.2  # << the 6-radian phi_min/phi_max span


def test_jets_mode_b_jets_are_displaced_along_their_axis():
    # a single jet, forced to always be a b-jet, with a fixed decay length --
    # every particle's vertex should land exactly `decay_length` away from
    # the (default, origin) primary vertex, regardless of its own phi0
    # cone-smear (the vertex is set by the jet axis, not by each particle).
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=50,
            mode="jets",
            jet_count_min=1,
            jet_count_max=1,
            jet_cone_sigma=0.05,
            b_jet_fraction=1.0,
            b_jet_decay_length_min=50.0,
            b_jet_decay_length_max=50.0,
        )
    )
    rows = sample_particles(np.random.default_rng(5), config, event_id=0)
    assert len(rows) == 50
    for row in rows:
        assert math.hypot(row["x0"], row["y0"]) == pytest.approx(50.0)


def test_jets_mode_b_jet_fraction_zero_keeps_every_jet_at_the_primary_vertex():
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=50,
            mode="jets",
            jet_count_min=3,
            jet_count_max=3,
            b_jet_fraction=0.0,
            vertex_x=1.0,
            vertex_y=-2.0,
        )
    )
    rows = sample_particles(np.random.default_rng(6), config, event_id=0)
    assert all(row["x0"] == pytest.approx(1.0) and row["y0"] == pytest.approx(-2.0) for row in rows)


def test_jets_mode_b_jet_decay_length_stays_within_configured_bounds():
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=300,
            mode="jets",
            jet_count_min=4,
            jet_count_max=4,
            jet_cone_sigma=0.01,
            b_jet_fraction=1.0,
            b_jet_decay_length_min=10.0,
            b_jet_decay_length_max=90.0,
        )
    )
    rows = sample_particles(np.random.default_rng(11), config, event_id=0)
    distances = [math.hypot(row["x0"], row["y0"]) for row in rows]
    assert all(10.0 - 1e-9 <= d <= 90.0 + 1e-9 for d in distances)


def test_jets_mode_particles_carry_jet_id_and_is_b_jet_truth():
    config = SimConfig(
        gun=ParticleGunConfig(n_particles=100, mode="jets", jet_count_min=3, jet_count_max=3, b_jet_fraction=0.5)
    )
    rows = sample_particles(np.random.default_rng(4), config, event_id=0)
    assert all(row["jet_id"] in (0, 1, 2) for row in rows)
    # every row on the same jet_id agrees on is_b_jet -- it's an axis property
    by_jet = {}
    for row in rows:
        by_jet.setdefault(row["jet_id"], set()).add(row["is_b_jet"])
    assert all(len(flags) == 1 for flags in by_jet.values())


def test_standard_and_anomaly_modes_tag_particles_as_not_belonging_to_a_jet():
    standard = sample_particles(
        np.random.default_rng(0), SimConfig(gun=ParticleGunConfig(n_particles=5, mode="standard")), event_id=0
    )
    assert all(row["jet_id"] == -1 and row["is_b_jet"] is False for row in standard)

    anomaly = sample_particles(
        np.random.default_rng(0),
        SimConfig(gun=ParticleGunConfig(n_particles=3, mode="anomaly", anomaly_rate=1.0)),
        event_id=0,
    )
    assert len(anomaly) > 3  # the injected cluster landed
    assert all(row["jet_id"] == -1 and row["is_b_jet"] is False for row in anomaly)


def test_jets_mode_new_b_jet_knobs_default_to_off_and_preserve_multiplicity():
    # b_jet_track_boost/pt_boost/muon fractions all default to 0.0, so even
    # with every jet forced to be a b-jet, the total particle count and pt
    # range stay exactly what they were before these knobs existed.
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=40,
            mode="jets",
            jet_count_min=2,
            jet_count_max=2,
            b_jet_fraction=1.0,
            pt_min=5.0,
            pt_max=5.0,
        )
    )
    rows = sample_particles(np.random.default_rng(7), config, event_id=0)
    assert len(rows) == 40
    assert all(row["energy"] == pytest.approx(5.0) for row in rows)
    assert all(row["species"] != "mu-" and row["species"] != "mu+" for row in rows)


def test_jets_mode_b_jet_track_boost_adds_extra_particles_only_to_b_jets():
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=20,
            mode="jets",
            jet_count_min=1,
            jet_count_max=1,
            b_jet_fraction=1.0,
            b_jet_track_boost=0.5,
        )
    )
    rows = sample_particles(np.random.default_rng(1), config, event_id=0)
    # 20 baseline on the one (b-)jet axis, +50% = 10 extra -> 30 total
    assert len(rows) == 30
    assert all(row["is_b_jet"] for row in rows)


def test_jets_mode_b_jet_track_boost_leaves_light_jets_untouched():
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=20,
            mode="jets",
            jet_count_min=1,
            jet_count_max=1,
            b_jet_fraction=0.0,
            b_jet_track_boost=0.5,
        )
    )
    rows = sample_particles(np.random.default_rng(1), config, event_id=0)
    assert len(rows) == 20


def test_jets_mode_b_jet_pt_boost_scales_pt_for_b_jet_particles_only():
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=10,
            mode="jets",
            jet_count_min=1,
            jet_count_max=1,
            b_jet_fraction=1.0,
            b_jet_pt_boost=0.2,
            pt_min=10.0,
            pt_max=10.0,
        )
    )
    rows = sample_particles(np.random.default_rng(2), config, event_id=0)
    assert all(row["energy"] == pytest.approx(12.0) for row in rows)


def test_jets_mode_injects_a_muon_when_fraction_is_one():
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=5,
            mode="jets",
            jet_count_min=1,
            jet_count_max=1,
            b_jet_fraction=0.0,
            jet_muon_fraction=1.0,
        )
    )
    rows = sample_particles(np.random.default_rng(9), config, event_id=0)
    assert len(rows) == 6  # the 5 standard particles + one injected muon
    muon_rows = [row for row in rows if row["species"] in ("mu-", "mu+")]
    assert len(muon_rows) == 1
    assert muon_rows[0]["is_b_jet"] is False


def test_jets_mode_b_jet_muon_fraction_is_independent_of_light_jet_fraction():
    # every jet is a b-jet, b-jet muon fraction is 1.0, light-jet fraction is
    # 0.0 -- since there are no light jets here this just pins the b-jet path
    # deterministically: exactly one injected muon per jet axis.
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=30,
            mode="jets",
            jet_count_min=3,
            jet_count_max=3,
            b_jet_fraction=1.0,
            jet_muon_fraction=0.0,
            b_jet_muon_fraction=1.0,
        )
    )
    rows = sample_particles(np.random.default_rng(13), config, event_id=0)
    muon_rows = [row for row in rows if row["species"] in ("mu-", "mu+")]
    assert len(muon_rows) == 3
    assert all(row["is_b_jet"] for row in muon_rows)


def test_anomaly_mode_never_injects_when_rate_is_zero():
    config = SimConfig(gun=ParticleGunConfig(n_particles=3, mode="anomaly", anomaly_rate=0.0))
    rows = sample_particles(np.random.default_rng(2), config, event_id=0)
    assert len(rows) == 3


def test_anomaly_mode_injects_a_lined_up_calo_and_dimuon_cluster_when_rate_is_one():
    config = SimConfig(
        gun=ParticleGunConfig(
            n_particles=3,
            mode="anomaly",
            anomaly_rate=1.0,
            anomaly_calo_species="photon",
            anomaly_calo_scale=2.0,
            anomaly_muon_scale=1.5,
            pt_min=10.0,
            pt_max=10.0,
        )
    )
    rows = sample_particles(np.random.default_rng(1), config, event_id=0)
    assert len(rows) == 3 + 4  # the 3 standard particles, plus the injected cluster

    extra = rows[3:]
    species_counts = {name: [row["species"] for row in extra].count(name) for name in ("photon", "mu+", "mu-")}
    assert species_counts == {"photon": 2, "mu+": 1, "mu-": 1}

    # the two photons are exactly back-to-back, and each muon lines up with
    # (shares the axis of) one of them -- the "lined up" anomalous topology.
    photon_phis = [row["phi0"] for row in extra if row["species"] == "photon"]
    assert abs(math.remainder(photon_phis[1] - photon_phis[0], 2 * math.pi)) == pytest.approx(math.pi)
    for row in extra:
        if row["species"] in ("mu+", "mu-"):
            assert any(math.remainder(row["phi0"] - phi, 2 * math.pi) == pytest.approx(0.0) for phi in photon_phis)

    # energies scale off pt_max, and are well above the standard particles'
    photon_energy = get_species("photon")
    assert all(row["energy"] == pytest.approx(20.0) for row in extra if row["species"] == "photon")
    assert all(row["energy"] == pytest.approx(15.0) for row in extra if row["species"] in ("mu+", "mu-"))
    assert photon_energy.charge == 0.0  # sanity: the calo species really is neutral


def test_anomaly_mode_falls_back_to_standard_sampling_for_the_baseline_particles():
    # with the injected cluster turned off, `anomaly` mode is indistinguishable
    # from `standard` for the baseline n_particles (same rng draws).
    gun_kwargs = dict(n_particles=6, phi_min=-1.0, phi_max=1.0, pt_min=2.0, pt_max=8.0)
    standard = SimConfig(gun=ParticleGunConfig(mode="standard", **gun_kwargs))
    anomaly = SimConfig(gun=ParticleGunConfig(mode="anomaly", anomaly_rate=0.0, **gun_kwargs))
    rows_standard = sample_particles(np.random.default_rng(9), standard, event_id=0)
    rows_anomaly = sample_particles(np.random.default_rng(9), anomaly, event_id=0)
    assert rows_standard == rows_anomaly
