from pathlib import Path

import pytest
from detector2d.calorimeter import CaloRing
from detector2d.geometry import CircleLayer, LineLayer
from detectorsim2d.config import GUN_MODES, ParticleGunConfig, SimConfig, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
CONFIG_PATH = CONFIGS / "default.yaml"
BARREL6_CONFIG_PATH = CONFIGS / "barrel6.yaml"
FULL_DETECTOR_CONFIG_PATH = CONFIGS / "full_detector.yaml"
JETS_CONFIG_PATH = CONFIGS / "jets.yaml"
ANOMALY_CONFIG_PATH = CONFIGS / "anomaly.yaml"


def test_defaults_with_no_config():
    config = load_config(None)
    assert config == SimConfig()
    assert config.layers == []
    assert config.n_events == 1
    assert config.tracker_boundary is None


def test_tracker_boundary_loaded_from_yaml(tmp_path):
    config_path = tmp_path / "with_boundary.yaml"
    config_path.write_text("tracker_boundary: 210.0\n")
    config = load_config(config_path)
    assert config.tracker_boundary == 210.0


def test_load_default_yaml_parses_layers_and_gun():
    config = load_config(CONFIG_PATH)
    assert len(config.layers) == 6
    assert isinstance(config.layers[0], LineLayer)
    assert config.layers[0].p1 == (10.0, -50.0)
    assert config.layers[0].pitch == 1.0
    assert isinstance(config.layers[-1], CircleLayer)
    assert config.layers[-1].radius == 5.0

    assert config.magnetic_field.bz == 1.0
    assert config.gun.n_particles == 3
    assert config.gun.charges == (-1.0, 1.0)
    assert config.n_events == 1


def test_load_barrel6_yaml_detailed_builds_tilted_module_rings():
    config = load_config(BARREL6_CONFIG_PATH)
    # every module is a LineLayer; layer_ids 0-5 each appear many times (one
    # per module in that layer's ring), not once each -- this exercises
    # detectorsim2d's delegation to detector2d.config.build_layers_from_raw for
    # the `detector:` form (see detector2d's own tests for the parsing unit
    # tests).
    assert all(isinstance(layer, LineLayer) for layer in config.layers)
    layer_ids = {layer.layer_id for layer in config.layers}
    assert layer_ids == {0, 1, 2, 3, 4, 5}
    per_layer_counts = {lid: sum(1 for layer in config.layers if layer.layer_id == lid) for lid in layer_ids}
    assert all(count > 1 for count in per_layer_counts.values())
    # precision layers (0-2) use the precision module size/pitch, outer (3-5) the outer one
    assert all(layer.pitch == 0.1 for layer in config.layers if layer.layer_id in (0, 1, 2))
    assert all(layer.pitch == 0.5 for layer in config.layers if layer.layer_id in (3, 4, 5))


def test_a_tracker_only_config_is_unaffected_by_the_calorimeter_extension():
    """No `calorimeter:`/`muon:`/`field.regions:` keys means exactly the old
    behaviour: no calo rings, no piecewise field, no response model."""
    config = load_config(BARREL6_CONFIG_PATH)
    assert not any(isinstance(layer, CaloRing) for layer in config.layers)
    assert config.field_regions is None
    assert config.response is None
    assert config.magnetic_field.bz == 1.0  # the scalar `field: {bz: 1.0}` form


def test_load_full_detector_yaml_builds_every_subsystem():
    config = load_config(FULL_DETECTOR_CONFIG_PATH)
    by_system = {}
    for layer in config.layers:
        by_system.setdefault(layer.system, []).append(layer)

    assert set(by_system) == {"tracker", "ecal", "hcal", "muon"}
    assert len(by_system["ecal"]) == 3
    assert len(by_system["hcal"]) == 2
    assert len(by_system["muon"]) == 3 * 3 * 8  # stations x planes x sides
    assert all(isinstance(layer, CaloRing) for layer in by_system["ecal"] + by_system["hcal"])
    assert [ring.radius for ring in by_system["ecal"]] == [225.0, 255.0, 285.0]
    assert [ring.n_phi for ring in by_system["ecal"]] == [256, 256, 256]
    assert [ring.n_phi for ring in by_system["hcal"]] == [64, 64]


def test_full_detector_yaml_staggers_the_middle_ecal_layer_by_half_a_cell():
    config = load_config(FULL_DETECTOR_CONFIG_PATH)
    ecal = sorted(
        (l for l in config.layers if l.system == "ecal"), key=lambda ring: ring.radius
    )
    assert ecal[0].phi_offset == 0.0
    assert ecal[1].phi_offset == pytest.approx(0.5 * ecal[1].dphi)
    assert ecal[2].phi_offset == 0.0


def test_full_detector_yaml_field_is_strong_then_zero_then_reversed():
    config = load_config(FULL_DETECTOR_CONFIG_PATH)
    regions = config.field_regions
    assert regions is not None
    assert [r.bz for r in regions.regions] == [2.0, 0.0, -1.0]
    assert regions.bz_at(100.0) == 2.0  # tracker
    assert regions.bz_at(350.0) == 0.0  # calorimeters
    assert regions.bz_at(600.0) == -1.0  # muon system: half strength, reversed
    # the particles table's nominal radius uses the innermost field
    assert config.magnetic_field.bz == 2.0


def test_full_detector_yaml_loads_the_response_model_and_gun_species():
    config = load_config(FULL_DETECTOR_CONFIG_PATH)
    assert config.response.em.layer_fractions == (0.60, 0.28, 0.12)
    assert config.response.hadron.layer_fractions == (0.65, 0.35)
    assert config.response.mip_energy == {"ecal": 0.3, "hcal": 0.8}
    assert config.response.stochastic == {"ecal": 0.10, "hcal": 0.50}
    assert "mu+" in config.gun.species and "photon" in config.gun.species
    assert config.world_radius == 800.0
    assert config.max_path_length == 4000.0


def test_scalar_field_still_parses_as_one_constant_field(tmp_path):
    config_path = tmp_path / "scalar_field.yaml"
    config_path.write_text("field:\n  bz: 1.5\n  k: 0.3\n")
    config = load_config(config_path)
    assert config.magnetic_field.bz == 1.5
    assert config.magnetic_field.k == 0.3
    assert config.field_regions is None  # not a piecewise map


def test_gun_defaults_to_standard_mode():
    assert ParticleGunConfig().mode == "standard"
    assert GUN_MODES == ("standard", "jets", "anomaly")


def test_unknown_gun_mode_is_rejected():
    with pytest.raises(ValueError):
        ParticleGunConfig(mode="not-a-mode")


def test_load_jets_yaml_sets_jets_mode_and_jet_parameters():
    config = load_config(JETS_CONFIG_PATH)
    assert config.gun.mode == "jets"
    assert config.gun.jet_count_min == 2
    assert config.gun.jet_count_max == 4
    assert config.gun.jet_cone_sigma == pytest.approx(0.1)
    assert config.gun.n_particles == 50


def test_gun_b_jet_parameters_default_to_a_15_percent_displaced_fraction():
    gun = ParticleGunConfig()
    assert gun.b_jet_fraction == pytest.approx(0.15)
    assert gun.b_jet_decay_length_min == 0.0
    assert gun.b_jet_decay_length_max == pytest.approx(20.0)


def test_load_jets_yaml_sets_b_jet_parameters():
    config = load_config(JETS_CONFIG_PATH)
    assert config.gun.b_jet_fraction == pytest.approx(0.15)
    assert config.gun.b_jet_decay_length_min == pytest.approx(10.0)
    assert config.gun.b_jet_decay_length_max == pytest.approx(90.0)


def test_gun_b_jet_track_pt_and_muon_knobs_default_to_off():
    gun = ParticleGunConfig()
    assert gun.b_jet_track_boost == 0.0
    assert gun.b_jet_pt_boost == 0.0
    assert gun.jet_muon_fraction == 0.0
    assert gun.b_jet_muon_fraction == 0.0


def test_gun_b_jet_track_pt_and_muon_knobs_are_settable():
    gun = ParticleGunConfig(
        b_jet_track_boost=0.2, b_jet_pt_boost=0.15, jet_muon_fraction=0.02, b_jet_muon_fraction=0.15
    )
    assert gun.b_jet_track_boost == pytest.approx(0.2)
    assert gun.b_jet_pt_boost == pytest.approx(0.15)
    assert gun.jet_muon_fraction == pytest.approx(0.02)
    assert gun.b_jet_muon_fraction == pytest.approx(0.15)


def test_load_anomaly_yaml_sets_anomaly_mode_and_anomaly_parameters():
    config = load_config(ANOMALY_CONFIG_PATH)
    assert config.gun.mode == "anomaly"
    assert config.gun.anomaly_rate == pytest.approx(0.5)
    assert config.gun.anomaly_calo_species == "photon"
    assert config.gun.anomaly_calo_scale == pytest.approx(2.0)
    assert config.gun.anomaly_muon_scale == pytest.approx(1.5)


def test_detector_and_flat_layers_are_mutually_exclusive(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        """
        detector:
          mode: simplified
          layers: [{layer_id: 0, radius: 10.0, kind: precision}]
          module_types: {precision: {half_length: 1.0, tilt_deg: 0.0, overlap_fraction: 0.0}}
        layers:
          - {kind: circle, layer_id: 0, center: [0.0, 0.0], radius: 10.0}
        """
    )
    with pytest.raises(ValueError):
        load_config(bad_config)
