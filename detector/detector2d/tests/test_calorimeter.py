import math

import pytest
from detector2d.calorimeter import CaloRing, build_calo_stack
from detector2d.geometry import CircleLayer
from detector2d.intersect import first_intersection
from detector2d.geometry import Trajectory


def _ring(n_phi=8, radius=100.0, phi_offset=0.0):
    return CaloRing(layer_id=0, center=(0.0, 0.0), radius=radius, n_phi=n_phi, phi_offset=phi_offset)


def test_calo_ring_is_a_circle_layer_so_intersection_works_unchanged():
    """The whole reason CaloRing subclasses CircleLayer: detector2d.intersect
    needs no knowledge of cells to find where a particle entered one."""
    ring = _ring(radius=100.0)
    assert isinstance(ring, CircleLayer)
    hit = first_intersection(Trajectory(x0=0.0, y0=0.0, phi0=0.0), ring)
    assert hit is not None
    assert (hit.x, hit.y) == pytest.approx((100.0, 0.0))


def test_pitch_is_derived_from_the_cell_count():
    ring = _ring(n_phi=64, radius=100.0)
    assert ring.pitch == pytest.approx(2 * math.pi * 100.0 / 64)
    assert ring.dphi == pytest.approx(2 * math.pi / 64)


def test_cells_tile_the_full_circle_exactly_once():
    ring = _ring(n_phi=16)
    seen = {ring.cell_index(phi) for phi in [i * ring.dphi + 0.5 * ring.dphi for i in range(16)]}
    assert seen == set(range(16))


def test_cell_index_and_center_round_trip():
    ring = _ring(n_phi=32)
    for index in range(32):
        assert ring.cell_index(ring.cell_center_phi(index)) == index


def test_cell_indexing_wraps_across_the_branch_cut():
    """Cell 0 and cell n_phi-1 are physical neighbours; a phi just below the
    +-pi branch cut must not fall off the end of the range."""
    ring = _ring(n_phi=8)
    assert ring.cell_index(-0.01) == ring.cell_index(2 * math.pi - 0.01) == 7
    assert ring.cell_index(0.01) == 0
    assert all(0 <= ring.cell_index(phi) < 8 for phi in [-math.pi, -3.0, 0.0, 3.0, math.pi, 7.0])


def test_cell_position_lies_on_the_ring():
    ring = _ring(n_phi=12, radius=250.0)
    for index in range(12):
        x, y = ring.cell_position(index)
        assert math.hypot(x, y) == pytest.approx(250.0)


def test_stack_fills_outward_with_rings_at_slab_centers():
    rings = build_calo_stack(layer_id_base=100, r_inner=210.0, n_layers=3, thickness=30.0, n_phi=64)
    assert [r.layer_id for r in rings] == [100, 101, 102]
    assert [r.radius for r in rings] == pytest.approx([225.0, 255.0, 285.0])
    assert all(r.thickness == 30.0 for r in rings)
    # the stack's full radial extent is exactly r_inner .. r_inner + n*thickness
    assert rings[0].radius - rings[0].thickness / 2 == pytest.approx(210.0)
    assert rings[-1].radius + rings[-1].thickness / 2 == pytest.approx(300.0)


def test_stack_carries_the_system_tag():
    rings = build_calo_stack(100, 210.0, 2, 30.0, 64, system="hcal")
    assert {r.system for r in rings} == {"hcal"}


def test_half_bin_stagger_puts_layer_1_boundaries_mid_cell_in_layers_0_and_2():
    """The point of staggering the middle ECAL layer: a shower landing exactly
    on a cell boundary in layer 0 -- the worst case, its energy split between
    two cells -- lands in the *middle* of a layer-1 cell, where it is measured
    cleanly. Layers 0 and 2 stay aligned with each other."""
    rings = build_calo_stack(100, 210.0, 3, 30.0, 128, phi_stagger=[0.0, 0.5, 0.0])
    layer0, layer1, layer2 = rings

    boundary_phi = layer0.cell_edges(10)[1]  # a cell boundary in layer 0
    assert layer0.phi_offset == layer2.phi_offset == 0.0
    assert layer1.phi_offset == pytest.approx(0.5 * layer1.dphi)

    # that boundary sits exactly at the center of whichever layer-1 cell holds it
    index1 = layer1.cell_index(boundary_phi)
    low, high = layer1.cell_edges(index1)
    assert boundary_phi == pytest.approx(0.5 * (low + high))

    # layers 0 and 2, both unstaggered, bin identically
    for phi in (-2.0, -0.3, 0.0, 1.7, 3.0):
        assert layer0.cell_index(phi) == layer2.cell_index(phi)
        # ...and layer 1's cells are offset from theirs by half a cell
        assert abs(layer1.cell_center_phi(layer1.cell_index(phi))
                   - layer0.cell_center_phi(layer0.cell_index(phi))) == pytest.approx(
            0.5 * layer0.dphi
        )


def test_stagger_list_is_padded_and_truncated_to_the_layer_count():
    rings = build_calo_stack(100, 210.0, 3, 30.0, 64, phi_stagger=[0.5])
    assert rings[0].phi_offset == pytest.approx(0.5 * rings[0].dphi)
    assert rings[1].phi_offset == 0.0
    assert rings[2].phi_offset == 0.0


def test_invalid_geometry_is_rejected():
    with pytest.raises(ValueError):
        CaloRing(layer_id=0, center=(0.0, 0.0), radius=10.0, n_phi=0)
    with pytest.raises(ValueError):
        build_calo_stack(0, 10.0, 0, 5.0, 8)
