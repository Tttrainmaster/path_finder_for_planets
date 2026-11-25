"""
================================================================================
UNIT TESTS: Planetary Probe Path Feasibility & Safety Sorter
================================================================================
Validates physical constraint checks, sorting order, multi-criteria scoring,
and edge cases for planetary probe navigation.
"""

import unittest
import math
from planetary_probe_path_sorter import (
    PlanetaryBody,
    ProbeProfile,
    TerrainMap,
    PathEvaluator,
    ProbePathSorter,
    PathFeasibilityStatus,
    MARS,
    MOON_SOUTH_POLE,
    EUROPA
)

class TestPlanetaryProbePathSorter(unittest.TestCase):

    def setUp(self):
        # Controlled terrain with known geometry
        self.terrain = TerrainMap(
            grid_size_x=50,
            grid_size_y=50,
            cell_resolution_m=10.0,
            body=MARS,
            seed=42
        )
        self.probe = ProbeProfile(
            name="Test Probe",
            mass_kg=500.0,
            max_safe_slope_deg=20.0,
            hard_limit_slope_deg=25.0,
            max_hazard_tolerance=0.50,
            cruise_speed_mps=0.20,
            battery_capacity_wh=1000.0,
            initial_soc=0.80,
            critical_soc_threshold=0.20,
            solar_panel_area_m2=1.5,
            solar_efficiency=0.25,
            base_avionics_power_w=60.0
        )
        self.sorter = ProbePathSorter(self.terrain, self.probe)

    def test_flat_safe_path_is_feasible(self):
        """A path through flat, hazard-free ground must be FEASIBLE with high CSI."""
        # Synthesize a known safe flat zone in South-West corner
        safe_waypoints = [(10.0, 10.0), (30.0, 20.0), (50.0, 30.0)]
        candidate = [("P_SAFE", "Safe Path", safe_waypoints)]
        
        results = self.sorter.evaluate_and_sort_paths(candidate)
        self.assertEqual(len(results), 1)
        report = results[0]
        
        self.assertEqual(report.status, PathFeasibilityStatus.FEASIBLE)
        self.assertGreater(report.composite_safety_index, 75.0)
        self.assertEqual(len(report.failure_reasons), 0)

    def test_fatal_slope_rollover_constraint(self):
        """A path traversing a cliff (> 28 degrees) must be marked UNFEASIBLE."""
        # Deep crater rim traversal
        steep_waypoints = [(180.0, 150.0), (180.0, 240.0)]
        candidate = [("P_STEEP", "Steep Wall Crossing", steep_waypoints)]
        
        results = self.sorter.evaluate_and_sort_paths(candidate)
        report = results[0]
        
        self.assertEqual(report.status, PathFeasibilityStatus.UNFEASIBLE)
        self.assertTrue(any("ROLLOVER" in r for r in report.failure_reasons))
        self.assertGreater(report.max_slope_deg, self.probe.hard_limit_slope_deg)

    def test_immobilization_hazard_constraint(self):
        """A path directly through an intense boulder field must trigger immobilization warning."""
        # (350, 180) is an engineered boulder cluster with hazard > 0.8
        boulder_waypoints = [(330.0, 180.0), (370.0, 180.0)]
        candidate = [("P_BOULDER", "Boulder Field Direct", boulder_waypoints)]
        
        results = self.sorter.evaluate_and_sort_paths(candidate)
        report = results[0]
        
        self.assertEqual(report.status, PathFeasibilityStatus.UNFEASIBLE)
        self.assertTrue(any("IMMOBILIZATION" in r for r in report.failure_reasons))
        self.assertGreater(report.max_hazard_encountered, self.probe.max_hazard_tolerance)

    def test_sorting_order_safety_precedence(self):
        """
        Feasible paths MUST always rank ahead of unfeasible paths,
        and higher Composite Safety Index (CSI) must sort before lower CSI.
        """
        path_safe = ("P1", "Safe Clear Route", [(10.0, 10.0), (40.0, 20.0), (60.0, 30.0)])
        path_marginal = ("P2", "Steep Sloped Route", [(100.0, 100.0), (180.0, 200.0)])
        path_suicidal = ("P3", "Direct Crater Suicide Route", [(150.0, 150.0), (180.0, 240.0)])
        
        candidates = [path_suicidal, path_marginal, path_safe]
        sorted_reports = self.sorter.evaluate_and_sort_paths(candidates)
        
        # Verify rank 1 is feasible
        self.assertEqual(sorted_reports[0].path_id, "P1")
        self.assertEqual(sorted_reports[0].status, PathFeasibilityStatus.FEASIBLE)
        
        # Verify monotonicity of Composite Safety Index within same status class
        for i in range(len(sorted_reports) - 1):
            curr_rep = sorted_reports[i]
            next_rep = sorted_reports[i + 1]
            if curr_rep.status == next_rep.status:
                self.assertGreaterEqual(curr_rep.composite_safety_index, next_rep.composite_safety_index)

    def test_battery_depletion_in_perpetual_darkness(self):
        """Tests probe power depletion when traversing persistent shadowed terrain."""
        # Create a probe with low starting battery and zero solar in shadow
        depleted_probe = ProbeProfile(
            name="Low Battery Probe",
            battery_capacity_wh=100.0,
            initial_soc=0.28,
            critical_soc_threshold=0.25,
            base_avionics_power_w=150.0, # High power draw
            solar_panel_area_m2=0.0      # Night / complete shadow
        )
        evaluator = PathEvaluator(self.terrain, depleted_probe)
        long_waypoints = [(10.0, 10.0), (200.0, 200.0)]
        report = evaluator.evaluate_path("P_DARK", "Darkness Run", long_waypoints)
        
        self.assertEqual(report.status, PathFeasibilityStatus.UNFEASIBLE)
        self.assertTrue(any("CRITICAL POWER" in r for r in report.failure_reasons))
        self.assertLessEqual(report.minimum_soc_pct, depleted_probe.critical_soc_threshold * 100.0)

    def test_multi_planetary_body_adaptability(self):
        """Verifies evaluator adapts to different gravitational and radiation regimes."""
        # Mars vs Europa
        mars_eval = PathEvaluator(self.terrain, self.probe)
        europa_terrain = TerrainMap(grid_size_x=50, grid_size_y=50, body=EUROPA, seed=42)
        europa_eval = PathEvaluator(europa_terrain, self.probe)
        
        wps = [(20.0, 20.0), (100.0, 80.0)]
        rep_mars = mars_eval.evaluate_path("MARS_1", "Mars Path", wps)
        rep_europa = europa_eval.evaluate_path("EUR_1", "Europa Path", wps)
        
        # Europa has 2.5 mSv/hr background vs Mars 0.03 mSv/hr
        self.assertGreater(rep_europa.cumulative_radiation_msv, rep_mars.cumulative_radiation_msv * 10.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
