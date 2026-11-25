"""
================================================================================
PLANETARY PROBE PATH FEASIBILITY & SAFETY EVALUATION ENGINE
================================================================================
An aerospace/robotics engineering tool for extraterrestrial mission planning.
Evaluates, ranks, and sorts planetary probe trajectories across complex terrain
using multi-criteria physical constraints:
  - Incline & Rollover Stability (Digital Elevation Model gradients)
  - Terrain Roughness & Hazard Clearance (Boulder fields, craters, sand traps)
  - Power & Battery State-of-Charge Dynamics (Solar flux vs climb/avionics load)
  - Environmental & Radiation Dose
  - Telemetry & Communication Line-of-Sight (Relay/Earth visibility)
  - Kinematic Efficiency & Traversal Time

Author: Aerospace Robotics & Navigation Systems Engineer
================================================================================
"""

from __future__ import annotations
import math
import heapq
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ==============================================================================
# 1. PHYSICAL CONSTANTS & PLANETARY BODY DEFINITIONS
# ==============================================================================

@dataclass(frozen=True)
class PlanetaryBody:
    """Astrodynamic and environmental parameters of a target celestial body."""
    name: str
    gravity: float              # Surface gravity (m/s^2)
    solar_flux_peak: float       # Peak solar irradiance at surface (W/m^2)
    ambient_temp_k: float        # Mean surface temperature (Kelvin)
    surface_friction_coeff: float # Soil/regolith static friction coefficient
    radiation_background_msv: float # Background radiation dose rate (mSv/hr)
    dust_rate_msv: float           # Dust accumulation rate (mSv/hr)


# Pre-configured celestial bodies
MARS = PlanetaryBody(
    name="Mars (Jezero Crater)",
    gravity=3.721,
    solar_flux_peak=590.0,
    ambient_temp_k=210.0,
    surface_friction_coeff=0.55,
    radiation_background_msv=0.030,
    dust_rate_msv=input("Enter estimated dust accumulation rate on Mars (mSv/hr): ")
)

MOON_SOUTH_POLE = PlanetaryBody(
    name="Moon (Shackleton Rim)",
    gravity=1.620,
    solar_flux_peak=1361.0,
    ambient_temp_k=120.0,
    surface_friction_coeff=0.60,
    radiation_background_msv=0.055,
    dust_rate_msv=input("Enter estimated dust accumulation rate on Moon (mSv/hr): ")
)

EUROPA = PlanetaryBody(
    name="Europa (Chaos Terrain)",
    gravity=1.315,
    solar_flux_peak=50.0,        # Very low solar (Jovian distance)
    ambient_temp_k=100.0,
    surface_friction_coeff=0.45, # Cryo-ice surface
    radiation_background_msv=2.500, # Extreme Jovian magnetosphere radiation
    dust_rate_msv=input("Enter estimated dust accumulation rate on Europa (mSv/hr): ")        ,#can change per foot have to claculate at the time
)


# ==============================================================================
# 2. PROBE SPECIFICATIONS & OPERATIONAL CONSTRAINTS
# ==============================================================================

@dataclass
class ProbeProfile:
    """Engineering characteristics and survivability thresholds of the probe."""
    name: str
    mass_kg: float = 850.0                    # Rover/probe mass (e.g. Perseverance-class)
    max_safe_slope_deg: float = 22.0          # Critical tip-over/wheel-slip slope
    hard_limit_slope_deg: float = input("Enter the hard limit slope (degrees): ")      # Physical rollover threshold
    max_hazard_tolerance: float = 0.60        # Max traversable obstacle/boulder index [0, 1]
    cruise_speed_mps: float = 0.15            # Traverse velocity (m/s ~ 0.54 km/h)
    
    # Power & Energy Architecture
    battery_capacity_wh: float = 2000.0       # Li-ion / nuclear-assisted battery capacity (Wh)
    initial_soc: float = 0.95                 # Starting State-of-Charge [0.0 - 1.0]
    critical_soc_threshold: float = 0.25      # Below this: survival heater / brownout failure
    solar_panel_area_m2: float = 1.8          # Effective photovoltaic area (m^2)
    solar_efficiency: float = 0.28            # High-efficiency space PV cells
    base_avionics_power_w: float = 90.0       # Telemetry, computers, thermal heaters (Watts)
    rolling_resistance_coeff: float = 0.08    # Regolith rolling resistance coefficient
    drive_motor_efficiency: float = 0.75      # Electromechanical conversion efficiency
    
    # Communication & Environmental
    max_allowable_comm_blackout_pct: float = 40.0 # Max allowable LOS blackout percentage
    max_allowable_radiation_dose_msv: float = 5.0 # Mission dose threshold


# ==============================================================================
# 3. PLANETARY TERRAIN & ENVIRONMENT MODEL
# ==============================================================================

class TerrainMap:
    """
    Represents a 2D Digital Elevation Model (DEM) and multi-spectral hazard map
    of a planetary exploration zone.
    """
    def __init__(
        self,
        grid_size_x: int = 100,
        grid_size_y: int = 100,
        cell_resolution_m: float = 5.0,
        body: PlanetaryBody = str(input("Enter the planetary body): ")),
        seed: int = 42
    ):
        if grid_size_x < 2 or grid_size_y < 2:
            raise ValueError("Terrain grid dimensions must each be at least 2 cells.")
        if not math.isfinite(cell_resolution_m) or cell_resolution_m <= 0:
            raise ValueError("Cell resolution must be a finite positive number.")
        self.nx = grid_size_x
        self.ny = grid_size_y
        self.resolution = cell_resolution_m
        self.body = body
        # Keep a local generator so terrain construction never changes NumPy's
        # process-wide random state. The synthetic terrain is currently deterministic.
        self.rng = np.random.default_rng(seed)
        
        # Physical matrices
        self.elevation = np.zeros((self.ny, self.nx))
        self.hazard_density = np.zeros((self.ny, self.nx)) # [0=clear sand, 1=lethal boulder/fissure]
        self.solar_illumination = np.ones((self.ny, self.nx)) # [0=deep shadow, 1=full sunlight]
        self.comm_line_of_sight = np.ones((self.ny, self.nx), dtype=bool)
        
        self._generate_synthetic_planetary_landscape()

    def _generate_synthetic_planetary_landscape(self):
        """Constructs realistic crater topography, rock fields, and shadows."""
        # Grid samples are cell centers spaced exactly ``resolution`` meters apart.
        x = np.arange(self.nx) * self.resolution
        y = np.arange(self.ny) * self.resolution
        X, Y = np.meshgrid(x, y)
        
        # 1. Rolling volcanic terrain base
        self.elevation = (
            15.0 * np.sin(X / 120.0) * np.cos(Y / 140.0) +
            8.0 * np.sin(X / 60.0 + Y / 70.0) +
            4.0 * np.cos(X / 30.0)
        )
        
        # 2. Primary Impact Crater (Center-Left)
        crater1_center = (180.0, 240.0)
        crater1_r = 90.0
        dist1 = np.sqrt((X - crater1_center[0])**2 + (Y - crater1_center[1])**2)
        crater1_mask = dist1 < crater1_r
        # Crater bowl and raised rim
        crater1_depth = 35.0 * np.exp(-(dist1 / 55.0)**2)
        crater1_rim = 12.0 * np.exp(-((dist1 - crater1_r) / 18.0)**2)
        self.elevation -= crater1_depth
        self.elevation += crater1_rim
        
        # Crater interior hazard and rock ejecta field
        self.hazard_density += 0.75 * np.exp(-(dist1 / (crater1_r * 0.85))**2)
        
        # 3. Secondary Steep Canyon / Ridge (Diagonal)
        ridge_dist = np.abs(0.6 * X - Y + 80.0) / np.sqrt(0.6**2 + 1.0)
        ridge_mask = (ridge_dist < 30.0) & (X > 200.0)
        self.elevation[ridge_mask] += 25.0 * np.cos(ridge_dist[ridge_mask] / 30.0 * np.pi / 2)
        self.hazard_density[ridge_mask] += 0.45
        
        # 4. Boulder Fields / Regolith Sinkholes (Localized Hazards)
        boulder_cluster1 = np.exp(-((X - 350.0)**2 + (Y - 180.0)**2) / (45.0**2))
        boulder_cluster2 = np.exp(-((X - 120.0)**2 + (Y - 380.0)**2) / (35.0**2))
        self.hazard_density += 0.80 * boulder_cluster1 + 0.65 * boulder_cluster2
        self.hazard_density = np.clip(self.hazard_density, 0.0, 1.0)
        
        # 5. Solar Illumination & Deep Shadows (Sun vector from North-West at 30° elevation)
        # Low sun creates long shadows behind high rims and inside the deep crater bowl
        shadow_zone = (dist1 < 60.0) & (X > crater1_center[0]) # Deep east wall of crater
        self.solar_illumination[shadow_zone] = 0.05 # 95% shadow
        ridge_shadow = (ridge_dist < 22.0) & (X > 200.0) & (0.6 * X - Y + 80.0 < 0)
        self.solar_illumination[ridge_shadow] = 0.15
        
        # 6. Communication Occlusion (Depression blocks direct line to orbiter/Earth)
        comm_occlusion = (dist1 < 50.0) & (self.elevation < -15.0)
        self.comm_line_of_sight[comm_occlusion] = False

    def get_interpolated_state(self, x_m: float, y_m: float) -> Tuple[float, float, float, bool]:
        """
        Bilinear interpolation to retrieve exact elevation, hazard,
        solar illumination, and comm status at metric coordinates (x, y).
        """
        gx = max(0.0, min(self.nx - 1.0, x_m / self.resolution))
        gy = max(0.0, min(self.ny - 1.0, y_m / self.resolution))
        
        ix, iy = int(gx), int(gy)
        fx, fy = gx - ix, gy - iy
        
        # Elevation
        e00 = self.elevation[iy, ix]
        e10 = self.elevation[iy, min(ix + 1, self.nx - 1)]
        e01 = self.elevation[min(iy + 1, self.ny - 1), ix]
        e11 = self.elevation[min(iy + 1, self.ny - 1), min(ix + 1, self.nx - 1)]
        elev = (1 - fx) * (1 - fy) * e00 + fx * (1 - fy) * e10 + (1 - fx) * fy * e01 + fx * fy * e11
        
        # Hazard
        h00 = self.hazard_density[iy, ix]
        h10 = self.hazard_density[iy, min(ix + 1, self.nx - 1)]
        h01 = self.hazard_density[min(iy + 1, self.ny - 1), ix]
        h11 = self.hazard_density[min(iy + 1, self.ny - 1), min(ix + 1, self.nx - 1)]
        haz = (1 - fx) * (1 - fy) * h00 + fx * (1 - fy) * h10 + (1 - fx) * fy * h01 + fx * fy * h11
        
        # Solar
        s00 = self.solar_illumination[iy, ix]
        s10 = self.solar_illumination[iy, min(ix + 1, self.nx - 1)]
        s01 = self.solar_illumination[min(iy + 1, self.ny - 1), ix]
        s11 = self.solar_illumination[min(iy + 1, self.ny - 1), min(ix + 1, self.nx - 1)]
        solar = (1 - fx) * (1 - fy) * s00 + fx * (1 - fy) * s10 + (1 - fx) * fy * s01 + fx * fy * s11
        
        # Comm line-of-sight
        comm = bool(self.comm_line_of_sight[iy, ix])
        
        return elev, haz, solar, comm


# ==============================================================================
# 4. PATH EVALUATION METRICS & FEASIBILITY ENGINE
# ==============================================================================

class PathFeasibilityStatus(Enum):
    FEASIBLE = "FEASIBLE (Within Model Limits)"
    MARGINAL = "MARGINAL (Review Constraints)"
    UNFEASIBLE = "UNFEASIBLE (Model Constraint Violated)"


@dataclass
class TrajectoryTelemetry:
    """Telemetry sequence recorded during path evaluation."""
    times_sec: List[float] = field(default_factory=list)
    distances_m: List[float] = field(default_factory=list)
    elevations_m: List[float] = field(default_factory=list)
    slopes_deg: List[float] = field(default_factory=list)
    hazards: List[float] = field(default_factory=list)
    soc_history: List[float] = field(default_factory=list)
    power_net_w: List[float] = field(default_factory=list)
    comm_status: List[bool] = field(default_factory=list)


@dataclass
class PathEvaluationReport:
    """Comprehensive aerospace evaluation result for a candidate path."""
    path_id: str
    name: str
    waypoints: List[Tuple[float, float]]
    status: PathFeasibilityStatus
    failure_reasons: List[str]
    
    # Kinematics & Geometry
    total_distance_m: float
    total_time_hours: float
    max_slope_deg: float
    mean_slope_deg: float
    total_elevation_gain_m: float
    
    # Hazard & Survivability
    max_hazard_encountered: float
    mean_hazard_encountered: float
    cumulative_hazard_risk: float
    
    # Energy Profile
    minimum_soc_pct: float
    final_soc_pct: float
    net_energy_consumed_wh: float
    
    # Environmental & Telemetry
    comm_coverage_pct: float
    max_comm_blackout_min: float
    cumulative_radiation_msv: float
    
    # Normalized Multi-Criteria Scores (0.0 to 100.0)
    score_slope_safety: float
    score_hazard_clearance: float
    score_energy_reserve: float
    score_communication: float
    score_traversal_efficiency: float
    
    # Composite Final Score (High is Safe & Feasible)
    composite_safety_index: float
    
    telemetry: Optional[TrajectoryTelemetry] = None


class PathEvaluator:
    """
    Estimates traversal over synthetic terrain using simplified kinematic,
    power, hazard, radiation, and communications models.
    """
    def __init__(self, terrain: TerrainMap, probe: ProbeProfile):
        self.terrain = terrain
        self.probe = probe
        self.body = terrain.body

    def evaluate_path(
        self,
        path_id: str,
        name: str,
        waypoints: List[Tuple[float, float]],
        sampling_step_m: float = 2.5
    ) -> PathEvaluationReport:
        """
        Performs high-fidelity forward simulation along the path waypoints.
        """
        if len(waypoints) < 2:
            raise ValueError("A path must contain at least a start and a goal waypoint.")
        if not math.isfinite(sampling_step_m) or sampling_step_m <= 0:
            raise ValueError("sampling_step_m must be a finite positive number.")
        for point in waypoints:
            if len(point) != 2 or not all(math.isfinite(float(v)) for v in point):
                raise ValueError("Waypoints must be finite (x, y) coordinate pairs.")
            if not (0 <= point[0] <= (self.terrain.nx - 1) * self.terrain.resolution and
                    0 <= point[1] <= (self.terrain.ny - 1) * self.terrain.resolution):
                raise ValueError("Waypoint lies outside the terrain map bounds.")

        failure_reasons: List[str] = []
        telemetry = TrajectoryTelemetry()
        
        # 1. Resample waypoints into dense discrete steps
        dense_points: List[Tuple[float, float]] = []
        for i in range(len(waypoints) - 1):
            p1 = np.array(waypoints[i])
            p2 = np.array(waypoints[i + 1])
            seg_len = np.linalg.norm(p2 - p1)
            steps = max(1, int(np.ceil(seg_len / sampling_step_m)))
            for s in range(steps):
                alpha = s / steps
                pt = p1 + alpha * (p2 - p1)
                dense_points.append((float(pt[0]), float(pt[1])))
        dense_points.append(waypoints[-1])
        
        # State variables
        current_time_s = 0.0
        current_dist_m = 0.0
        current_soc = self.probe.initial_soc
        min_soc = current_soc
        total_elev_gain = 0.0
        max_slope = 0.0
        slopes_list = []
        hazards_list = []
        comm_list = []
        hazard_distance_sum = 0.0
        comm_visible_distance = 0.0
        radiation_dose = 0.0
        current_blackout_s = 0.0
        max_blackout_s = 0.0
        
        # Initial point telemetry
        e_prev, h_prev, s_prev, c_prev = self.terrain.get_interpolated_state(dense_points[0][0], dense_points[0][1])
        telemetry.times_sec.append(0.0)
        telemetry.distances_m.append(0.0)
        telemetry.elevations_m.append(e_prev)
        telemetry.slopes_deg.append(0.0)
        telemetry.hazards.append(h_prev)
        telemetry.soc_history.append(current_soc * 100.0)
        telemetry.power_net_w.append(0.0)
        telemetry.comm_status.append(c_prev)
        
        # 2. Sequential physical simulation along path
        for i in range(1, len(dense_points)):
            x0, y0 = dense_points[i - 1]
            x1, y1 = dense_points[i]
            dx = x1 - x0
            dy = y1 - y0
            ds_horiz = math.hypot(dx, dy)
            
            if ds_horiz < 1e-6:
                continue
                
            e_curr, h_curr, s_curr, c_curr = self.terrain.get_interpolated_state(x1, y1)
            de = e_curr - e_prev
            ds_3d = math.hypot(ds_horiz, de)
            
            # Slope calculation
            slope_rad = math.atan2(de, ds_horiz)
            slope_deg = math.degrees(abs(slope_rad))
            slopes_list.append(slope_deg)
            hazards_list.append(h_curr)
            comm_list.append(c_curr)
            hazard_distance_sum += h_curr * ds_3d
            if c_curr:
                comm_visible_distance += ds_3d
            
            if slope_deg > max_slope:
                max_slope = slope_deg
                
            if de > 0:
                total_elev_gain += de
                
            # Time delta for step
            dt_s = ds_3d / self.probe.cruise_speed_mps
            current_time_s += dt_s
            current_dist_m += ds_3d
            
            # Communication tracking
            if not c_curr:
                current_blackout_s += dt_s
                if current_blackout_s > max_blackout_s:
                    max_blackout_s = current_blackout_s
            else:
                current_blackout_s = 0.0
                
            # Radiation accumulation
            radiation_dose += (self.body.radiation_background_msv / 3600.0) * dt_s
            
            # 3. Dynamic Power & Energy Balance
            # Physical climbing mechanical resistance & rolling resistance
            # P_drive = (m * g * (C_rr * cos(theta) + sin(theta))) * v / eta
            g = self.body.gravity
            m = self.probe.mass_kg
            v = self.probe.cruise_speed_mps
            crr = self.probe.rolling_resistance_coeff
            eta = self.probe.drive_motor_efficiency
            
            force_rolling = crr * m * g * math.cos(slope_rad)
            force_climb = m * g * math.sin(slope_rad) # positive if uphill, negative if downhill
            
            # Motor drive mechanical power (cannot regenerate fully on loose planetary soil)
            tractive_force = max(0.1 * force_rolling, force_rolling + max(0.0, force_climb))
            p_drive_w = (tractive_force * v) / eta
            p_required_w = self.probe.base_avionics_power_w + p_drive_w
            
            # Solar harvesting power: Area * Irradiance * Efficiency * Illumination factor
            p_solar_w = (
                self.probe.solar_panel_area_m2 *
                self.body.solar_flux_peak *
                self.probe.solar_efficiency *
                s_curr
            )
            p_net_w = p_solar_w - p_required_w
            
            # Delta energy in Watt-hours
            delta_energy_wh = (p_net_w * dt_s) / 3600.0
            new_soc = current_soc + (delta_energy_wh / self.probe.battery_capacity_wh)
            current_soc = max(0.0, min(1.0, new_soc)) # clamped between 0 and 100%
            
            if current_soc < min_soc:
                min_soc = current_soc
                
            # Store telemetry
            telemetry.times_sec.append(current_time_s)
            telemetry.distances_m.append(current_dist_m)
            telemetry.elevations_m.append(e_curr)
            telemetry.slopes_deg.append(slope_deg)
            telemetry.hazards.append(h_curr)
            telemetry.soc_history.append(current_soc * 100.0)
            telemetry.power_net_w.append(p_net_w)
            telemetry.comm_status.append(c_curr)
            
            e_prev = e_curr
            
        # 4. Aerospace Hard Constraints Verification
        max_hazard = max([h_prev, *hazards_list]) if hazards_list else h_prev
        cumulative_hazard = hazard_distance_sum / 100.0
        mean_slope = float(np.mean(slopes_list)) if slopes_list else 0.0
        mean_hazard = hazard_distance_sum / current_dist_m if current_dist_m else 0.0
        comm_coverage_pct = (comm_visible_distance / current_dist_m * 100.0) if current_dist_m else 0.0
        
        # Hard Failure Criteria checks
        if max_slope >= self.probe.hard_limit_slope_deg:
            failure_reasons.append(
                f"FATAL ROLLOVER HAZARD: Peak slope {max_slope:.1f}° exceeds critical stability limit ({self.probe.hard_limit_slope_deg:.1f}°)"
            )
        elif max_slope > self.probe.max_safe_slope_deg:
            failure_reasons.append(
                f"HIGH SLIP RISK: Slope {max_slope:.1f}° exceeds safe operational envelope ({self.probe.max_safe_slope_deg:.1f}°)"
            )
            
        if max_hazard >= self.probe.max_hazard_tolerance:
            failure_reasons.append(
                f"IMMOBILIZATION HAZARD: Boulder/crater roughness {max_hazard:.2f} exceeds wheel clearance threshold ({self.probe.max_hazard_tolerance:.2f})"
            )
            
        if min_soc <= self.probe.critical_soc_threshold:
            failure_reasons.append(
                f"CRITICAL POWER DEPLETION: Battery SoC dropped to {min_soc*100:.1f}%, below survival reserve ({self.probe.critical_soc_threshold*100:.1f}%)"
            )
            
        if (100.0 - comm_coverage_pct) > self.probe.max_allowable_comm_blackout_pct:
            failure_reasons.append(
                f"COMMUNICATION BLACKOUT: Occlusion of {100.0 - comm_coverage_pct:.1f}% exceeds telemetry reliability limit ({self.probe.max_allowable_comm_blackout_pct:.1f}%)"
            )
            
        if radiation_dose > self.probe.max_allowable_radiation_dose_msv:
            failure_reasons.append(
                f"EXCESSIVE RADIATION: Cumulative dose {radiation_dose:.2f} mSv violates rad-hard budget ({self.probe.max_allowable_radiation_dose_msv:.2f} mSv)"
            )
            
        # Determine overall classification
        if any("FATAL" in r or "IMMOBILIZATION" in r or "CRITICAL POWER" in r for r in failure_reasons):
            status = PathFeasibilityStatus.UNFEASIBLE
        elif len(failure_reasons) > 0:
            status = PathFeasibilityStatus.MARGINAL
        else:
            status = PathFeasibilityStatus.FEASIBLE
            
        # 5. Multi-Criteria Normalized Safety Scoring (0 to 100)
        # S_slope: High if slopes are well below probe's limits
        score_slope = max(0.0, 100.0 * (1.0 - (mean_slope / self.probe.max_safe_slope_deg)**1.5))
        if max_slope >= self.probe.hard_limit_slope_deg:
            score_slope *= 0.1 # Severe penalty
            
        # S_hazard: High if boulder density / roughness is minimal
        score_hazard = max(0.0, 100.0 * (1.0 - (mean_hazard / self.probe.max_hazard_tolerance)**1.2))
        if max_hazard >= self.probe.max_hazard_tolerance:
            score_hazard *= 0.1
            
        # S_energy: High if battery stays full and does not dip
        score_energy = max(0.0, 100.0 * ((min_soc - self.probe.critical_soc_threshold) / (1.0 - self.probe.critical_soc_threshold)))
        if min_soc <= self.probe.critical_soc_threshold:
            score_energy = 0.0
            
        # S_comm: High if continuous telemetry is maintained
        score_comm = comm_coverage_pct
        
        # S_efficiency: Straight-line Euclidean distance vs actual traversed path
        euclidean_dist = math.hypot(waypoints[-1][0] - waypoints[0][0], waypoints[-1][1] - waypoints[0][1])
        path_efficiency = min(1.0, euclidean_dist / max(1.0, current_dist_m))
        score_efficiency = path_efficiency * 100.0
        
        # Composite Safety Index (Aerospace weighted utility function)
        # Weights: Incline Safety (30%), Hazard Avoidance (30%), Energy Reserve (20%), Comm Coverage (10%), Efficiency (10%)
        weights = {
            'slope': 0.30,
            'hazard': 0.30,
            'energy': 0.20,
            'comm': 0.10,
            'efficiency': 0.10
        }
        
        raw_csi = (
            weights['slope'] * score_slope +
            weights['hazard'] * score_hazard +
            weights['energy'] * score_energy +
            weights['comm'] * score_comm +
            weights['efficiency'] * score_efficiency
        )
        
        # Status gating penalty on Composite Score
        if status == PathFeasibilityStatus.UNFEASIBLE:
            composite_score = raw_csi * 0.25 # Infeasible paths capped heavily
        elif status == PathFeasibilityStatus.MARGINAL:
            composite_score = raw_csi * 0.70
        else:
            composite_score = raw_csi
            
        initial_energy_wh = self.probe.initial_soc * self.probe.battery_capacity_wh
        current_energy_wh = current_soc * self.probe.battery_capacity_wh
        net_energy_consumed = initial_energy_wh - current_energy_wh
            
        return PathEvaluationReport(
            path_id=path_id,
            name=name,
            waypoints=waypoints,
            status=status,
            failure_reasons=failure_reasons,
            total_distance_m=current_dist_m,
            total_time_hours=current_time_s / 3600.0,
            max_slope_deg=max_slope,
            mean_slope_deg=mean_slope,
            total_elevation_gain_m=total_elev_gain,
            max_hazard_encountered=max_hazard,
            mean_hazard_encountered=mean_hazard,
            cumulative_hazard_risk=cumulative_hazard,
            minimum_soc_pct=min_soc * 100.0,
            final_soc_pct=current_soc * 100.0,
            net_energy_consumed_wh=net_energy_consumed,
            comm_coverage_pct=comm_coverage_pct,
            max_comm_blackout_min=max_blackout_s / 60.0,
            cumulative_radiation_msv=radiation_dose,
            score_slope_safety=score_slope,
            score_hazard_clearance=score_hazard,
            score_energy_reserve=score_energy,
            score_communication=score_comm,
            score_traversal_efficiency=score_efficiency,
            composite_safety_index=composite_score,
            telemetry=telemetry
        )


# ==============================================================================
# 5. PATHFINDER ALGORITHM & CANDIDATE GENERATION (A* & HEURISTIC ROUTES)
# ==============================================================================

class PlanetaryPathPlanner:
    """
    Automated pathfinder that computes weighted candidate paths:
      - Direct Baseline (straight line across terrain)
      - Minimum Hazard & Slope (Multi-objective A* prioritizing rover survivability)
      - Solar Ridge Following (Maximizes PV power & illumination)
      - Perimeter / Ridge Bypass (Circumnavigates hazardous craters)
    """
    def __init__(self, terrain: TerrainMap, probe: ProbeProfile):
        self.terrain = terrain
        self.probe = probe

    def find_optimal_safe_path(
        self,
        start_m: Tuple[float, float],
        goal_m: Tuple[float, float],
        hazard_weight: float = 8.0,
        slope_weight: float = 6.0,
        energy_weight: float = 3.0
    ) -> List[Tuple[float, float]]:
        """
        Multi-objective cost-weighted A* algorithm over the Digital Elevation Model.
        Cost function explicitly penalizes steep gradients, boulder fields, and shadows.
        """
        res = self.terrain.resolution
        max_x = (self.terrain.nx - 1) * res
        max_y = (self.terrain.ny - 1) * res
        for label, point in (("start", start_m), ("goal", goal_m)):
            if (len(point) != 2 or not all(math.isfinite(float(v)) for v in point) or
                    not (0 <= point[0] <= max_x and 0 <= point[1] <= max_y)):
                raise ValueError(f"{label} coordinate lies outside the terrain map bounds.")
        start_idx = (int(round(start_m[0] / res)), int(round(start_m[1] / res)))
        goal_idx = (int(round(goal_m[0] / res)), int(round(goal_m[1] / res)))
        if start_idx == goal_idx:
            return [start_m, goal_m]
        
        # Priority queue: (f_score, g_score, (ix, iy))
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0.0, start_idx))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_scores: Dict[Tuple[int, int], float] = {start_idx: 0.0}
        
        # 8-connected grid neighborhood
        neighbors = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == goal_idx:
                # Reconstruct path
                curr = current
                path = []
                while curr in came_from:
                    path.append((curr[0] * res, curr[1] * res))
                    curr = came_from[curr]
                path.append((start_idx[0] * res, start_idx[1] * res))
                path.reverse()
                path[0] = start_m
                path[-1] = goal_m
                # Keep every searched grid step; subsampling could cut across
                # terrain cells that A* never checked for hazards or slope.
                return path
                
            cx, cy = current
            curr_e = self.terrain.elevation[min(cy, self.terrain.ny - 1), min(cx, self.terrain.nx - 1)]
            
            for dx, dy in neighbors:
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < self.terrain.nx and 0 <= ny < self.terrain.ny):
                    continue
                    
                neighbor = (nx, ny)
                step_dist_m = math.hypot(dx * res, dy * res)
                
                next_e = self.terrain.elevation[ny, nx]
                next_haz = self.terrain.hazard_density[ny, nx]
                next_solar = self.terrain.solar_illumination[ny, nx]
                
                # Incline computation
                delta_z = next_e - curr_e
                slope_deg = math.degrees(math.atan2(abs(delta_z), step_dist_m))
                
                # Hard impassable wall
                if (slope_deg >= self.probe.hard_limit_slope_deg or
                        next_haz >= self.probe.max_hazard_tolerance):
                    continue
                    
                # Multi-objective traversal cost penalty
                slope_penalty = (slope_deg / self.probe.max_safe_slope_deg) ** 2.0
                hazard_penalty = (next_haz / self.probe.max_hazard_tolerance) ** 2.0
                shadow_penalty = (1.0 - next_solar)
                
                step_cost = step_dist_m * (
                    1.0 +
                    hazard_weight * hazard_penalty +
                    slope_weight * slope_penalty +
                    energy_weight * shadow_penalty
                )
                
                tentative_g = current_g + step_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    # Heuristic: Euclidean distance to goal
                    h = math.hypot((goal_idx[0] - nx) * res, (goal_idx[1] - ny) * res)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    came_from[neighbor] = current
                    
        # No route satisfies the planner's constraints. Let the caller handle
        # infeasibility instead of presenting an unchecked straight line.
        return []


# ==============================================================================
# 6. CANDIDATE PATH SORTER & DECISION MATRIX
# ==============================================================================

class ProbePathSorter:
    """
    Ranks, filters, and sorts candidate planetary trajectories.
    Sorting Order:
      1. Feasibility Class (FEASIBLE > MARGINAL > UNFEASIBLE)
      2. Composite Safety Index (CSI) descending
      3. Minimum Energy Reserve (SoC) descending
      4. Hazard Margin descending
    """
    def __init__(self, terrain: TerrainMap, probe: ProbeProfile):
        self.evaluator = PathEvaluator(terrain, probe)
        self.probe = probe
        self.terrain = terrain

    def evaluate_and_sort_paths(
        self,
        candidate_paths: List[Tuple[str, str, List[Tuple[float, float]]]]
    ) -> List[PathEvaluationReport]:
        """
        Evaluates a list of (path_id, name, waypoints) tuples and returns them
        strictly sorted from safest/most feasible to least feasible.
        """
        reports: List[PathEvaluationReport] = []
        for pid, name, waypoints in candidate_paths:
            report = self.evaluator.evaluate_path(pid, name, waypoints)
            reports.append(report)
            
        # Priority mapping for feasibility status
        status_priority = {
            PathFeasibilityStatus.FEASIBLE: 0,
            PathFeasibilityStatus.MARGINAL: 1,
            PathFeasibilityStatus.UNFEASIBLE: 2
        }
        
        # Multi-tier deterministic sorting key
        def sorting_key(rep: PathEvaluationReport):
            return (
                status_priority[rep.status],               # Feasible paths first
                -round(rep.composite_safety_index, 3),      # Highest composite score first
                -round(rep.minimum_soc_pct, 2),             # Highest battery safety margin
                round(rep.max_hazard_encountered, 3),       # Lowest peak hazard
                round(rep.max_slope_deg, 2)                 # Lowest peak slope
            )
            
        reports.sort(key=sorting_key)
        return reports


# ==============================================================================
# 7. MISSION REPORTING & VISUALIZATION
# ==============================================================================

def print_engineering_summary_table(reports: List[PathEvaluationReport], probe: ProbeProfile, body: PlanetaryBody):
    """Prints a structured engineering telemetry and sorting analysis table."""
    print("\n" + "=" * 115)
    print(f" PLANETARY EXPLORATION PROBE PATH FEASIBILITY & SAFETY EVALUATION REPORT")
    print(f" Target Body: {body.name} | Surface Gravity: {body.gravity} m/s² | Solar Flux Peak: {body.solar_flux_peak} W/m²")
    print(f" Probe: {probe.name} (Mass: {probe.mass_kg} kg | Max Slope: {probe.max_safe_slope_deg}° | Battery: {probe.battery_capacity_wh} Wh)")
    print("=" * 115)
    print(f"{'Rank':<5}{'Path Name':<28}{'Status':<16}{'Safety (CSI)':<14}{'Dist (m)':<10}{'Max Slope':<11}{'Max Haz':<9}{'Min SoC':<10}{'Comm %':<8}")
    print("-" * 115)
    
    for rank, rep in enumerate(reports, start=1):
        status_short = rep.status.name
        print(
            f"#{rank:<4}{rep.name[:26]:<28}{status_short:<16}"
            f"{rep.composite_safety_index:>8.2f} / 100  "
            f"{rep.total_distance_m:>7.1f}   "
            f"{rep.max_slope_deg:>6.1f}°    "
            f"{rep.max_hazard_encountered:>6.2f}   "
            f"{rep.minimum_soc_pct:>6.1f}%   "
            f"{rep.comm_coverage_pct:>5.1f}%"
        )
        
    print("=" * 115)
    print("\nDETAILED CONSTRAINT VERIFICATION & MISSION RISK AUDIT:")
    print("-" * 115)
    for rank, rep in enumerate(reports, start=1):
        print(f"\n[RANK #{rank}] -> {rep.name} (ID: {rep.path_id})")
        print(f"  * Feasibility Status: {rep.status.value}")
        print(f"  * Composite Safety Index: {rep.composite_safety_index:.2f} / 100.0")
        print(f"  * Traversal Time: {rep.total_time_hours:.2f} hrs | Elevation Gain: +{rep.total_elevation_gain_m:.1f} m")
        print(f"  * Energy Balance: Final SoC: {rep.final_soc_pct:.1f}% | Net Consumed: {rep.net_energy_consumed_wh:+.1f} Wh")
        print(f"  * Telemetry Line-of-Sight: {rep.comm_coverage_pct:.1f}% (Max blackout: {rep.max_comm_blackout_min:.1f} min)")
        print(f"  * Cumulative Radiation: {rep.cumulative_radiation_msv:.3f} mSv")
        
        if rep.failure_reasons:
            print("  * Safety Violations & Warnings:")
            for reason in rep.failure_reasons:
                print(f"      [!] {reason}")
        else:
            print("  * Safety Margin: ALL operational parameters are strictly within nominal flight envelope.")
            
    print("=" * 115)


def generate_mission_visualization_plot(
    terrain: TerrainMap,
    reports: List[PathEvaluationReport],
    output_filename: str = "probe_planetary_path_analysis.png"
):
    """
    Generates a 4-panel publication-grade aerospace engineering figure:
      1. Digital Elevation Model with Hazard Overlays & Trajectories
      2. Multi-Criteria Radar Comparison Chart
      3. Battery State-of-Charge Dynamics over Time
      4. Slope & Incline Safety Margin Profiles
    """
    fig = plt.figure(figsize=(18, 12), dpi=150)
    gs = GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.28)
    
    # Color palette for paths
    colors = ['#00E676', '#00B0FF', '#FFD600', '#FF5252', '#E040FB', '#FF6D00']
    
    # --------------------------------------------------------------------------
    # Panel 1: Topographic Map with Hazards and Paths
    # --------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    extent = [0, terrain.nx * terrain.resolution, 0, terrain.ny * terrain.resolution]
    
    # Elevation contours
    elev_contour = ax1.imshow(
        terrain.elevation,
        origin='lower',
        extent=extent,
        cmap='terrain',
        alpha=0.75
    )
    cbar1 = plt.colorbar(elev_contour, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Elevation (meters)", fontsize=10)
    
    # Hazard zones overlay (red hatch/alpha)
    hazard_masked = np.ma.masked_where(terrain.hazard_density < 0.35, terrain.hazard_density)
    ax1.imshow(hazard_masked, origin='lower', extent=extent, cmap='hot', alpha=0.35)
    
    # Plot candidate paths
    for idx, rep in enumerate(reports):
        col = colors[idx % len(colors)]
        wps = np.array(rep.waypoints)
        linestyle = '-' if rep.status == PathFeasibilityStatus.FEASIBLE else ('--' if rep.status == PathFeasibilityStatus.MARGINAL else ':')
        ax1.plot(
            wps[:, 0], wps[:, 1],
            label=f"#{idx+1}: {rep.name[:18]} (CSI: {rep.composite_safety_index:.1f})",
            color=col,
            linewidth=2.5 if idx == 0 else 1.8,
            linestyle=linestyle,
            marker='o' if len(wps) < 15 else None,
            markersize=4
        )
        
    # Mark Start and Goal
    start_pt = reports[0].waypoints[0]
    goal_pt = reports[0].waypoints[-1]
    ax1.scatter([start_pt[0]], [start_pt[1]], color='lime', edgecolors='black', s=160, zorder=6, label="Lander Drop Site")
    ax1.scatter([goal_pt[0]], [goal_pt[1]], color='gold', edgecolors='black', s=160, marker='*', zorder=6, label="Science Destination")
    
    ax1.set_title("Planetary Surface DEM & Trajectory Candidate Overlay", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Easting (meters)", fontsize=10)
    ax1.set_ylabel("Northing (meters)", fontsize=10)
    ax1.legend(loc='upper right', fontsize=8, framealpha=0.85)
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    # --------------------------------------------------------------------------
    # Panel 2: Multi-Criteria Radar / Spider Chart
    # --------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1], polar=True)
    categories = ['Slope Safety', 'Hazard Clearance', 'Energy Reserve', 'Comm Coverage', 'Path Efficiency']
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Complete loop
    
    for idx, rep in enumerate(reports):
        col = colors[idx % len(colors)]
        values = [
            rep.score_slope_safety,
            rep.score_hazard_clearance,
            rep.score_energy_reserve,
            rep.score_communication,
            rep.score_traversal_efficiency
        ]
        values += values[:1]
        ax2.plot(angles, values, color=col, linewidth=2.0, label=f"#{idx+1} {rep.name[:15]}")
        ax2.fill(angles, values, color=col, alpha=0.10)
        
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=9, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.set_title("Multi-Criteria Safety & Performance Radar (0-100)", fontsize=12, fontweight='bold', pad=15)
    ax2.legend(loc='lower right', bbox_to_anchor=(1.35, -0.05), fontsize=8)
    
    # --------------------------------------------------------------------------
    # Panel 3: Battery State-of-Charge Dynamics
    # --------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    for idx, rep in enumerate(reports):
        if rep.telemetry and len(rep.telemetry.times_sec) > 0:
            col = colors[idx % len(colors)]
            time_hours = [t / 3600.0 for t in rep.telemetry.times_sec]
            ax3.plot(
                time_hours,
                rep.telemetry.soc_history,
                label=f"#{idx+1} {rep.name[:18]}",
                color=col,
                linewidth=2.0
            )
            
    # Critical threshold line
    crit_soc = reports[0].minimum_soc_pct # baseline from probe config
    ax3.axhline(25.0, color='red', linestyle='--', linewidth=1.8, label="Critical Brownout Limit (25%)")
    ax3.set_title("Battery State-of-Charge (SoC %) Traversal Timeline", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Mission Elapsed Time (hours)", fontsize=10)
    ax3.set_ylabel("Battery SoC (%)", fontsize=10)
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(loc='lower left', fontsize=8)
    
    # --------------------------------------------------------------------------
    # Panel 4: Incline & Slope Safety Profile
    # --------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    for idx, rep in enumerate(reports):
        if rep.telemetry and len(rep.telemetry.distances_m) > 0:
            col = colors[idx % len(colors)]
            ax4.plot(
                rep.telemetry.distances_m,
                rep.telemetry.slopes_deg,
                label=f"#{idx+1} {rep.name[:18]}",
                color=col,
                linewidth=1.8,
                alpha=0.85
            )
            
    ax4.axhline(22.0, color='orange', linestyle='--', linewidth=1.5, label="Safe Operating Incline (22°)")
    ax4.axhline(28.0, color='red', linestyle='-', linewidth=2.0, label="Hard Rollover Limit (28°)")
    ax4.set_title("Terrain Incline Profile & Rollover Hazard Envelope", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Traversed Distance (meters)", fontsize=10)
    ax4.set_ylabel("Slope Angle (degrees)", fontsize=10)
    ax4.set_ylim(0, 36)
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend(loc='upper right', fontsize=8)
    
    plt.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.08, wspace=0.28, hspace=0.32)
    plt.savefig(output_filename, dpi=180, bbox_inches='tight')
    plt.close()
    print(f"\n[+] High-Resolution Mission Visualization Plot saved to: {output_filename}")


# ==============================================================================
# 8. COMPLETE SIMULATION & DEMO EXECUTION
# ==============================================================================

def run_planetary_probe_demo():
    """
    Executes a comprehensive scenario:
      1. Initializes Mars Jezero Crater environment.
      2. Configures a scientific rover probe profile.
      3. Computes and generates multiple diverse candidate paths:
         - Candidate A: Direct Euclidean Route (Dangerous crater crossing)
         - Candidate B: Crater Deep Basin Descent (Severe shadow & rock traps)
         - Candidate C: East Ridge Bypass (Longer perimeter detour)
         - Candidate D: Autonomous Multi-Objective Optimal Safe Path (A*)
         - Candidate E: West Sunlit Plateau Route
      4. Sorts all candidate paths by safety and feasibility.
      5. Generates aerospace report and visualization.
    """
    print("[*] Initializing Planetary Environment (Mars - Jezero Crater Analog)...")
    body = MARS
    terrain = TerrainMap(
        grid_size_x=100,
        grid_size_y=100,
        cell_resolution_m=5.0, # 500m x 500m exploration sector
        body=body,
        seed=101
    )
    
    print("[*] Configuring Planetary Probe Engineering Specification...")
    probe = ProbeProfile(
        name="ARES-IV Exploration Rover",
        mass_kg=890.0,
        max_safe_slope_deg=22.0,
        hard_limit_slope_deg=28.0,
        max_hazard_tolerance=0.60,
        cruise_speed_mps=0.15,
        battery_capacity_wh=2200.0,
        initial_soc=0.92,
        critical_soc_threshold=0.25,
        solar_panel_area_m2=2.2,
        solar_efficiency=0.30,
        base_avionics_power_w=85.0
    )
    
    # Define Start and Goal in metric coordinates
    start_coord = (50.0, 60.0)      # Safe flat drop zone in South-West
    goal_coord = (440.0, 420.0)    # Science sample target in North-East
    
    planner = PlanetaryPathPlanner(terrain, probe)
    
    print("[*] Synthesizing Candidate Navigation Trajectories...")
    
    # Path 1: Direct Straight-Line Route (Cuts straight through central crater and rock fields)
    path_direct = [
        start_coord,
        (150.0, 150.0),
        (250.0, 240.0),
        (350.0, 330.0),
        goal_coord
    ]
    
    # Path 2: Crater Deep Basin Crossing (Descends into deep shadow & high boulder hazard)
    path_crater_basin = [
        start_coord,
        (120.0, 180.0),
        (180.0, 240.0), # Crater center (deep shadow & steep walls)
        (240.0, 300.0),
        (360.0, 380.0),
        goal_coord
    ]
    
    # Path 3: East Ridge Bypass (Circumnavigates crater around the eastern rim)
    path_east_ridge = [
        start_coord,
        (160.0, 80.0),
        (300.0, 100.0),
        (380.0, 200.0),
        (420.0, 320.0),
        goal_coord
    ]
    
    # Path 4: West Sunlit Plateau Route (Hugs the sunlit western hills for maximum solar power)
    path_west_plateau = [
        start_coord,
        (60.0, 180.0),
        (70.0, 300.0),
        (140.0, 400.0),
        (280.0, 450.0),
        goal_coord
    ]
    
    # Path 5: Multi-Objective Optimal Safe Path (A* algorithm minimizing hazards & slopes)
    print("    -> Computing Multi-Objective Terrain-Optimal A* Path...")
    path_ast_safe = planner.find_optimal_safe_path(
        start_coord,
        goal_coord,
        hazard_weight=10.0,
        slope_weight=8.0,
        energy_weight=4.0
    )
    
    candidate_paths = [
        ("TRAJ-001", "Direct Euclidean Route", path_direct),
        ("TRAJ-002", "Crater Deep Basin Descent", path_crater_basin),
        ("TRAJ-003", "East Ridge Circumvention", path_east_ridge),
        ("TRAJ-004", "West Sunlit Plateau Route", path_west_plateau),
    ]
    if path_ast_safe:
        candidate_paths.append(("TRAJ-005", "Weighted A* Candidate", path_ast_safe))
    else:
        print("[!] A* found no route satisfying the planner's slope and hazard limits.")
    
    print(f"[*] Evaluating and Sorting {len(candidate_paths)} Candidate Trajectories...")
    sorter = ProbePathSorter(terrain, probe)
    sorted_reports = sorter.evaluate_and_sort_paths(candidate_paths)
    
    # Output aerospace tables
    print_engineering_summary_table(sorted_reports, probe, body)
    
    # Output visual figure
    generate_mission_visualization_plot(
        terrain,
        sorted_reports,
        output_filename="probe_planetary_path_analysis.png"
    )
    
    return sorted_reports


if __name__ == "__main__":
    run_planetary_probe_demo()
