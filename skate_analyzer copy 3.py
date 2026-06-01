#!/usr/bin/env python3
"""
Freeskating Motion Analysis with Automatic Rest Detection & Reset
==================================================================

Detects rest periods throughout the session and applies zero-velocity 
updates at each one. More robust than single initial correction.

Usage:
    python freeskate_analyzer.py skate_data.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, find_peaks
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter1d
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class SkateConfig:
    gravity: float = 9.81  # m/s² for physics, but we don't use it for rest detection anymore
    acc_cutoff_hz: float = 10.0
    
    # Rest = low dynamic acceleration
    rest_acceleration_threshold: float = 0.3  # G units - adjust this!
    min_rest_duration: float = 2.0
    rest_smoothing_window: float = 0.2
    
    correction_method: str = 'linear_interp'
    colormap: str = 'plasma'
    line_width: float = 2.0


class OrientationUtils:
    @staticmethod
    def euler_to_rotation_matrix(pitch: float, yaw: float, roll: float, degrees: bool = True) -> np.ndarray:
        if degrees:
            pitch, yaw, roll = np.radians([pitch, yaw, roll])
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        return Rz @ Ry @ Rx
    
    @staticmethod
    def rotate_acceleration(acc_body: np.ndarray, R: np.ndarray) -> np.ndarray:
        return R @ acc_body


class AdaptiveDriftCorrector:
    """
    Detect rest periods throughout data and apply corrections.
    """
    
    def __init__(self, timestamps: np.ndarray, config: SkateConfig):
        self.timestamps = timestamps
        self.dt = timestamps[1] - timestamps[0]
        self.config = config
        self.rest_periods: List[Tuple[int, int]] = []  # (start_idx, end_idx)
        
    def detect_rest_periods(self, acc_world: np.ndarray) -> List[Tuple[int, int]]:
        print("Detecting rest periods...")
        
        acc_mag = np.sqrt(np.sum(acc_world**2, axis=1))
        
        print(f"  Accel magnitude range: [{acc_mag.min():.3f}, {acc_mag.max():.3f}]")
        
        # FIXED: Rest = LOW acceleration (near 0), not near gravity
        # Threshold should be above noise floor but below motion
        rest_threshold = 0.3  # G units - adjust based on your data
        
        # Rest when acceleration is below threshold
        is_rest = acc_mag < rest_threshold
        
        print(f"  Rest threshold: {rest_threshold}")
        print(f"  Samples below threshold: {np.sum(is_rest)} / {len(is_rest)}")
        
        # Smooth to remove single-sample spikes
        window = max(3, int(self.config.rest_smoothing_window / self.dt))
        is_rest_smooth = uniform_filter1d(is_rest.astype(float), size=window) > 0.5
        
        # Require minimum duration
        min_samples = int(self.config.min_rest_duration / self.dt)
        
        # Find continuous rest periods
        rest_periods = []
        in_rest = False
        start_idx = 0
        
        for i, rest in enumerate(is_rest_smooth):
            if rest and not in_rest:
                start_idx = i
                in_rest = True
            elif not rest and in_rest:
                if i - start_idx >= min_samples:
                    rest_periods.append((start_idx, i))
                    print(f"    Rest: {self.timestamps[start_idx]:.1f}s - {self.timestamps[i]:.1f}s "
                        f"({(i-start_idx)*self.dt:.1f}s)")
                in_rest = False
        
        # Handle data ending during rest
        if in_rest and len(acc_mag) - start_idx >= min_samples:
            rest_periods.append((start_idx, len(acc_mag)))
            print(f"    Rest: {self.timestamps[start_idx]:.1f}s - end "
                f"({(len(acc_mag)-start_idx)*self.dt:.1f}s)")
        
        self.rest_periods = rest_periods
        print(f"  Total rest periods found: {len(rest_periods)}")
        
        return rest_periods
    
    def apply_corrections(self, velocity: np.ndarray) -> np.ndarray:
        """
        Apply drift correction using detected rest periods.
        """
        print(f"Applying {self.config.correction_method} correction...")
        
        if not self.rest_periods:
            print("  No rest periods found! Skipping correction.")
            return velocity
        
        v_corrected = velocity.copy()
        
        if self.config.correction_method == 'hard_reset':
            # Simple: force velocity to zero at each rest period
            for start, end in self.rest_periods:
                v_corrected[start:end] = 0
                
        elif self.config.correction_method == 'linear_interp':
            # Linear interpolation between rest periods
            # At each rest, velocity should be zero
            
            # Mark all rest samples
            is_rest = np.zeros(len(velocity), dtype=bool)
            for start, end in self.rest_periods:
                is_rest[start:end] = True
            
            # For each axis, interpolate drift between rests
            for axis in range(3):
                v_axis = velocity[:, axis].copy()
                
                # At rest periods, drift = measured velocity (should be zero, but isn't due to drift)
                # So we subtract this drift
                
                # Find drift values at rest periods (use median of each rest)
                drift_points_t = []
                drift_points_v = []
                
                for start, end in self.rest_periods:
                    t_mid = self.timestamps[(start + end) // 2]
                    v_drift = np.median(v_axis[start:end])  # Measured velocity = drift
                    drift_points_t.append(t_mid)
                    drift_points_v.append(v_drift)
                
                # Add start and end points for extrapolation
                if drift_points_t[0] > self.timestamps[0]:
                    drift_points_t.insert(0, self.timestamps[0])
                    drift_points_v.insert(0, drift_points_v[0])  # Assume same drift at start
                
                if drift_points_t[-1] < self.timestamps[-1]:
                    drift_points_t.append(self.timestamps[-1])
                    drift_points_v.append(drift_points_v[-1])  # Assume same drift at end
                
                # Interpolate drift across entire timeline
                drift = np.interp(self.timestamps, drift_points_t, drift_points_v)
                
                # Subtract drift
                v_corrected[:, axis] = v_axis - drift
                
                print(f"  Axis {axis}: drift range [{min(drift_points_v):.3f}, {max(drift_points_v):.3f}] m/s")
        
        elif self.config.correction_method == 'spline':
            # Smooth spline fit through rest points
            from scipy.interpolate import UnivariateSpline
            
            for axis in range(3):
                drift_points_t = []
                drift_points_v = []
                
                for start, end in self.rest_periods:
                    t_mid = self.timestamps[(start + end) // 2]
                    v_drift = np.median(velocity[start:end, axis])
                    drift_points_t.append(t_mid)
                    drift_points_v.append(v_drift)
                
                # Fit spline
                spline = UnivariateSpline(drift_points_t, drift_points_v, s=len(drift_points_t))
                drift = spline(self.timestamps)
                v_corrected[:, axis] = velocity[:, axis] - drift
        
        return v_corrected
    
    def get_rest_mask(self) -> np.ndarray:
        """Return boolean mask of rest periods."""
        mask = np.zeros(len(self.timestamps), dtype=bool)
        for start, end in self.rest_periods:
            mask[start:end] = True
        return mask


class SkateAnalyzer:
    def __init__(self, config: SkateConfig = None):
        self.config = config or SkateConfig()
        self.df = None
        self.timestamps = None
        self.dt = None
        self.positions = None
        self.velocities = None
        self.world_accelerations = None
        self.drift_corrector = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        required = ['timestamp', 'pitch', 'yaw', 'roll', 'accel_x', 'accel_y', 'accel_z']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Convert milliseconds to seconds
        timestamps = df['timestamp'].values.astype(float)
        df['time_sec'] = (timestamps - timestamps[0]) / 1e3
        
        self.df = df
        self.timestamps = df['time_sec'].values
        self.dt = np.median(np.diff(self.timestamps))
        
        print(f"Loaded {len(df)} samples at {1/self.dt:.1f} Hz")
        print(f"Duration: {self.timestamps[-1]:.2f} seconds")
        
        return df
    
    def lowpass_filter(self, data: np.ndarray, cutoff: float) -> np.ndarray:
        nyquist = 1 / (2 * self.dt)
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0 or normal_cutoff <= 0:
            return data
        b, a = butter(4, normal_cutoff, btype='low')
        return filtfilt(b, a, data, axis=0)
    
    def process_accelerations(self) -> np.ndarray:
        print("Processing accelerations...")
        
        n = len(self.df)
        acc_world = np.zeros((n, 3))
        
        for i in range(n):
            R = OrientationUtils.euler_to_rotation_matrix(
                self.df['pitch'].iloc[i],
                self.df['yaw'].iloc[i], 
                self.df['roll'].iloc[i]
            )
            acc_body = np.array([
                self.df['accel_x'].iloc[i],
                self.df['accel_y'].iloc[i],
                self.df['accel_z'].iloc[i]
            ])
            acc_world[i] = OrientationUtils.rotate_acceleration(acc_body, R)
        
        # Remove gravity
        gravity_estimate = np.median(acc_world[:, 2])
        acc_world[:, 2] -= gravity_estimate
        
        # Filter after gravity removal
        for i in range(3):
            acc_world[:, i] = self.lowpass_filter(acc_world[:, i], self.config.acc_cutoff_hz)
        
        self.world_accelerations = acc_world
        return acc_world
    
    def integrate_to_velocity(self) -> np.ndarray:
        print("Integrating to velocity...")
        
        velocity = np.zeros_like(self.world_accelerations)
        for i in range(3):
            velocity[:, i] = cumulative_trapezoid(
                self.world_accelerations[:, i], 
                self.timestamps, 
                initial=0
            )
        
        self.velocities = velocity
        return velocity
    
    def apply_adaptive_drift_correction(self) -> np.ndarray:
        """
        Detect rests and apply correction.
        """
        self.drift_corrector = AdaptiveDriftCorrector(self.timestamps, self.config)
        self.drift_corrector.detect_rest_periods(self.world_accelerations)
        v_corrected = self.drift_corrector.apply_corrections(self.velocities)
        self.velocities = v_corrected
        return v_corrected
    
    def integrate_to_position(self) -> np.ndarray:
        print("Integrating to position...")
        
        position = np.zeros_like(self.velocities)
        for i in range(3):
            position[:, i] = cumulative_trapezoid(
                self.velocities[:, i],
                self.timestamps,
                initial=0
            )
        
        self.positions = position
        return position
    
    def calculate_statistics(self) -> dict:
        vel_mag = np.sqrt(np.sum(self.velocities**2, axis=1))
        rest_mask = self.drift_corrector.get_rest_mask()
        
        return {
            'duration_sec': self.timestamps[-1],
            'rest_time_sec': np.sum(rest_mask) * self.dt,
            'rest_percentage': np.mean(rest_mask) * 100,
            'num_rests': len(self.drift_corrector.rest_periods),
            'total_distance_m': np.sum(np.sqrt(np.sum(np.diff(self.positions, axis=0)**2, axis=1))),
            'max_velocity_ms': np.max(vel_mag),
            'avg_velocity_ms': np.mean(vel_mag[~rest_mask]) if not np.all(rest_mask) else 0,
            'max_acceleration_ms2': np.max(np.sqrt(np.sum(self.world_accelerations**2, axis=1))),
            'height_range_m': np.max(self.positions[:, 2]) - np.min(self.positions[:, 2]),
        }
    
    def analyze(self) -> 'SkateAnalyzer':
        self.process_accelerations()
        self.integrate_to_velocity()
        self.apply_adaptive_drift_correction()
        self.integrate_to_position()
        return self


class Visualizer:
    def __init__(self, analyzer: SkateAnalyzer):
        self.analyzer = analyzer
        self.config = analyzer.config
        
    def plot_rest_detection(self, save_path: Optional[str] = None):
        """Visualize how rest periods were detected."""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        t = self.analyzer.timestamps
        acc = self.analyzer.world_accelerations
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        vel = self.analyzer.velocities
        rest_mask = self.analyzer.drift_corrector.get_rest_mask()
        
        # Plot 1: Acceleration magnitude with rest periods
        ax1 = axes[0]
        ax1.plot(t, acc_mag, 'b-', alpha=0.7, label='|Acc|')
        ax1.axhline(y=self.config.gravity, color='k', linestyle='--', alpha=0.5, label='Expected g')
        ax1.axhline(y=self.config.gravity + self.config.rest_acceleration_threshold, 
                   color='r', linestyle='--', alpha=0.5, label='Rest threshold')
        ax1.fill_between(t, 0, 50, where=rest_mask, alpha=0.2, color='green', label='Detected rest')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title('Rest Detection: Acceleration')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Raw vs Corrected Velocity
        ax2 = axes[1]
        # Re-integrate to show raw
        acc = self.analyzer.world_accelerations
        v_raw = np.zeros_like(acc)
        for i in range(3):
            v_raw[:, i] = cumulative_trapezoid(acc[:, i], t, initial=0)
        v_raw_mag = np.sqrt(np.sum(v_raw**2, axis=1))
        v_corr_mag = np.sqrt(np.sum(self.analyzer.velocities**2, axis=1))
        
        ax2.plot(t, v_raw_mag, 'r--', alpha=0.5, label='Raw (uncorrected)')
        ax2.plot(t, v_corr_mag, 'g-', linewidth=2, label='Corrected')
        ax2.fill_between(t, 0, v_raw_mag.max(), where=rest_mask, alpha=0.2, color='green')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.set_title('Velocity: Raw vs Corrected')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Velocity components
        ax3 = axes[2]
        ax3.plot(t, self.analyzer.velocities[:, 0], 'r-', label='Vx', alpha=0.7)
        ax3.plot(t, self.analyzer.velocities[:, 1], 'g-', label='Vy', alpha=0.7)
        ax3.plot(t, self.analyzer.velocities[:, 2], 'b-', label='Vz', alpha=0.7)
        ax3.fill_between(t, -10, 10, where=rest_mask, alpha=0.2, color='green')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Velocity (m/s)')
        ax3.set_title('Corrected Velocity Components')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved rest detection plot to {save_path}")
        
        plt.show()
        return fig
    
    def plot_3d_trajectory(self, save_path: Optional[str] = None):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        pos = self.analyzer.positions
        vel = np.sqrt(np.sum(self.analyzer.velocities**2, axis=1))
        rest_mask = self.analyzer.drift_corrector.get_rest_mask()
        
        norm = plt.Normalize(vel.min(), vel.max())
        cmap = plt.get_cmap(self.config.colormap)
        
        # Plot motion segments (colored by velocity)
        motion_idx = np.where(~rest_mask)[0]
        for i in range(len(motion_idx) - 1):
            idx = motion_idx[i]
            if idx + 1 < len(pos):
                color = cmap(norm(vel[idx]))
                ax.plot3D(pos[idx:idx+2, 0], pos[idx:idx+2, 1], pos[idx:idx+2, 2], 
                         color=color, linewidth=self.config.line_width)
        
        # Plot rest periods in gray
        for start, end in self.analyzer.drift_corrector.rest_periods:
            ax.plot3D(pos[start:end, 0], pos[start:end, 1], pos[start:end, 2], 
                     'gray', linewidth=4, alpha=0.5)
        
        ax.scatter(*pos[0], color='green', s=100, marker='o', label='Start')
        ax.scatter(*pos[-1], color='red', s=100, marker='s', label='End')
        
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(vel)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
        cbar.set_label('Velocity (m/s)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Freeskating 3D Trajectory (gray = rest periods)')
        ax.legend()
        
        # Equal aspect
        max_range = np.array([pos[:, i].max() - pos[:, i].min() for i in range(3)]).max() / 2.0
        mid = [(pos[:, i].max() + pos[:, i].min()) * 0.5 for i in range(3)]
        if max_range > 0:
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D plot to {save_path}")
        
        plt.show()
        return fig
    
    def plot_analysis_dashboard(self, save_path: Optional[str] = None):
        fig = plt.figure(figsize=(16, 12))
        
        t = self.analyzer.timestamps
        pos = self.analyzer.positions
        vel = self.analyzer.velocities
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        acc = self.analyzer.world_accelerations
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        rest_mask = self.analyzer.drift_corrector.get_rest_mask()
        
        # 1. 3D Trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=1, alpha=0.7)
        ax1.scatter(*pos[0], color='green', s=50, label='Start')
        ax1.scatter(*pos[-1], color='red', s=50, label='End')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # 2. X-Y view with rest periods
        ax2 = fig.add_subplot(2, 3, 2)
        motion = ~rest_mask
        ax2.scatter(pos[motion, 0], pos[motion, 1], c=vel_mag[motion], cmap='plasma', s=10)
        ax2.scatter(pos[rest_mask, 0], pos[rest_mask, 1], c='gray', s=20, alpha=0.5, label='Rest')
        ax2.plot(pos[0, 0], pos[0, 1], 'go', markersize=10)
        ax2.plot(pos[-1, 0], pos[-1, 1], 'rs', markersize=10)
        ax2.set_aspect('equal')
        ax2.set_title('Top-Down View')
        
        # 3. Height
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.fill_between(t, pos[:, 2], alpha=0.3)
        ax3.plot(t, pos[:, 2], 'b-', linewidth=1)
        ax3.fill_between(t, pos[:, 2].min(), pos[:, 2].max(), where=rest_mask, alpha=0.2, color='gray')
        ax3.set_title('Vertical Profile')
        ax3.set_ylabel('Height (m)')
        
        # 4. Velocity
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(t, vel[:, 0], 'r-', label='Vx', alpha=0.7)
        ax4.plot(t, vel[:, 1], 'g-', label='Vy', alpha=0.7)
        ax4.plot(t, vel[:, 2], 'b-', label='Vz', alpha=0.7)
        ax4.plot(t, vel_mag, 'k-', label='|V|', linewidth=2)
        ax4.fill_between(t, -10, 50, where=rest_mask, alpha=0.2, color='gray')
        ax4.set_title('Velocity')
        ax4.legend()
        
        # 5. Acceleration
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(t, acc[:, 0], 'r-', alpha=0.7)
        ax5.plot(t, acc[:, 1], 'g-', alpha=0.7)
        ax5.plot(t, acc[:, 2], 'b-', alpha=0.7)
        ax5.plot(t, acc_mag, 'k-', linewidth=2)
        ax5.fill_between(t, 0, 50, where=rest_mask, alpha=0.2, color='gray')
        ax5.set_title('Acceleration')
        
        # 6. Orientation
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(t, self.analyzer.df['pitch'], 'r-', label='Pitch')
        ax6.plot(t, self.analyzer.df['yaw'], 'g-', label='Yaw')
        ax6.plot(t, self.analyzer.df['roll'], 'b-', label='Roll')
        ax6.fill_between(t, -200, 200, where=rest_mask, alpha=0.2, color='gray')
        ax6.set_title('Orientation')
        ax6.legend()
        
        plt.suptitle('Freeskating Analysis (Auto Rest Detection)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved dashboard to {save_path}")
        
        plt.show()
        return fig
    
    def print_statistics(self):
        stats = self.analyzer.calculate_statistics()
        print("\n" + "="*50)
        print("FREESKATING SESSION STATISTICS")
        print("="*50)
        print(f"Duration:           {stats['duration_sec']:.2f} s")
        print(f"Rest periods:       {stats['num_rests']} ({stats['rest_time_sec']:.1f}s / {stats['rest_percentage']:.1f}%)")
        print(f"Total Distance:     {stats['total_distance_m']:.2f} m")
        print(f"Max Velocity:       {stats['max_velocity_ms']:.2f} m/s ({stats['max_velocity_ms']*3.6:.1f} km/h)")
        print(f"Avg Velocity:       {stats['avg_velocity_ms']:.2f} m/s (during motion)")
        print(f"Max Acceleration:   {stats['max_acceleration_ms2']:.2f} m/s²")
        print(f"Height Variation:   {stats['height_range_m']:.3f} m")
        print("="*50)


def main():
    if len(sys.argv) < 2:
        filepath = "skate_data.csv"
        print(f"Usage: python {sys.argv[0]} <csv_file>")
        print(f"Using default: {filepath}")
    else:
        filepath = sys.argv[1]
    
    # Configure
    config = SkateConfig(
        rest_acceleration_threshold=0.5,
        min_rest_duration=0.5,
        correction_method='linear_interp'  # 'hard_reset', 'linear_interp', or 'spline'
    )
    
    analyzer = SkateAnalyzer(config)
    
    try:
        analyzer.load_data(filepath)
        analyzer.analyze()
        
        viz = Visualizer(analyzer)
        viz.print_statistics()
        
        print("\nGenerating visualizations...")
        viz.plot_rest_detection("rest_detection.png")
        viz.plot_3d_trajectory("freeskate_3d.png")
        viz.plot_analysis_dashboard("freeskate_dashboard.png")
        
        # Save data
        output_df = pd.DataFrame({
            'time_sec': analyzer.timestamps,
            'pos_x': analyzer.positions[:, 0],
            'pos_y': analyzer.positions[:, 1],
            'pos_z': analyzer.positions[:, 2],
            'vel_x': analyzer.velocities[:, 0],
            'vel_y': analyzer.velocities[:, 1],
            'vel_z': analyzer.velocities[:, 2],
            'vel_mag': np.sqrt(np.sum(analyzer.velocities**2, axis=1)),
            'is_rest': analyzer.drift_corrector.get_rest_mask(),
        })
        output_df.to_csv("freeskate_processed.csv", index=False)
        print("\nSaved to freeskate_processed.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()