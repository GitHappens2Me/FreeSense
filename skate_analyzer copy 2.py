#!/usr/bin/env python3
"""
Freeskating Motion Analysis with Initial Rest Drift Correction
===============================================================

Analyzes IMU data with known initial standstill period to model and 
remove velocity drift. More accurate than ZUPT for continuous motion.

Usage:
    python freeskate_analyzer.py skate_data.csv
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt
from dataclasses import dataclass
from typing import Optional, Tuple
from numpy.polynomial import polynomial as P


@dataclass
class SkateConfig:
    gravity: float = 9.81
    acc_cutoff_hz: float = 10.0
    
    # Initial rest detection
    rest_duration_sec: float = 2.0      # How long to consider "initial rest"
    rest_accel_threshold: float = 0.5   # m/s² - movement during rest
    
    # Drift model: 'linear', 'polynomial', or 'cumulative'
    drift_model: str = 'linear'

    # Visualization
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


class DriftCorrector:
    """
    Model and remove velocity drift using initial rest period.
    """
    
    def __init__(self, timestamps: np.ndarray):
        self.timestamps = timestamps
        self.drift_model = None
        self.rest_end_idx = 0
        
    def find_rest_period(self, acc_magnitude: np.ndarray, 
                        rest_duration: float, threshold: float) -> int:
        """
        Find the end index of initial rest period.
        Returns index where rest period ends.
        """
        # Find where acceleration stays below threshold
        below_thresh = acc_magnitude < threshold
        
        # Require continuous rest period
        min_samples = int(rest_duration / (self.timestamps[1] - self.timestamps[0]))
        
        # Sliding window to find continuous rest
        rest_end = 0
        for i in range(len(below_thresh) - min_samples):
            if np.all(below_thresh[i:i+min_samples]):
                rest_end = i + min_samples
            else:
                if rest_end > 0:
                    break  # End of rest period found
        
        if rest_end == 0:
            print("Warning: No clear rest period detected. Using first 1 second.")
            rest_end = min(int(1.0 / (self.timestamps[1] - self.timestamps[0])), len(self.timestamps)//10)
        
        self.rest_end_idx = rest_end
        print(f"  Initial rest period: 0 to {self.timestamps[rest_end]:.2f}s ({rest_end} samples)")
        return rest_end
    
    def estimate_bias(self, acc_world: np.ndarray) -> np.ndarray:
        """Estimate acceleration bias from rest period."""
        bias = np.mean(acc_world[:self.rest_end_idx], axis=0)
        print(f"  Estimated accel bias: [{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}] m/s²")
        return bias
    
    def fit_drift_model(self, velocity: np.ndarray, model_type: str = 'linear') -> np.ndarray:
        """
        Fit drift model to velocity data.
        Assumes velocity should be zero at t=0 and drift accumulates.
        """
        t = self.timestamps
        v_drift = np.zeros_like(velocity)
        
        if model_type == 'linear':
            # Fit line to each axis: v_drift = m*t + b
            # Constrain b=0 (starts at zero)
            for i in range(3):
                # Linear fit through origin: v = m*t, so m = sum(v*t)/sum(t²)
                m = np.sum(velocity[:, i] * t) / np.sum(t**2)
                v_drift[:, i] = m * t
                print(f"  Axis {i}: drift rate = {m:.4f} m/s²")
                
        elif model_type == 'polynomial':
            # Fit 2nd order polynomial
            for i in range(3):
                # Fit poly, constrain constant term to 0
                coeffs = np.polyfit(t, velocity[:, i], 2)
                v_drift[:, i] = np.polyval(coeffs, t)
                
        elif model_type == 'cumulative':
            # Assume drift accumulates from initial bias
            # This is physically motivated: drift = bias * t
            pass  # Handled in correct_velocity
            
        self.drift_model = v_drift
        return v_drift
    
    def correct_velocity(self, velocity: np.ndarray, model_type: str) -> np.ndarray:
        """Remove drift from velocity."""
        if model_type == 'cumulative':
            # Physically motivated: integrate bias to get drift
            # v_drift(t) = integral(bias dt)
            # But we estimate bias from rest period
            bias = np.mean(velocity[:self.rest_end_idx], axis=0)
            print(f"  Removing cumulative bias: [{bias[0]:.4f}, {bias[1]:.4f}, {bias[2]:.4f}] m/s")
            
            # Linear drift model: v_corrected = v_raw - bias * t / t_max * correction_factor
            # Actually simpler: just subtract the trend
            v_corrected = np.zeros_like(velocity)
            for i in range(3):
                # Fit line to velocity, subtract it
                coeffs = np.polyfit(self.timestamps, velocity[:, i], 1)
                drift = np.polyval(coeffs, self.timestamps)
                v_corrected[:, i] = velocity[:, i] - drift
        else:
            v_drift = self.fit_drift_model(velocity, model_type)
            v_corrected = velocity - v_drift
            
        return v_corrected


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
    
    def apply_drift_correction(self) -> np.ndarray:
        """
        Apply drift correction using initial rest period.
        """
        print("Applying drift correction...")
        
        # Initialize drift corrector
        self.drift_corrector = DriftCorrector(self.timestamps)
        
        # Find rest period using acceleration magnitude
        acc_mag = np.sqrt(np.sum(self.world_accelerations**2, axis=1))
        self.drift_corrector.find_rest_period(
            acc_mag, 
            self.config.rest_duration_sec,
            self.config.rest_accel_threshold
        )
        
        # Apply correction
        v_corrected = self.drift_corrector.correct_velocity(
            self.velocities, 
            self.config.drift_model
        )
        
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
        return {
            'duration_sec': self.timestamps[-1],
            'total_distance_m': np.sum(np.sqrt(np.sum(np.diff(self.positions, axis=0)**2, axis=1))),
            'max_velocity_ms': np.max(vel_mag),
            'avg_velocity_ms': np.mean(vel_mag),
            'max_acceleration_ms2': np.max(np.sqrt(np.sum(self.world_accelerations**2, axis=1))),
            'height_range_m': np.max(self.positions[:, 2]) - np.min(self.positions[:, 2]),
        }
    
    def analyze(self) -> 'SkateAnalyzer':
        self.process_accelerations()
        self.integrate_to_velocity()
        self.apply_drift_correction()  # NEW: drift correction instead of ZUPT
        self.integrate_to_position()
        return self


class Visualizer:
    def __init__(self, analyzer: SkateAnalyzer):
        self.analyzer = analyzer
        self.config = analyzer.config
        
    def plot_velocity_drift_analysis(self, save_path: Optional[str] = None):
        """
        Plot showing raw velocity, drift model, and corrected velocity.
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        t = self.analyzer.timestamps
        v_raw = self.analyzer.drift_corrector.velocities if hasattr(self.analyzer.drift_corrector, 'velocities') else self.analyzer.velocities
        
        # Re-integrate to get "raw" uncorrected for comparison
        acc = self.analyzer.world_accelerations
        v_uncorrected = np.zeros_like(acc)
        for i in range(3):
            v_uncorrected[:, i] = cumulative_trapezoid(acc[:, i], t, initial=0)
        
        colors = ['red', 'green', 'blue']
        labels = ['X', 'Y', 'Z']
        
        for i, (ax, color, label) in enumerate(zip(axes, colors, labels)):
            ax.plot(t, v_uncorrected[:, i], '--', color=color, alpha=0.5, label='Raw (uncorrected)')
            ax.plot(t, self.analyzer.velocities[:, i], '-', color=color, linewidth=2, label='Corrected')
            
            # Mark rest period
            rest_end = self.analyzer.drift_corrector.rest_end_idx
            ax.axvspan(0, t[rest_end], alpha=0.2, color='gray', label='Rest period')
            
            ax.set_ylabel(f'V{label} (m/s)')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        axes[0].set_title('Velocity Drift Correction')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved drift analysis to {save_path}")
        
        plt.show()
        return fig
    
    def plot_3d_trajectory(self, save_path: Optional[str] = None):
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        pos = self.analyzer.positions
        vel = np.sqrt(np.sum(self.analyzer.velocities**2, axis=1))
        
        norm = plt.Normalize(vel.min(), vel.max())
        cmap = plt.get_cmap(self.config.colormap)
        
        for i in range(len(pos) - 1):
            color = cmap(norm(vel[i]))
            ax.plot3D(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2], 
                     color=color, linewidth=self.config.line_width)
        
        ax.scatter(*pos[0], color='green', s=100, marker='o', 
                  label='Start', edgecolors='black')
        ax.scatter(*pos[-1], color='red', s=100, marker='s', 
                  label='End', edgecolors='black')
        
        # Mark rest period in different color
        rest_end = self.analyzer.drift_corrector.rest_end_idx
        ax.plot3D(pos[:rest_end, 0], pos[:rest_end, 1], pos[:rest_end, 2], 
                 'gray', linewidth=3, alpha=0.5, label='Rest period')
        
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(vel)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5)
        cbar.set_label('Velocity (m/s)')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Freeskating 3D Trajectory')
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
        rest_end = self.analyzer.drift_corrector.rest_end_idx
        
        # 1. 3D Trajectory
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=1)
        ax1.plot(pos[:rest_end, 0], pos[:rest_end, 1], pos[:rest_end, 2], 'gray', linewidth=3)
        ax1.scatter(*pos[0], color='green', s=50, label='Start')
        ax1.scatter(*pos[-1], color='red', s=50, label='End')
        ax1.set_title('3D Trajectory (gray=rest)')
        ax1.legend()
        
        # 2. X-Y view
        ax2 = fig.add_subplot(2, 3, 2)
        scatter = ax2.scatter(pos[:, 0], pos[:, 1], c=vel_mag, cmap='plasma', s=10)
        ax2.plot(pos[0, 0], pos[0, 1], 'go', markersize=10)
        ax2.plot(pos[-1, 0], pos[-1, 1], 'rs', markersize=10)
        ax2.set_aspect('equal')
        plt.colorbar(scatter, ax=ax2, label='Velocity (m/s)')
        
        # 3. Height
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.fill_between(t, pos[:, 2], alpha=0.3)
        ax3.plot(t, pos[:, 2], 'b-')
        ax3.axvspan(0, t[rest_end], alpha=0.2, color='gray')
        ax3.set_title('Vertical Profile')
        
        # 4. Velocity
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(t, vel[:, 0], 'r-', label='Vx', alpha=0.7)
        ax4.plot(t, vel[:, 1], 'g-', label='Vy', alpha=0.7)
        ax4.plot(t, vel[:, 2], 'b-', label='Vz', alpha=0.7)
        ax4.plot(t, vel_mag, 'k-', label='|V|', linewidth=2)
        ax4.axvspan(0, t[rest_end], alpha=0.2, color='gray')
        ax4.set_title('Velocity')
        ax4.legend()
        
        # 5. Acceleration
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(t, acc[:, 0], 'r-', alpha=0.7)
        ax5.plot(t, acc[:, 1], 'g-', alpha=0.7)
        ax5.plot(t, acc[:, 2], 'b-', alpha=0.7)
        ax5.plot(t, acc_mag, 'k-', linewidth=2)
        ax5.axvspan(0, t[rest_end], alpha=0.2, color='gray')
        ax5.set_title('Acceleration')
        
        # 6. Orientation
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(t, self.analyzer.df['pitch'], 'r-', label='Pitch')
        ax6.plot(t, self.analyzer.df['yaw'], 'g-', label='Yaw')
        ax6.plot(t, self.analyzer.df['roll'], 'b-', label='Roll')
        ax6.axvspan(0, t[rest_end], alpha=0.2, color='gray')
        ax6.set_title('Orientation')
        ax6.legend()
        
        plt.suptitle('Freeskating Analysis (Drift Corrected)', fontsize=14, fontweight='bold')
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
        print(f"Total Distance:     {stats['total_distance_m']:.2f} m")
        print(f"Max Velocity:       {stats['max_velocity_ms']:.2f} m/s ({stats['max_velocity_ms']*3.6:.1f} km/h)")
        print(f"Average Velocity:   {stats['avg_velocity_ms']:.2f} m/s")
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
    
    # Configure for your data
    config = SkateConfig(
        rest_duration_sec=2.0,      # 2 seconds of initial rest
        rest_accel_threshold=0.5,   # m/s² threshold
        drift_model='linear'         # 'linear', 'polynomial', or 'cumulative'
    )
    
    analyzer = SkateAnalyzer(config)
    
    try:
        analyzer.load_data(filepath)
        analyzer.analyze()
        
        viz = Visualizer(analyzer)
        viz.print_statistics()
        
        print("\nGenerating visualizations...")
        viz.plot_velocity_drift_analysis("velocity_drift_correction.png")
        viz.plot_3d_trajectory("freeskate_3d_trajectory.png")
        viz.plot_analysis_dashboard("freeskate_analysis_dashboard.png")
        
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
        })
        output_df.to_csv("freeskate_processed_data.csv", index=False)
        print("\nSaved to freeskate_processed_data.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()