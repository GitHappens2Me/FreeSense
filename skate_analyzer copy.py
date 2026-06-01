
#!/usr/bin/env python3
"""
Freeskating (Urban Inline Skating) Motion Analysis & 3D Trajectory Visualization
=================================================================================

Analyzes IMU data from inline skates with pre-calculated orientation (pitch/yaw/roll) 
to reconstruct 3D skating path. Includes drift correction and visualization.

Usage:
    python freeskate_analyzer.py skate_data.csv

Data Format Expected:
    timestamp, pitch, yaw, roll, accel_x, accel_y, accel_z
    (timestamp in milliseconds from time.ticks_ms(), acceleration in m/s², angles in degrees)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import binary_dilation, binary_erosion
from dataclasses import dataclass
from typing import Optional
import warnings


@dataclass
class SkateConfig:
    """Configuration parameters for freeskating analysis."""
    # Physical constants
    gravity: float = 9.81  # m/s²
    
    # Filtering - adjusted for dynamic urban skating
    acc_cutoff_hz: float = 10.0      # Higher cutoff for quick urban moves
    
    # Drift correction - tuned for intermittent motion
    zero_velocity_threshold: float = 0.5  # m/s
    min_stationary_duration: float = 0.3  # seconds
    
    # Visualization
    colormap: str = 'plasma'
    line_width: float = 2.0


class OrientationUtils:
    """Handle rotation matrices and coordinate transformations."""
    
    @staticmethod
    def euler_to_rotation_matrix(pitch: float, yaw: float, roll: float, degrees: bool = True) -> np.ndarray:
        """
        Convert Euler angles (pitch, yaw, roll) to rotation matrix.
        Order: ZYX (yaw-pitch-roll) - common for IMUs
        """
        if degrees:
            pitch = np.radians(pitch)
            yaw = np.radians(yaw)
            roll = np.radians(roll)
        
        # Roll (X)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        # Pitch (Y)
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        # Yaw (Z)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        
        # Combined: R = Rz * Ry * Rx
        return Rz @ Ry @ Rx
    
    @staticmethod
    def rotate_acceleration(acc_body: np.ndarray, rotation_matrix: np.ndarray) -> np.ndarray:
        """Rotate acceleration from body frame to world frame."""
        return rotation_matrix @ acc_body


class SkateAnalyzer:
    """Main analysis class for freeskate IMU data."""
    
    def __init__(self, config: SkateConfig = None):
        self.config = config or SkateConfig()
        self.df: Optional[pd.DataFrame] = None
        self.timestamps: Optional[np.ndarray] = None
        self.dt: Optional[float] = None
        self.positions: Optional[np.ndarray] = None
        self.velocities: Optional[np.ndarray] = None
        self.world_accelerations: Optional[np.ndarray] = None
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Load CSV data. Timestamps are in milliseconds."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Validate columns
        required = ['timestamp', 'pitch', 'yaw', 'roll', 'accel_x', 'accel_y', 'accel_z']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Found: {list(df.columns)}")
        
        # Convert milliseconds to seconds
        timestamps = df['timestamp'].values.astype(float)
        df['time_sec'] = (timestamps - timestamps[0]) / 1e3
        
        self.df = df
        self.timestamps = df['time_sec'].values
        self.dt = np.median(np.diff(self.timestamps))
        
        print(f"Loaded {len(df)} samples")
        print(f"Duration: {self.timestamps[-1]:.2f} seconds")
        print(f"Sample rate: {1/self.dt:.1f} Hz")
        
        return df
    
    def lowpass_filter(self, data: np.ndarray, cutoff: float) -> np.ndarray:
        """Apply Butterworth lowpass filter."""
        nyquist = 1 / (2 * self.dt)
        normal_cutoff = cutoff / nyquist
        if normal_cutoff >= 1.0 or normal_cutoff <= 0:
            return data
        b, a = butter(4, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data, axis=0)
    
    def process_accelerations(self) -> np.ndarray:
        """Transform accelerations from body to world frame and remove gravity."""
        print("Processing accelerations...")
        
        n = len(self.df)
        acc_world = np.zeros((n, 3))
        
        for i in range(n):
            # Get rotation matrix
            R = OrientationUtils.euler_to_rotation_matrix(
                self.df['pitch'].iloc[i],
                self.df['yaw'].iloc[i], 
                self.df['roll'].iloc[i]
            )
            
            # Body frame acceleration
            acc_body = np.array([
                self.df['accel_x'].iloc[i],
                self.df['accel_y'].iloc[i],
                self.df['accel_z'].iloc[i]
            ])
            
            # Rotate to world frame
            acc_world[i] = OrientationUtils.rotate_acceleration(acc_body, R)
        
        # Remove gravity (Z is roughly up)
        gravity_estimate = np.median(acc_world[:, 2])
        acc_world[:, 2] -= gravity_estimate
        
        # Apply lowpass filter AFTER gravity removal
        for i in range(3):
            acc_world[:, i] = self.lowpass_filter(acc_world[:, i], self.config.acc_cutoff_hz)
        
        self.world_accelerations = acc_world
        return acc_world
    
    def integrate_to_velocity(self) -> np.ndarray:
        """Integrate acceleration to get velocity."""
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
    
    def detect_stationary_periods(self) -> np.ndarray:
        """
        Detect when skater is stationary (for drift correction).
        """
        # Calculate acceleration magnitude
        acc_mag = np.sqrt(np.sum(self.world_accelerations**2, axis=1))
        
        # Calculate velocity magnitude
        vel_mag = np.sqrt(np.sum(self.velocities**2, axis=1))
        
        # Stationary: low velocity AND relatively stable acceleration
        acc_stable = np.abs(acc_mag - self.config.gravity) < 2.0
        stationary = (vel_mag < self.config.zero_velocity_threshold) & acc_stable
        
        # Morphological operations to clean up
        stationary = binary_dilation(stationary, iterations=3)
        stationary = binary_erosion(stationary, iterations=3)
        
        return stationary
    
    def apply_zero_velocity_update(self) -> np.ndarray:
        """Correct velocity drift using zero-velocity updates."""
        print("Applying zero-velocity updates...")
        
        stationary = self.detect_stationary_periods()
        
        print(f"  Detected {np.sum(stationary)} stationary samples ({np.sum(stationary)/len(stationary)*100:.1f}%)")
        
        velocity_corrected = self.velocities.copy()
        velocity_corrected[stationary] = 0
        
        # Smooth with median filter
        for i in range(3):
            velocity_corrected[:, i] = medfilt(velocity_corrected[:, i], kernel_size=5)
        
        self.velocities = velocity_corrected
        return velocity_corrected
    
    def integrate_to_position(self) -> np.ndarray:
        """Integrate velocity to get position."""
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
        """Calculate movement statistics."""
        stats = {
            'duration_sec': self.timestamps[-1],
            'total_distance_m': np.sum(np.sqrt(np.sum(np.diff(self.positions, axis=0)**2, axis=1))),
            'max_velocity_ms': np.max(np.sqrt(np.sum(self.velocities**2, axis=1))),
            'avg_velocity_ms': np.mean(np.sqrt(np.sum(self.velocities**2, axis=1))),
            'max_acceleration_ms2': np.max(np.sqrt(np.sum(self.world_accelerations**2, axis=1))),
            'height_range_m': np.max(self.positions[:, 2]) - np.min(self.positions[:, 2]),
        }
        return stats
    
    def analyze(self) -> 'SkateAnalyzer':
        """Run full analysis pipeline."""
        self.process_accelerations()
        self.integrate_to_velocity()
        self.apply_zero_velocity_update()
        self.integrate_to_position()
        return self


class Visualizer:
    """Create visualizations for the freeskate data."""
    
    def __init__(self, analyzer: SkateAnalyzer):
        self.analyzer = analyzer
        self.config = analyzer.config
        
    def plot_3d_trajectory(self, save_path: Optional[str] = None):
        """Create 3D visualization of the skating path."""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = self.analyzer.positions
        velocities = np.sqrt(np.sum(self.analyzer.velocities**2, axis=1))
        
        # Create color gradient based on velocity
        norm = plt.Normalize(velocities.min(), velocities.max())
        cmap = plt.get_cmap(self.config.colormap)
        
        # Plot trajectory with color gradient
        for i in range(len(positions) - 1):
            color = cmap(norm(velocities[i]))
            ax.plot3D(positions[i:i+2, 0], positions[i:i+2, 1], 
                     positions[i:i+2, 2], color=color, 
                     linewidth=self.config.line_width)
        
        # Mark start and end
        ax.scatter(*positions[0], color='green', s=100, marker='o', 
                  label='Start', edgecolors='black', linewidths=1)
        ax.scatter(*positions[-1], color='red', s=100, marker='s', 
                  label='End', edgecolors='black', linewidths=1)
        
        # Add colorbar
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(velocities)
        cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        cbar.set_label('Velocity (m/s)', fontsize=10)
        
        # Labels and title
        ax.set_xlabel('X (m)', fontsize=11)
        ax.set_ylabel('Y (m)', fontsize=11)
        ax.set_zlabel('Z (m)', fontsize=11)
        ax.set_title('Freeskating 3D Trajectory\n(colored by velocity)', fontsize=13)
        
        # Equal aspect ratio
        max_range = np.array([
            positions[:, 0].max() - positions[:, 0].min(),
            positions[:, 1].max() - positions[:, 1].min(),
            positions[:, 2].max() - positions[:, 2].min()
        ]).max() / 2.0
        
        mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
        mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
        mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
        
        if max_range > 0:
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.legend(loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved 3D plot to {save_path}")
        
        plt.show()
        return fig
    
    def plot_analysis_dashboard(self, save_path: Optional[str] = None):
        """Create comprehensive analysis dashboard."""
        fig = plt.figure(figsize=(16, 12))
        
        # Extract data
        t = self.analyzer.timestamps
        pos = self.analyzer.positions
        vel = self.analyzer.velocities
        vel_mag = np.sqrt(np.sum(vel**2, axis=1))
        acc = self.analyzer.world_accelerations
        acc_mag = np.sqrt(np.sum(acc**2, axis=1))
        
        # 1. 3D Trajectory (top left)
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        ax1.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=1)
        ax1.scatter(*pos[0], color='green', s=50, label='Start')
        ax1.scatter(*pos[-1], color='red', s=50, label='End')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # 2. X-Y Trajectory (top middle)
        ax2 = fig.add_subplot(2, 3, 2)
        scatter = ax2.scatter(pos[:, 0], pos[:, 1], c=vel_mag, cmap='plasma', s=10)
        ax2.plot(pos[0, 0], pos[0, 1], 'go', markersize=10, label='Start')
        ax2.plot(pos[-1, 0], pos[-1, 1], 'rs', markersize=10, label='End')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Top-Down View (colored by velocity)')
        ax2.set_aspect('equal')
        ax2.legend()
        plt.colorbar(scatter, ax=ax2, label='Velocity (m/s)')
        
        # 3. Height profile (top right)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.fill_between(t, pos[:, 2], alpha=0.3)
        ax3.plot(t, pos[:, 2], 'b-', linewidth=1)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Height (m)')
        ax3.set_title('Vertical Profile')
        ax3.grid(True, alpha=0.3)
        
        # 4. Velocity over time (bottom left)
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(t, vel[:, 0], 'r-', label='Vx', alpha=0.7)
        ax4.plot(t, vel[:, 1], 'g-', label='Vy', alpha=0.7)
        ax4.plot(t, vel[:, 2], 'b-', label='Vz', alpha=0.7)
        ax4.plot(t, vel_mag, 'k-', label='|V|', linewidth=2)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Velocity (m/s)')
        ax4.set_title('Velocity Components')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. Acceleration over time (bottom middle)
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(t, acc[:, 0], 'r-', label='Ax', alpha=0.7)
        ax5.plot(t, acc[:, 1], 'g-', label='Ay', alpha=0.7)
        ax5.plot(t, acc[:, 2], 'b-', label='Az', alpha=0.7)
        ax5.plot(t, acc_mag, 'k-', label='|A|', linewidth=2)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Acceleration (m/s²)')
        ax5.set_title('Acceleration Components')
        ax5.legend(loc='upper right')
        ax5.grid(True, alpha=0.3)
        
        # 6. Orientation (bottom right)
        ax6 = fig.add_subplot(2, 3, 6)
        ax6.plot(t, self.analyzer.df['pitch'], 'r-', label='Pitch')
        ax6.plot(t, self.analyzer.df['yaw'], 'g-', label='Yaw')
        ax6.plot(t, self.analyzer.df['roll'], 'b-', label='Roll')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Angle (degrees)')
        ax6.set_title('Orientation')
        ax6.legend(loc='upper right')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('Freeskating Motion Analysis Dashboard', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved dashboard to {save_path}")
        
        plt.show()
        return fig
    
    def print_statistics(self):
        """Print movement statistics."""
        stats = self.analyzer.calculate_statistics()
        
        print("\n" + "="*50)
        print("FREESKATING SESSION STATISTICS")
        print("="*50)
        print(f"Duration:           {stats['duration_sec']:.2f} seconds")
        print(f"Total Distance:     {stats['total_distance_m']:.2f} meters")
        print(f"Max Velocity:       {stats['max_velocity_ms']:.2f} m/s ({stats['max_velocity_ms']*3.6:.1f} km/h)")
        print(f"Average Velocity:   {stats['avg_velocity_ms']:.2f} m/s ({stats['avg_velocity_ms']*3.6:.1f} km/h)")
        print(f"Max Acceleration:   {stats['max_acceleration_ms2']:.2f} m/s²")
        print(f"Height Variation:   {stats['height_range_m']:.3f} meters")
        print("="*50)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python freeskate_analyzer.py <csv_file>")
        print("Using default: skate_data.csv")
        print("\nExpected CSV format:")
        print("timestamp,pitch,yaw,roll,accel_x,accel_y,accel_z")
        print("timestamp in milliseconds (from time.ticks_ms())")
        filepath = "skate_data.csv"
    else:
        filepath = sys.argv[1]
    
    # Create analyzer with default config
    config = SkateConfig()
    analyzer = SkateAnalyzer(config)
    
    try:
        # Load and process data
        analyzer.load_data(filepath)
        
        print(f"\nSample rate: {1/analyzer.dt:.1f} Hz")
        print(f"Time step: {analyzer.dt*1000:.1f} ms")
        
        analyzer.analyze()
        
        # Create visualizations
        viz = Visualizer(analyzer)
        viz.print_statistics()
        
        print("\nGenerating visualizations...")
        viz.plot_3d_trajectory(save_path="freeskate_3d_trajectory.png")
        viz.plot_analysis_dashboard(save_path="freeskate_analysis_dashboard.png")
        
        # Save processed data
        output_df = pd.DataFrame({
            'time_sec': analyzer.timestamps,
            'pos_x': analyzer.positions[:, 0],
            'pos_y': analyzer.positions[:, 1],
            'pos_z': analyzer.positions[:, 2],
            'vel_x': analyzer.velocities[:, 0],
            'vel_y': analyzer.velocities[:, 1],
            'vel_z': analyzer.velocities[:, 2],
            'vel_mag': np.sqrt(np.sum(analyzer.velocities**2, axis=1)),
            'acc_world_x': analyzer.world_accelerations[:, 0],
            'acc_world_y': analyzer.world_accelerations[:, 1],
            'acc_world_z': analyzer.world_accelerations[:, 2],
        })
        output_df.to_csv("freeskate_processed_data.csv", index=False)
        print("\nSaved processed data to freeskate_processed_data.csv")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
