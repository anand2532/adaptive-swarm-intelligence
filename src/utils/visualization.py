import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional
from environment.world import World
from environment.drone import Drone
from environment.obstacles import Building, DynamicObstacle

class Visualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(12, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
    def plot_world(self, world: World, show: bool = True) -> None:
        """Plot the entire world state"""
        self.ax.clear()
        
        # Set world boundaries
        self.ax.set_xlim([0, world.dimensions[0]])
        self.ax.set_ylim([0, world.dimensions[1]])
        self.ax.set_zlim([0, world.dimensions[2]])
        
        # Plot buildings
        self._plot_buildings(world.obstacles)
        
        # Plot drones
        self._plot_drones(world.drones)
        
        # Plot target points
        self._plot_targets(world.target_points)
        
        # Plot coverage
        self._plot_coverage(world.coverage_grid)
        
        if show:
            plt.pause(0.01)

    def _plot_buildings(self, buildings: List[Building]) -> None:
        """Plot building obstacles"""
        for building in buildings:
            x, y, z = building.position
            dx, dy, dz = building.size
            
            # Create building surfaces
            xx, yy = np.meshgrid([x, x+dx], [y, y+dy])
            self.ax.plot_surface(xx, yy, z+dz*np.ones(xx.shape), alpha=0.3)
            self.ax.plot_surface(xx, yy, z*np.ones(xx.shape), alpha=0.3)
            
            yy, zz = np.meshgrid([y, y+dy], [z, z+dz])
            self.ax.plot_surface(x*np.ones(yy.shape), yy, zz, alpha=0.3)
            self.ax.plot_surface((x+dx)*np.ones(yy.shape), yy, zz, alpha=0.3)
            
            xx, zz = np.meshgrid([x, x+dx], [z, z+dz])
            self.ax.plot_surface(xx, y*np.ones(xx.shape), zz, alpha=0.3)
            self.ax.plot_surface(xx, (y+dy)*np.ones(xx.shape), zz, alpha=0.3)

    def _plot_drones(self, drones: List[Drone]) -> None:
        """Plot drone positions and states"""
        for drone in drones:
            color = 'g' if drone.state.status == 'active' else 'r'
            self.ax.scatter(*drone.state.position, c=color, marker='o')
            
            # Plot velocity vector
            self.ax.quiver(*drone.state.position, *drone.state.velocity, 
                          length=1.0, normalize=True, color=color)

    def _plot_targets(self, targets: List[np.ndarray]) -> None:
        """Plot target points"""
        if targets:
            points = np.array(targets)
            self.ax.scatter(points[:,0], points[:,1], points[:,2], 
                          c='y', marker='*')

    def _plot_coverage(self, coverage_grid: np.ndarray) -> None:
        """Plot coverage visualization"""
        x, y, z = np.where(coverage_grid > 0)
        if len(x) > 0:
            self.ax.scatter(x, y, z, c='b', alpha=0.1, marker='.')

    def save_plot(self, filename: str) -> None:
        """Save current plot to file"""
        self.fig.savefig(filename)

    def close(self) -> None:
        """Close visualization window"""
        plt.close(self.fig)