import numpy as np
from typing import List, Tuple, Optional
import numpy.typing as npt
from .drone import Drone
from .obstacles import Obstacle, Building


class World:
    """Represents the 3D environment where the drone swarm operates"""

    def __init__(
        self,
        dimensions: Tuple[float, float, float],
        num_drones: int = 10,
        num_buildings: int = 20,
    ):
        """
        Initialize the world with given dimensions
        """
        self.dimensions = dimensions
        self.drones: List[Drone] = []
        self.obstacles: List[Obstacle] = []
        self.target_points: List[npt.NDArray[np.float64]] = []

        # Initialize coverage grid (discretized space)
        self.grid_resolution = 1.0  # 1 meter per grid cell
        self.coverage_grid = np.zeros(
            (
                int(dimensions[0] / self.grid_resolution),
                int(dimensions[1] / self.grid_resolution),
                int(dimensions[2] / self.grid_resolution),
            )
        )

        self.setup_environment(num_drones, num_buildings)

    def setup_environment(self, num_drones: int, num_buildings: int) -> None:
        """
        Set up the initial environment with drones and obstacles
        """
        # Create drones with random initial positions
        for i in range(num_drones):
            position = self.get_random_valid_position()
            drone = Drone(drone_id=i, initial_position=position)
            self.drones.append(drone)

        # Create buildings
        for _ in range(num_buildings):
            self.generate_random_building()

        # Generate target points for coverage
        self.generate_target_points()

    def generate_random_building(self) -> None:
        """
        Generate a random building obstacle
        """
        position = np.random.uniform(
            [0, 0, 0], [self.dimensions[0] * 0.8, self.dimensions[1] * 0.8, 0]
        )

        size = np.random.uniform([10, 10, 20], [30, 30, 50])
        building = Building(position=position, size=size)

        # Check for overlap with other buildings
        if not any(building.intersects(obs) for obs in self.obstacles):
            self.obstacles.append(building)

    def generate_target_points(self, num_points: int = 50) -> None:
        """
        Generate points that need to be covered by the swarm
        """
        self.target_points = []
        for _ in range(num_points):
            point = self.get_random_valid_position()
            self.target_points.append(point)

    def get_random_valid_position(self) -> npt.NDArray[np.float64]:
        """
        Get a random position that doesn't collide with obstacles
        """
        while True:
            position = np.random.uniform([0, 0, 0], self.dimensions)
            if not self.check_collision(position):
                return position

    def check_collision(self, position: npt.NDArray[np.float64]) -> bool:
        """
        Check if a position collides with any obstacle
        """
        return any(obs.contains_point(position) for obs in self.obstacles)

    def update_coverage(
        self,
        drone_positions: List[npt.NDArray[np.float64]],
        coverage_radius: float = 5.0,
    ) -> None:
        """
        Update coverage grid based on drone positions
        """
        for position in drone_positions:
            # Convert to grid coordinates
            grid_pos = (position / self.grid_resolution).astype(int)
            radius_cells = int(coverage_radius / self.grid_resolution)

            # Update coverage in a sphere around the drone
            x, y, z = grid_pos
            for dx in range(-radius_cells, radius_cells + 1):
                for dy in range(-radius_cells, radius_cells + 1):
                    for dz in range(-radius_cells, radius_cells + 1):
                        if dx * dx + dy * dy + dz * dz <= radius_cells * radius_cells:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if (
                                0 <= nx < self.coverage_grid.shape[0]
                                and 0 <= ny < self.coverage_grid.shape[1]
                                and 0 <= nz < self.coverage_grid.shape[2]
                            ):
                                self.coverage_grid[nx, ny, nz] = 1

    def get_coverage_percentage(self) -> float:
        """
        Calculate the percentage of space covered
        """
        return np.mean(self.coverage_grid)

    def get_drone_neighbors(self, drone_id: int, max_distance: float) -> List[int]:
        """
        Get IDs of neighboring drones within max_distance
        """
        drone = self.drones[drone_id]
        neighbors = []

        for other_drone in self.drones:
            if other_drone.state.id != drone_id:
                distance = np.linalg.norm(
                    drone.state.position - other_drone.state.position
                )
                if distance <= max_distance:
                    neighbors.append(other_drone.state.id)

        return neighbors

    def reset(self) -> None:
        """
        Reset the world state
        """
        self.coverage_grid.fill(0)
        for drone in self.drones:
            drone.reset(self.get_random_valid_position())
        self.generate_target_points()

    def step(self, dt: float) -> None:
        """
        Update world state for one timestep
        """
        # Update coverage based on current drone positions
        drone_positions = [
            drone.state.position
            for drone in self.drones
            if drone.state.status == "active"
        ]
        self.update_coverage(drone_positions)

        # Check for collisions between drones
        self.check_drone_collisions()

    def check_drone_collisions(self, safe_distance: float = 2.0) -> None:
        """
        Check and handle collisions between drones
        """
        for i, drone1 in enumerate(self.drones):
            if drone1.state.status != "active":
                continue

            for j, drone2 in enumerate(self.drones[i + 1 :], i + 1):
                if drone2.state.status != "active":
                    continue

                distance = np.linalg.norm(drone1.state.position - drone2.state.position)
                if distance < safe_distance:
                    # Implement collision avoidance or failure handling
                    drone1.state.status = "failed"
                    drone2.state.status = "failed"
