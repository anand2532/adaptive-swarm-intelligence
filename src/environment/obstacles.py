import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List
import numpy.typing as npt

class Obstacle(ABC):
    """Abstract base class for obstacles in the environment"""
    
    @abstractmethod
    def contains_point(self, point: npt.NDArray[np.float64]) -> bool:
        """Check if a point is inside or on the obstacle"""
        pass
    
    @abstractmethod
    def intersects(self, other: 'Obstacle') -> bool:
        """Check if this obstacle intersects with another"""
        pass
    
    @abstractmethod
    def get_boundary_points(self) -> Tuple[npt.NDArray[np.float64], 
                                         npt.NDArray[np.float64]]:
        """Return the min and max points of the obstacle's bounding box"""
        pass

class Building(Obstacle):
    """Represents a building as a rectangular prism obstacle"""
    
    def __init__(self, position: npt.NDArray[np.float64], 
                 size: npt.NDArray[np.float64]):
        """
        Initialize a building with position and size
        position: [x, y, z] of the building's origin (bottom-left-front corner)
        size: [width, length, height] of the building
        """
        self.position = position
        self.size = size
        
        # Compute bounding box corners
        self.min_point = position
        self.max_point = position + size

    def contains_point(self, point: npt.NDArray[np.float64]) -> bool:
        """Check if a point is inside or on the building"""
        return np.all((point >= self.min_point) & (point <= self.max_point))

    def intersects(self, other: Obstacle) -> bool:
        """Check if this building intersects with another obstacle"""
        other_min, other_max = other.get_boundary_points()
        
        return not (np.any(self.max_point < other_min) or 
                   np.any(self.min_point > other_max))

    def get_boundary_points(self) -> Tuple[npt.NDArray[np.float64], 
                                         npt.NDArray[np.float64]]:
        """Get the min and max points of the building's bounding box"""
        return self.min_point, self.max_point

    def get_closest_point(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Get the closest point on the building surface to a given point"""
        closest = np.clip(point, self.min_point, self.max_point)
        return closest

class DynamicObstacle(Obstacle):
    """Represents a moving obstacle like a vehicle or pedestrian"""
    
    def __init__(self, 
                 position: npt.NDArray[np.float64],
                 velocity: npt.NDArray[np.float64],
                 radius: float):
        self.position = position
        self.velocity = velocity
        self.radius = radius

    def update(self, dt: float, bounds: npt.NDArray[np.float64]) -> None:
        """Update obstacle position and handle boundary collisions"""
        new_position = self.position + self.velocity * dt
        
        # Bounce off boundaries
        for i in range(3):
            if new_position[i] < 0 or new_position[i] > bounds[i]:
                self.velocity[i] *= -1
                new_position[i] = np.clip(new_position[i], 0, bounds[i])
                
        self.position = new_position

    def contains_point(self, point: npt.NDArray[np.float64]) -> bool:
        """Check if point is within the obstacle's radius"""
        return np.linalg.norm(point - self.position) <= self.radius

    def intersects(self, other: Obstacle) -> bool:
        """Check intersection with another obstacle"""
        if isinstance(other, DynamicObstacle):
            return np.linalg.norm(self.position - other.position) <= (self.radius + other.radius)
        else:
            other_min, other_max = other.get_boundary_points()
            closest_point = np.clip(self.position, other_min, other_max)
            return np.linalg.norm(self.position - closest_point) <= self.radius

    def get_boundary_points(self) -> Tuple[npt.NDArray[np.float64], 
                                         npt.NDArray[np.float64]]:
        """Get bounding box corners"""
        min_point = self.position - self.radius
        max_point = self.position + self.radius
        return min_point, max_point

    def predict_position(self, t: float) -> npt.NDArray[np.float64]:
        """Predict position after time t"""
        return self.position + self.velocity * t

class ObstacleField:
    """Manages a collection of static and dynamic obstacles"""
    
    def __init__(self):
        self.static_obstacles: List[Building] = []
        self.dynamic_obstacles: List[DynamicObstacle] = []

    def add_building(self, position: npt.NDArray[np.float64], 
                    size: npt.NDArray[np.float64]) -> None:
        """Add a static building obstacle"""
        building = Building(position, size)
        if not any(building.intersects(obs) for obs in self.static_obstacles):
            self.static_obstacles.append(building)

    def add_dynamic_obstacle(self, position: npt.NDArray[np.float64],
                           velocity: npt.NDArray[np.float64],
                           radius: float) -> None:
        """Add a dynamic obstacle"""
        obstacle = DynamicObstacle(position, velocity, radius)
        self.dynamic_obstacles.append(obstacle)

    def update(self, dt: float, bounds: npt.NDArray[np.float64]) -> None:
        """Update all dynamic obstacles"""
        for obstacle in self.dynamic_obstacles:
            obstacle.update(dt, bounds)

    def get_nearest_obstacle(self, point: npt.NDArray[np.float64]) -> Optional[Tuple[Obstacle, float]]:
        """Find the nearest obstacle and its distance to a point"""
        nearest_obstacle = None
        min_distance = float('inf')
        
        # Check static obstacles
        for obstacle in self.static_obstacles:
            closest_point = obstacle.get_closest_point(point)
            distance = np.linalg.norm(point - closest_point)
            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle

        # Check dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            distance = np.linalg.norm(point - obstacle.position) - obstacle.radius
            if distance < min_distance:
                min_distance = distance
                nearest_obstacle = obstacle

        return (nearest_obstacle, min_distance) if nearest_obstacle else None

    def check_collision(self, point: npt.NDArray[np.float64]) -> bool:
        """Check if a point collides with any obstacle"""
        return any(obs.contains_point(point) for obs in self.static_obstacles + self.dynamic_obstacles)