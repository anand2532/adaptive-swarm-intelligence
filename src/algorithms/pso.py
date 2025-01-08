import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from environment.drone import Drone

class PSOOptimizer:
    def __init__(self, 
                 inertia_weight: float = 0.7,
                 cognitive_weight: float = 1.5,
                 social_weight: float = 1.5,
                 max_velocity: float = 10.0):
        self.w = inertia_weight  # Inertia weight
        self.c1 = cognitive_weight  # Cognitive weight
        self.c2 = social_weight  # Social weight
        self.max_velocity = max_velocity
        self.global_best_position = None
        self.global_best_value = float('-inf')

    def initialize_swarm(self, drones: List[Drone]) -> None:
        """Initialize swarm parameters"""
        positions = np.array([drone.state.position for drone in drones])
        self.global_best_position = positions[0].copy()

    def update_global_best(self, drones: List[Drone], 
                          fitness_values: npt.NDArray[np.float64]) -> None:
        """Update the global best position"""
        best_idx = np.argmax(fitness_values)
        if fitness_values[best_idx] > self.global_best_value:
            self.global_best_value = fitness_values[best_idx]
            self.global_best_position = drones[best_idx].state.position.copy()

    def compute_velocity(self, drone: Drone) -> npt.NDArray[np.float64]:
        """Compute PSO velocity for a drone"""
        r1, r2 = np.random.random(2)
        
        # PSO velocity update equation
        inertia = self.w * drone.state.velocity
        cognitive = self.c1 * r1 * (drone.personal_best_position - drone.state.position)
        social = self.c2 * r2 * (self.global_best_position - drone.state.position)
        
        new_velocity = inertia + cognitive + social
        
        # Apply velocity clamping
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > self.max_velocity:
            new_velocity = (new_velocity / velocity_magnitude) * self.max_velocity
            
        return new_velocity

    def calculate_fitness(self, drone: Drone, 
                        target_points: List[npt.NDArray[np.float64]], 
                        obstacles: List['Obstacle']) -> float:
        """Calculate fitness value for a drone's position"""
        # Distance to nearest target point
        min_target_distance = min(np.linalg.norm(drone.state.position - target)
                                for target in target_points)
        
        # Distance to nearest obstacle
        min_obstacle_distance = min(np.linalg.norm(drone.state.position - 
                                                  obstacle.position)
                                  for obstacle in obstacles)
        
        # Combine objectives (minimize target distance, maximize obstacle distance)
        fitness = -min_target_distance + 0.5 * min_obstacle_distance
        return float(fitness)

    def step(self, drones: List[Drone], target_points: List[npt.NDArray[np.float64]], 
            obstacles: List['Obstacle']) -> List[npt.NDArray[np.float64]]:
        """Perform one PSO step and return velocities for all drones"""
        # Calculate fitness for all drones
        fitness_values = np.array([
            self.calculate_fitness(drone, target_points, obstacles)
            for drone in drones
        ])
        
        # Update global and personal bests
        self.update_global_best(drones, fitness_values)
        for drone, fitness in zip(drones, fitness_values):
            drone.update_personal_best(fitness)
        
        # Compute new velocities
        return [self.compute_velocity(drone) for drone in drones]