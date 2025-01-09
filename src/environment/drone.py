import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy.typing as npt


@dataclass
class DroneState:
    """Represents the state of a drone at any given time"""

    position: npt.NDArray[np.float64]  # [x, y, z]
    velocity: npt.NDArray[np.float64]  # [vx, vy, vz]
    acceleration: npt.NDArray[np.float64]  # [ax, ay, az]
    battery: float  # Percentage remaining (0-100)
    status: str  # 'active', 'failed', 'charging'
    id: int  # Unique identifier


class Drone:
    """Represents a drone in the swarm with its physical properties and behaviors"""

    def __init__(
        self,
        drone_id: int,
        initial_position: npt.NDArray[np.float64],
        max_velocity: float = 10.0,
        max_acceleration: float = 5.0,
        battery_capacity: float = 100.0,
        communication_range: float = 50.0,
    ):
        """
        Initialize a drone with given parameters
        """
        self.state = DroneState(
            position=initial_position.copy(),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            battery=battery_capacity,
            status="active",
            id=drone_id,
        )

        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.battery_capacity = battery_capacity
        self.communication_range = communication_range

        # PSO parameters
        self.personal_best_position = initial_position.copy()
        self.personal_best_value = float("-inf")

    def update_state(
        self,
        dt: float,
        pso_velocity: npt.NDArray[np.float64],
        rl_action: npt.NDArray[np.float64],
    ) -> None:
        """
        Update drone state based on PSO and RL inputs
        """
        if self.state.status != "active":
            return

        # Combine PSO and RL inputs (weighted sum)
        pso_weight = 0.7
        rl_weight = 0.3
        desired_velocity = pso_weight * pso_velocity + rl_weight * rl_action

        # Apply acceleration limits
        acceleration = (desired_velocity - self.state.velocity) / dt
        acceleration_magnitude = np.linalg.norm(acceleration)
        if acceleration_magnitude > self.max_acceleration:
            acceleration = (
                acceleration / acceleration_magnitude
            ) * self.max_acceleration

        # Update velocity and position using physics
        self.state.acceleration = acceleration
        new_velocity = self.state.velocity + acceleration * dt

        # Apply velocity limits
        velocity_magnitude = np.linalg.norm(new_velocity)
        if velocity_magnitude > self.max_velocity:
            new_velocity = (new_velocity / velocity_magnitude) * self.max_velocity

        self.state.velocity = new_velocity
        self.state.position += new_velocity * dt

        # Update battery level (simplified model)
        self.update_battery(dt, velocity_magnitude)

    def update_battery(self, dt: float, velocity_magnitude: float) -> None:
        """
        Update battery level based on movement and time
        """
        # Battery consumption model: base rate + movement cost
        base_consumption = 0.1  # % per second
        movement_consumption = 0.05 * velocity_magnitude  # % per second based on speed

        total_consumption = (base_consumption + movement_consumption) * dt
        self.state.battery = max(0.0, self.state.battery - total_consumption)

        if self.state.battery <= 0:
            self.state.status = "failed"

    def is_in_communication_range(
        self, other_position: npt.NDArray[np.float64]
    ) -> bool:
        """
        Check if another position is within communication range
        """
        distance = np.linalg.norm(self.state.position - other_position)
        return distance <= self.communication_range

    def get_sensor_data(self) -> npt.NDArray[np.float64]:
        """
        Get sensor readings (position, velocity, battery)
        """
        return np.concatenate(
            [self.state.position, self.state.velocity, [self.state.battery]]
        )

    def update_personal_best(self, current_value: float) -> None:
        """
        Update personal best position for PSO
        """
        if current_value > self.personal_best_value:
            self.personal_best_value = current_value
            self.personal_best_position = self.state.position.copy()

    def reset(self, initial_position: npt.NDArray[np.float64]) -> None:
        """
        Reset drone to initial state
        """
        self.state.position = initial_position.copy()
        self.state.velocity = np.zeros(3)
        self.state.acceleration = np.zeros(3)
        self.state.battery = self.battery_capacity
        self.state.status = "active"
        self.personal_best_position = initial_position.copy()
        self.personal_best_value = float("-inf")
