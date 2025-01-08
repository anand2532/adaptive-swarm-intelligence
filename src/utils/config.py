import yaml
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class WorldConfig:
    dimensions: tuple
    num_drones: int
    num_buildings: int
    coverage_resolution: float

@dataclass
class DroneConfig:
    max_velocity: float
    max_acceleration: float
    battery_capacity: float
    communication_range: float

@dataclass
class PSOConfig:
    inertia_weight: float
    cognitive_weight: float
    social_weight: float
    max_velocity: float

@dataclass
class RLConfig:
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    memory_size: int
    batch_size: int
    hidden_dim: int

@dataclass
class TrainingConfig:
    num_episodes: int
    max_steps: int
    target_coverage: float
    update_target_freq: int
    save_freq: int

@dataclass
class SimulationConfig:
    world: WorldConfig
    drone: DroneConfig
    pso: PSOConfig
    rl: RLConfig
    training: TrainingConfig

class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> SimulationConfig:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return SimulationConfig(
            world=WorldConfig(**config_dict['world']),
            drone=DroneConfig(**config_dict['drone']),
            pso=PSOConfig(**config_dict['pso']),
            rl=RLConfig(**config_dict['rl']),
            training=TrainingConfig(**config_dict['training'])
        )

    @staticmethod
    def save(config: SimulationConfig, config_path: str) -> None:
        """Save configuration to YAML file"""
        config_dict = {
            'world': vars(config.world),
            'drone': vars(config.drone),
            'pso': vars(config.pso),
            'rl': vars(config.rl),
            'training': vars(config.training)
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Default configuration
DEFAULT_CONFIG = SimulationConfig(
    world=WorldConfig(
        dimensions=(100, 100, 50),
        num_drones=10,
        num_buildings=20,
        coverage_resolution=1.0
    ),
    drone=DroneConfig(
        max_velocity=10.0,
        max_acceleration=5.0,
        battery_capacity=100.0,
        communication_range=50.0
    ),
    pso=PSOConfig(
        inertia_weight=0.7,
        cognitive_weight=1.5,
        social_weight=1.5,
        max_velocity=10.0
    ),
    rl=RLConfig(
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        memory_size=10000,
        batch_size=64,
        hidden_dim=256
    ),
    training=TrainingConfig(
        num_episodes=1000,
        max_steps=1000,
        target_coverage=0.9,
        update_target_freq=100,
        save_freq=50
    )
)