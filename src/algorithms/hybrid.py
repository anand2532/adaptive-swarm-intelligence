import numpy as np
from typing import List, Dict, Any
import numpy.typing as npt
from environment.drone import Drone
from environment.world import World
from .pso import PSOOptimizer
from .rl_agent import RLAgent

class HybridController:
    def __init__(self, 
                 num_drones: int,
                 state_dim: int,
                 action_dim: int,
                 pso_weight: float = 0.7,
                 rl_weight: float = 0.3,
                 **kwargs):
        self.num_drones = num_drones
        self.pso_weight = pso_weight
        self.rl_weight = rl_weight
        
        # Initialize PSO and RL components
        self.pso = PSOOptimizer(**kwargs.get('pso_params', {}))
        self.rl_agents = [
            RLAgent(state_dim, action_dim, **kwargs.get('rl_params', {}))
            for _ in range(num_drones)
        ]

    def get_state(self, drone: Drone, world: World) -> npt.NDArray[np.float64]:
        """Get state representation for RL agent"""
        # Drone's own state
        drone_state = drone.get_sensor_data()
        
        # Nearest obstacle information
        nearest_obstacle_dist = float('inf')
        nearest_obstacle_dir = np.zeros(3)
        
        for obstacle in world.obstacles:
            obstacle_pos = obstacle.position
            distance = np.linalg.norm(drone.state.position - obstacle_pos)
            if distance < nearest_obstacle_dist:
                nearest_obstacle_dist = distance
                nearest_obstacle_dir = (obstacle_pos - drone.state.position) / distance
        
        # Nearest target information
        nearest_target_dist = float('inf')
        nearest_target_dir = np.zeros(3)
        
        for target in world.target_points:
            distance = np.linalg.norm(drone.state.position - target)
            if distance < nearest_target_dist:
                nearest_target_dist = distance
                nearest_target_dir = (target - drone.state.position) / distance
        
        # Combine all state information
        state = np.concatenate([
            drone_state,
            nearest_obstacle_dir,
            [nearest_obstacle_dist],
            nearest_target_dir,
            [nearest_target_dist]
        ])
        
        return state

    def compute_reward(self, drone: Drone, world: World) -> float:
        """Compute reward for RL agent"""
        # Coverage reward
        coverage_reward = world.get_coverage_percentage()
        
        # Distance to nearest target
        target_distances = [np.linalg.norm(drone.state.position - target)
                          for target in world.target_points]
        target_reward = -min(target_distances) if target_distances else 0
        
        # Obstacle avoidance reward
        obstacle_distances = [np.linalg.norm(drone.state.position - obs.position)
                            for obs in world.obstacles]
        obstacle_reward = min(obstacle_distances) if obstacle_distances else 0
        
        # Battery efficiency reward
        battery_reward = drone.state.battery / 100.0
        
        # Combine rewards
        total_reward = (0.4 * coverage_reward +
                       0.3 * target_reward +
                       0.2 * obstacle_reward +
                       0.1 * battery_reward)
        
        return float(total_reward)

    def step(self, world: World, training: bool = True) -> List[npt.NDArray[np.float64]]:
        """Compute hybrid control actions for all drones"""
        # Get PSO velocities
        pso_velocities = self.pso.step(world.drones, world.target_points, 
                                     world.obstacles)
        
        # Get RL actions and update agents
        actions = []
        for i, (drone, agent) in enumerate(zip(world.drones, self.rl_agents)):
            if drone.state.status != 'active':
                actions.append(np.zeros(3))
                continue
                
            # Get state and RL action
            state = self.get_state(drone, world)
            rl_action = agent.get_action(state, training)
            
            if training:
                # Store experience and train
                reward = self.compute_reward(drone, world)
                next_state = self.get_state(drone, world)
                done = drone.state.status != 'active'
                
                agent.store_transition(state, rl_action, reward, next_state, done)
                agent.train()
            
            # Combine PSO and RL actions
            combined_action = (self.pso_weight * pso_velocities[i] + 
                             self.rl_weight * rl_action)
            actions.append(combined_action)
        
        return actions

    def update_target_networks(self) -> None:
        """Update target networks of all RL agents"""
        for agent in self.rl_agents:
            agent.update_target_network()

    def save_agents(self, path: str) -> None:
        """Save all RL agents"""
        for i, agent in enumerate(self.rl_agents):
            agent.save_model(f"{path}/agent_{i}.pt")

    def load_agents(self, path: str) -> None:
        """Load all RL agents"""
        for i, agent in enumerate(self.rl_agents):
            agent.load_model(f"{path}/agent_{i}.pt")

    def get_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'pso_best_value': self.pso.global_best_value,
            'rl_epsilon': self.rl_agents[0].epsilon if self.rl_agents else None
        }