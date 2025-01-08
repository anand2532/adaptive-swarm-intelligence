import argparse
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import torch

from environment.world import World
from environment.drone import Drone
from algorithms.hybrid import HybridController
from utils.visualization import Visualizer
from utils.metrics import MetricsTracker
from utils.config import ConfigLoader

def parse_args():
    parser = argparse.ArgumentParser(description='Drone Swarm Simulation')
    parser.add_argument('--train', action='store_true', help='Run in training mode')
    parser.add_argument('--visualize', action='store_true', help='Run with visualization')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, help='Path to load checkpoint')
    return parser.parse_args()

def setup_directories():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'outputs/run_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoints_dir = output_dir / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    return output_dir, checkpoints_dir

def train(config, output_dir, checkpoints_dir, visualize=False):
    # Initialize components
    world = World(
        dimensions=config.world.dimensions,
        num_drones=config.world.num_drones,
        num_buildings=config.world.num_buildings
    )
    
    state_dim = 13  # position(3) + velocity(3) + obstacle_info(4) + battery(1) + status(2)
    action_dim = 3  # velocity vector (3D)
    
    controller = HybridController(
        num_drones=config.world.num_drones,
        state_dim=state_dim,
        action_dim=action_dim,
        pso_params=vars(config.pso),
        rl_params=vars(config.rl)
    )
    
    metrics = MetricsTracker()
    visualizer = Visualizer() if visualize else None
    
    # Training loop
    for episode in range(config.training.num_episodes):
        world.reset()
        metrics.start_episode(episode)
        episode_reward = 0
        
        for step in range(config.training.max_steps):
            # Get actions from hybrid controller
            actions = controller.step(world, training=True)
            
            # Update drone states
            for drone, action in zip(world.drones, actions):
                drone.update_state(dt=0.1, pso_velocity=action, rl_action=action)
            
            # Update world state
            world.step(dt=0.1)
            
            # Calculate rewards and update metrics
            reward = sum(controller.compute_reward(drone, world) 
                        for drone in world.drones)
            episode_reward += reward
            metrics.step_update(world, reward)
            
            # Visualize if enabled
            if visualizer and step % 10 == 0:
                visualizer.plot_world(world)
            
            # Check completion
            if world.get_coverage_percentage() >= config.training.target_coverage:
                break
        
        # End episode
        metrics.end_episode(
            success=world.get_coverage_percentage() >= config.training.target_coverage
        )
        
        # Save checkpoints
        if episode % config.training.save_freq == 0:
            checkpoint_path = checkpoints_dir / f'model_episode_{episode}.pt'
            controller.save_agents(checkpoint_path)
        
        # Save metrics
        metrics.save_metrics(output_dir / 'metrics.json')
        
        print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
              f"Coverage = {world.get_coverage_percentage():.2%}")

def visualize_simulation(config, checkpoint_path=None):
    world = World(
        dimensions=config.world.dimensions,
        num_drones=config.world.num_drones,
        num_buildings=config.world.num_buildings
    )
    
    state_dim = 13
    action_dim = 3
    
    controller = HybridController(
        num_drones=config.world.num_drones,
        state_dim=state_dim,
        action_dim=action_dim,
        pso_params=vars(config.pso),
        rl_params=vars(config.rl)
    )
    
    if checkpoint_path:
        controller.load_agents(checkpoint_path)
    
    visualizer = Visualizer()
    
    while True:
        world.reset()
        
        for step in range(config.training.max_steps):
            actions = controller.step(world, training=False)
            
            for drone, action in zip(world.drones, actions):
                drone.update_state(dt=0.1, pso_velocity=action, rl_action=action)
            
            world.step(dt=0.1)
            visualizer.plot_world(world)
            
            if world.get_coverage_percentage() >= config.training.target_coverage:
                break
        
        input("Press Enter to run another simulation (Ctrl+C to exit)...")

def main():
    args = parse_args()
    config = ConfigLoader.load(args.config)
    
    if args.train:
        output_dir, checkpoints_dir = setup_directories()
        train(config, output_dir, checkpoints_dir, visualize=args.visualize)
    elif args.visualize:
        visualize_simulation(config, args.checkpoint)
    else:
        print("Please specify --train or --visualize mode")

if __name__ == "__main__":
    main()