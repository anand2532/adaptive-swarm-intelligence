import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import time
from environment.world import World
from environment.drone import Drone

@dataclass
class Episode:
    """Data for a single training episode"""
    episode_id: int
    steps: int
    total_reward: float
    coverage_achieved: float
    collisions: int
    mission_success: bool
    completion_time: float

class MetricsTracker:
    def __init__(self):
        self.reset()
        self.episode_start_time = None
        
    def reset(self) -> None:
        """Reset all metrics"""
        self.episodes: List[Episode] = []
        self.current_episode = {
            'steps': 0,
            'reward': 0.0,
            'collisions': 0,
            'coverage': []
        }
        
    def start_episode(self, episode_id: int) -> None:
        """Start tracking a new episode"""
        self.current_episode = {
            'episode_id': episode_id,
            'steps': 0,
            'reward': 0.0,
            'collisions': 0,
            'coverage': []
        }
        self.episode_start_time = time.time()

    def step_update(self, world: World, reward: float) -> None:
        """Update metrics for current step"""
        self.current_episode['steps'] += 1
        self.current_episode['reward'] += reward
        self.current_episode['coverage'].append(world.get_coverage_percentage())

    def add_collision(self) -> None:
        """Record a collision event"""
        self.current_episode['collisions'] += 1

    def end_episode(self, success: bool) -> None:
        """Complete current episode tracking"""
        if self.episode_start_time is None:
            return
            
        completion_time = time.time() - self.episode_start_time
        episode = Episode(
            episode_id=self.current_episode['episode_id'],
            steps=self.current_episode['steps'],
            total_reward=self.current_episode['reward'],
            coverage_achieved=max(self.current_episode['coverage']),
            collisions=self.current_episode['collisions'],
            mission_success=success,
            completion_time=completion_time
        )
        self.episodes.append(episode)

    def get_statistics(self) -> Dict:
        """Calculate summary statistics"""
        if not self.episodes:
            return {}

        return {
            'num_episodes': len(self.episodes),
            'success_rate': sum(ep.mission_success for ep in self.episodes) / len(self.episodes),
            'avg_reward': np.mean([ep.total_reward for ep in self.episodes]),
            'avg_coverage': np.mean([ep.coverage_achieved for ep in self.episodes]),
            'avg_collisions': np.mean([ep.collisions for ep in self.episodes]),
            'avg_completion_time': np.mean([ep.completion_time for ep in self.episodes])
        }

    def save_metrics(self, filename: str) -> None:
        """Save metrics to file"""
        metrics_data = {
            'episodes': [vars(ep) for ep in self.episodes],
            'statistics': self.get_statistics()
        }
        with open(filename, 'w') as f:
            json.dump(metrics_data, f, indent=4)

    def load_metrics(self, filename: str) -> None:
        """Load metrics from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.episodes = [Episode(**ep_data) for ep_data in data['episodes']]