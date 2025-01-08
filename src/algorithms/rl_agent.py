import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List
import numpy.typing as npt
from collections import deque
import random

# class DroneNetwork(nn.Module):
#     def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(state_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, action_dim)
#         )
        
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.network(x)

class DroneNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:  # If input is a single state
            x = x.unsqueeze(0)
        return self.network(x)

class RLAgent:
    # def __init__(self, 
    #              state_dim: int,
    #              action_dim: int,
    #              learning_rate: float = 1e-4,
    #              gamma: float = 0.99,
    #              epsilon_start: float = 1.0,
    #              epsilon_end: float = 0.01,
    #              epsilon_decay: float = 0.995,
    #              memory_size: int = 10000,
    #              batch_size: int = 64):
        
    #     self.action_dim = action_dim
    #     self.gamma = gamma
    #     self.epsilon = epsilon_start
    #     self.epsilon_end = epsilon_end
    #     self.epsilon_decay = epsilon_decay
    #     self.batch_size = batch_size
        
    #     # Neural Networks
    #     self.policy_net = DroneNetwork(state_dim, action_dim)
    #     self.target_net = DroneNetwork(state_dim, action_dim)
    def __init__(self, 
                state_dim: int,
                action_dim: int,
                hidden_dim: int = 256,
                learning_rate: float = 1e-4,
                gamma: float = 0.99,
                epsilon_start: float = 1.0,
                epsilon_end: float = 0.01,
                epsilon_decay: float = 0.995,
                memory_size: int = 10000,
                batch_size: int = 64):
        
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        
        # Neural Networks
        self.policy_net = DroneNetwork(state_dim, action_dim, hidden_dim)
        self.target_net = DroneNetwork(state_dim, action_dim, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net.to(self.device)
        self.target_net.to(self.device)

    def get_action(self, state: npt.NDArray[np.float64], 
                   training: bool = True) -> npt.NDArray[np.float64]:
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            # Random action
            action = np.random.uniform(-1, 1, size=self.action_dim)
        else:
            # Network action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.policy_net(state_tensor).cpu().numpy()[0]
        
        return action

    # def store_transition(self, state: npt.NDArray[np.float64], 
    #                     action: npt.NDArray[np.float64],
    #                     reward: float, 
    #                     next_state: npt.NDArray[np.float64], 
    #                     done: bool) -> None:
    #     """Store transition in replay memory"""
    #     self.memory.append((state, action, reward, next_state, done))

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory"""
        self.memory.append((
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            reward,
            np.array(next_state, dtype=np.float32),
            done
        ))

    # def train(self) -> float:
    #     """Train the agent using a batch of experiences"""
    #     if len(self.memory) < self.batch_size:
    #         return 0.0

    #     # Sample batch
    #     batch = random.sample(self.memory, self.batch_size)
    #     states, actions, rewards, next_states, dones = zip(*batch)
        
    #     # Convert to tensors
    #     state_batch = torch.FloatTensor(states).to(self.device)
    #     action_batch = torch.FloatTensor(actions).to(self.device)
    #     reward_batch = torch.FloatTensor(rewards).to(self.device)
    #     next_state_batch = torch.FloatTensor(next_states).to(self.device)
    #     done_batch = torch.FloatTensor(dones).to(self.device)
        
    #     # Compute Q values
    #     current_q_values = self.policy_net(state_batch)
    #     next_q_values = self.target_net(next_state_batch).detach()
        
    #     # Compute expected Q values
    #     expected_q_values = reward_batch + (1 - done_batch) * self.gamma * \
    #                       next_q_values.max(1)[0]
        
    #     # Compute loss
    #     loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
    #     # Optimize
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
        
    #     # Update epsilon
    #     self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
    #     return loss.item()

    def train(self) -> float:
        if len(self.memory) < self.batch_size:
            return 0.0

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors with proper shapes
        state_batch = torch.FloatTensor(np.array(states)).to(self.device)
        action_batch = torch.FloatTensor(np.array(actions)).to(self.device)
        reward_batch = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_states)).to(self.device)
        done_batch = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)
        
        current_q_values = self.policy_net(state_batch)
        next_q_values = self.target_net(next_state_batch).detach()
        
        # Get Q values for the taken actions
        q_values = torch.sum(current_q_values * action_batch, dim=1, keepdim=True)
        next_max_q = torch.max(next_q_values, dim=1, keepdim=True)[0]
        expected_q_values = reward_batch + (1 - done_batch) * self.gamma * next_max_q
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return loss.item()

    def update_target_network(self) -> None:
        """Update target network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path: str) -> None:
        """Save model weights"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_model(self, path: str) -> None:
        """Load model weights"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])