import os
import random
import numpy as np
import cv2
import gymnasium as gym
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from undertale_gym import UndertaleEnv, MASK_LEFT, MASK_RIGHT, MASK_TOP, MASK_BOTTOM


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        # input_shape: (channels, height, width)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute output size for the dense layer
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[2], 8, 4), 4, 2), 3, 1)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 8, 4), 4, 2), 3, 1)
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)


def preprocess_frame(frame):
    """
    Crop the arena area, convert to grayscale, and resize.
    Output: 84x84 grayscale image.
    """
    # Crop to arena
    cropped = frame[MASK_TOP:MASK_BOTTOM, MASK_LEFT:MASK_RIGHT]
    # Convert to grayscale
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    # Resize to 84x84
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


class DQNAgent:
    def __init__(self, action_size, state_shape, device):
        self.action_size = action_size
        self.state_shape = state_shape
        self.device = device
        
        self.memory = ReplayBuffer(10000)
        self.batch_size = 32
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.9995
        self.tau = 0.005
        
        self.policy_net = DQN(state_shape, action_size).to(device)
        self.target_net = DQN(state_shape, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.policy_net(state_tensor)
        return torch.argmax(action_values[0]).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Q(s, a)
        q_values = self.policy_net(states).gather(1, actions)

        # max Q(s', a')
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Target net soft update
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        torch.save(self.policy_net.state_dict(), filepath)

    def load(self, filepath):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


def main():
    import glob
    import argparse
    
    parser = argparse.ArgumentParser(description="Train DQN Agent for Undertale")
    parser.add_argument("--render", action="store_true", help="Render the environment in a GUI window")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes to train")
    parser.add_argument("--difficulty", type=int, default=10, help="Bullet difficulty (0-100)")
    parser.add_argument("--pattern", type=str, default="random", help="Bullet pattern (random, rain_down, aimed, mixed, rain_sides)")
    parser.add_argument("--goal", type=int, default=500, help="Target goal for survival (max steps per episode)")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignoring existing checkpoints")
    parser.add_argument("--model", type=str, default="", help="Specific checkpoint to load (default: auto-load latest)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print(f"  PyTorch Device Setup")
    print(f"  Using device: {device}")
    if device.type == 'cpu':
        print("  WARNING: CUDA is not available. PyTorch is running on CPU.")
        print("  To use GPU, you need to install the CUDA version of PyTorch.")
    print("="*50 + "\n")

    # Set up environment
    render_mode = "human" if args.render else None
    env = UndertaleEnv(render_mode=render_mode, pattern=args.pattern, difficulty=args.difficulty, max_steps=args.goal)
    num_actions = env.action_space.n
    
    # We will stack 4 frames. Input shape: (4, 84, 84)
    state_shape = (4, 84, 84)
    agent = DQNAgent(num_actions, state_shape, device)

    episodes = args.episodes
    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    # Checkpoint logic
    if not args.no_resume:
        if args.model and os.path.exists(args.model):
            checkpoints = [args.model]
            latest_checkpoint = args.model
        else:
            checkpoints = glob.glob(os.path.join(save_dir, "*.pth"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            
        if checkpoints:
            print(f"Auto-loading checkpoint: {latest_checkpoint}...")
            try:
                agent.load(latest_checkpoint)
                print("Successfully loaded checkpoint. Resuming training.")
            except Exception as e:
                print(f"Failed to load checkpoint: {e}")
        else:
            print("No previous checkpoints found. Starting fresh.")
    else:
        print("Ignoring saved models. Starting fresh run.")

    # Frame stacking buffer
    # Each state is a stack of 4 consecutive frames to give the agent a sense of motion
    stack_size = 4
    
    global_step = 0
    
    for e in range(episodes):
        obs, info = env.reset()
        obs = preprocess_frame(obs)
        
        # Initialize the stack with the first frame repeated
        state_stack = deque([obs]*stack_size, maxlen=stack_size)
        state = np.array(state_stack) # Shape: (4, 84, 84)

        episode_reward = 0

        while True:
            # Action selection
            action = agent.act(state)

            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # Preprocess and stack
            next_obs = preprocess_frame(next_obs)
            next_state_stack = state_stack.copy()
            next_state_stack.append(next_obs)
            next_state = np.array(next_state_stack)
            
            # Change reward slightly to help DQN (optional)
            # The environment gives 1 + center_bonus per step, -100 on death
            
            # Save to memory
            done = terminated or truncated
            
            # Goal bonus: if the agent survived until the goal (truncated), 
            # give a large reward to differentiate it from just dying at the end.
            if truncated and not terminated:
                reward += 150.0  # Big bonus for reaching the target!
                
            agent.memory.push(state, action, reward, next_state, done)

            # Update state
            state = next_state
            state_stack = next_state_stack
            episode_reward += reward
            global_step += 1

            # Train
            agent.replay()

            if global_step % 5000 == 0:
                print(f"Global Step {global_step} | Sub-saving model...")
                agent.save(f"{save_dir}/dqn_checkpoint_step_{global_step}.pth")

            if done:
                status = "SURVIVED (GOAL)" if truncated and not terminated else "DIED (HIT)"
                print(f"Episode: {e+1:4d}/{episodes} | {status} | Score: {episode_reward:7.1f} | Steps: {info['step']:4d} | Epsilon: {agent.epsilon:.3f}")
                break
                
        if (e + 1) % 50 == 0:
            agent.save(f"{save_dir}/dqn_episode_{e+1}.pth")

    print("Training finished.")
    agent.save(f"{save_dir}/dqn_final.pth")
    env.close()

if __name__ == "__main__":
    main()
