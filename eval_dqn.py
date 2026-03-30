import os
import glob
import time
import numpy as np
import cv2
import torch
import gymnasium as gym
from collections import deque

from undertale_gym import UndertaleEnv
from train_dqn import DQNAgent, preprocess_frame

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate DQN Agent for Undertale")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    parser.add_argument("--delay", type=float, default=0.033, help="Delay between frames (seconds)")
    parser.add_argument("--model", type=str, default="models", help="Directory or path to the model")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*50)
    print(f"  DQN Evaluator")
    print(f"  Using device: {device}")
    print("="*50 + "\n")

    # Set up environment in human rendering mode
    env = UndertaleEnv(render_mode="human", pattern="random", difficulty=10)
    num_actions = env.action_space.n
    
    state_shape = (4, 84, 84)
    agent = DQNAgent(num_actions, state_shape, device)
    
    # Force epsilon to 0 (no random exploration)
    agent.epsilon = 0.0
    agent.epsilon_min = 0.0

    # Auto-load latest checkpoint
    save_dir = args.model
    if os.path.isdir(save_dir):
        checkpoints = glob.glob(os.path.join(save_dir, "*.pth"))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=os.path.getmtime)
            model_path = latest_checkpoint
        else:
            print(f"No previous checkpoints found in {save_dir}. Cannot evaluate.")
            return
    else:
        model_path = save_dir
        
    print(f"Loading checkpoint: {model_path}...")
    try:
        agent.load(model_path)
        print("Successfully loaded. Starting evaluation.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    stack_size = 4
    
    for e in range(args.episodes):
        obs, info = env.reset()
        obs = preprocess_frame(obs)
        
        state_stack = deque([obs]*stack_size, maxlen=stack_size)
        state = np.array(state_stack) 

        episode_reward = 0

        while True:
            # Action selection - Pure exploitation
            action = agent.act(state)

            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_obs = preprocess_frame(next_obs)
            next_state_stack = state_stack.copy()
            next_state_stack.append(next_obs)
            next_state = np.array(next_state_stack)
            
            state = next_state
            state_stack = next_state_stack
            episode_reward += reward

            time.sleep(args.delay)

            if terminated or truncated:
                print(f"Evaluation Episode {e+1}/{args.episodes} | Score: {episode_reward:7.1f} | Steps: {info['step']} | Hit: {info['hit']}")
                time.sleep(1) # Pause briefy before next episode
                break
                
    env.close()

if __name__ == "__main__":
    main()
