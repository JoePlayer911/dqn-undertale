"""
Test the procedural dodger's vision logic against the Undertale Gymnasium.
This bridges the CV-based decision-making from procedural_dodger.py 
with the undertale_gym.py environment.
"""

import cv2
import numpy as np
import time
import sys
import math

from undertale_gym import UndertaleEnv, MASK_LEFT, MASK_TOP, MASK_RIGHT, MASK_BOTTOM

# ─── Import vision logic from procedural_dodger ───
LOWER_WHITE = np.array([200, 200, 200], dtype=np.uint8)
UPPER_WHITE = np.array([255, 255, 255], dtype=np.uint8)


def find_heart(img_bgr):
    """Find the red heart in the image (same logic as procedural_dodger)."""
    lower_red1 = np.array([0, 0, 180], dtype=np.uint8)
    upper_red1 = np.array([80, 80, 255], dtype=np.uint8)
    mask = cv2.inRange(img_bgr, lower_red1, upper_red1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 5:
            x, y, w, h = cv2.boundingRect(largest)
            return (x + w // 2, y + h // 2, w, h)
    return None


def extract_grid_threats(img_bgr, white_mask, hx, hy, B, play_area, grid_size=11):
    """Dynamic NxN grid threat extraction (same as procedural_dodger)."""
    c = grid_size // 2
    x_bounds = [int(hx + (i - c - 0.5) * B) for i in range(grid_size + 1)]
    y_bounds = [int(hy + (j - c - 0.5) * B) for j in range(grid_size + 1)]
    
    h, w = img_bgr.shape[:2]
    mask_left, mask_top, mask_right, mask_bottom = play_area
    dir_names = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']
    threats = {r: {d: 0 for d in dir_names} for r in range(1, c + 1)}
    
    for row in range(grid_size):
        for col in range(grid_size):
            dc = col - c
            dr = row - c
            ring = max(abs(dc), abs(dr))
            if ring == 0:
                continue
            
            dir_name = None
            if dr == -ring and dc == 0: dir_name = 'up'
            elif dr == ring and dc == 0: dir_name = 'down'
            elif dc == -ring and dr == 0: dir_name = 'left'
            elif dc == ring and dr == 0: dir_name = 'right'
            elif (dr == -ring and dc < 0) or (dc == -ring and dr < 0): dir_name = 'up_left'
            elif (dr == -ring and dc > 0) or (dc == ring and dr < 0): dir_name = 'up_right'
            elif (dr == ring and dc < 0) or (dc == -ring and dr > 0): dir_name = 'down_left'
            elif (dr == ring and dc > 0) or (dc == ring and dr > 0): dir_name = 'down_right'
            
            if dir_name is None:
                continue
            
            x1, x2 = x_bounds[col], x_bounds[col+1]
            y1, y2 = y_bounds[row], y_bounds[row+1]
            
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            oob_penalty = 0
            if cx < mask_left or cx >= mask_right or cy < mask_top or cy >= mask_bottom:
                oob_penalty = 100
            
            x1_c = max(0, min(w, x1))
            y1_c = max(0, min(h, y1))
            x2_c = max(0, min(w, x2))
            y2_c = max(0, min(h, y2))
            
            t_count = oob_penalty
            if x1_c < x2_c and y1_c < y2_c:
                t_count += np.count_nonzero(white_mask[y1_c:y2_c, x1_c:x2_c])
            
            threats[ring][dir_name] += t_count
    
    return threats


def procedural_decide(img_bgr, prev_threats, play_area, B=10, grid_size=11):
    """
    Run the procedural dodger's full decision pipeline on a frame.
    Returns: (action_index, curr_threats, debug_info)
    """
    # Action name → gym action index
    action_to_idx = {
        'stay': 0, 'up': 1, 'down': 2, 'left': 3, 'right': 4,
        'up_left': 5, 'up_right': 6, 'down_left': 7, 'down_right': 8
    }
    
    heart_pos = find_heart(img_bgr)
    if not heart_pos:
        return 0, prev_threats, {"heart_found": False}
    
    hx, hy, hw, hh = heart_pos
    mask_left, mask_top, mask_right, mask_bottom = play_area
    h, w = img_bgr.shape[:2]
    
    # Precompute white mask
    white_mask = cv2.inRange(img_bgr, LOWER_WHITE, UPPER_WHITE)
    
    # Grid threats
    c = grid_size // 2
    curr_threats = extract_grid_threats(img_bgr, white_mask, hx, hy, B, play_area, grid_size)
    
    if prev_threats is None:
        prev_threats = curr_threats
    
    # Direction scores
    direction_scores = {d: 0 for d in curr_threats[1].keys()}
    
    opposite_dirs = {
        'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left',
        'up_left': 'down_right', 'up_right': 'down_left',
        'down_left': 'up_right', 'down_right': 'up_left'
    }
    
    # Linear weights
    for dir_name in curr_threats[1].keys():
        score = 0
        for r in range(1, c + 1):
            weight = (c - r + 1)
            score += curr_threats[r][dir_name] * weight
        direction_scores[dir_name] = score
    
    # Incoming check
    for dir_name in curr_threats[1].keys():
        is_incoming = False
        for r in range(1, c):
            if prev_threats[r+1][dir_name] > 5 and curr_threats[r][dir_name] > prev_threats[r][dir_name]:
                is_incoming = True
                break
        if is_incoming:
            direction_scores[dir_name] += 500
            opp_dir = opposite_dirs[dir_name]
            direction_scores[opp_dir] -= 50
    
    # Global flee instinct
    arena_roi = img_bgr[mask_top:mask_bottom, mask_left:mask_right]
    white_arena = cv2.inRange(arena_roi, LOWER_WHITE, UPPER_WHITE)
    y_coords, x_coords = np.nonzero(white_arena)
    
    if len(y_coords) > 0:
        avg_x = np.mean(x_coords) + mask_left
        avg_y = np.mean(y_coords) + mask_top
        
        bullet_dx = avg_x - hx
        bullet_dy = avg_y - hy
        distance = max(1, np.sqrt(bullet_dx**2 + bullet_dy**2))
        norm_bx = bullet_dx / distance
        norm_by = bullet_dy / distance
        
        dir_vectors = {
            'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0),
            'up_left': (-0.707, -0.707), 'up_right': (0.707, -0.707),
            'down_left': (-0.707, 0.707), 'down_right': (0.707, 0.707)
        }
        
        flee_multiplier = 80
        for dir_name, (vx, vy) in dir_vectors.items():
            dot_product = (vx * norm_bx) + (vy * norm_by)
            direction_scores[dir_name] += (dot_product * flee_multiplier)
    
    # Center bias
    arena_cx = (mask_left + mask_right) / 2
    arena_cy = (mask_top + mask_bottom) / 2
    center_dx = arena_cx - hx
    center_dy = arena_cy - hy
    center_dist = max(1, np.sqrt(center_dx**2 + center_dy**2))
    center_pull = min(center_dist * 0.5, 60)
    norm_cx = center_dx / center_dist
    norm_cy = center_dy / center_dist
    
    dir_vectors = {
        'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0),
        'up_left': (-0.707, -0.707), 'up_right': (0.707, -0.707),
        'down_left': (-0.707, 0.707), 'down_right': (0.707, 0.707)
    }
    for dir_name, (vx, vy) in dir_vectors.items():
        dot = (vx * norm_cx) + (vy * norm_cy)
        direction_scores[dir_name] -= (dot * center_pull)
    
    # Decision
    inner_threat_total = sum(curr_threats[1].values()) + sum(curr_threats[2].values())
    
    if inner_threat_total > 10:
        action_name = min(direction_scores, key=direction_scores.get)
    else:
        action_name = 'stay'
    
    action_idx = action_to_idx[action_name]
    return action_idx, curr_threats, {"heart_found": True, "action": action_name, "hx": hx, "hy": hy}


# ─── Main Benchmark ───
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Procedural Dodger Benchmark")
    parser.add_argument("pattern", nargs="?", default="random", help="Bullet pattern")
    parser.add_argument("-ui", action="store_true", help="Show visual window")
    parser.add_argument("-d", "--difficulty", type=int, default=50, help="Difficulty 0-100")
    parser.add_argument("-n", "--episodes", type=int, default=20, help="Number of episodes")
    args = parser.parse_args()

    show_ui = args.ui
    pattern = args.pattern
    num_episodes = args.episodes
    difficulty = args.difficulty
    
    print(f"{'='*60}")
    print(f" Procedural Dodger vs Undertale Gymnasium")
    print(f" Pattern: {pattern} | Difficulty: {difficulty} | Episodes: {num_episodes} | UI: {show_ui}")
    print(f"{'='*60}")

    env = UndertaleEnv(
        render_mode="human" if show_ui else None,
        pattern=pattern,
        max_steps=3000,
        difficulty=difficulty
    )

    play_area = (MASK_LEFT, MASK_TOP, MASK_RIGHT, MASK_BOTTOM)
    
    all_steps = []
    all_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        prev_threats = None
        episode_reward = 0
        step = 0

        while True:
            action, prev_threats, debug = procedural_decide(obs, prev_threats, play_area)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1

            if terminated or truncated:
                break

        survived = "SURVIVED" if not terminated else "HIT"
        all_steps.append(step)
        all_rewards.append(episode_reward)
        print(f"  Episode {ep+1:3d}/{num_episodes} | {survived:8s} | Steps: {step:5d} | Reward: {episode_reward:8.1f}")

    env.close()

    # Summary
    print(f"\n{'='*60}")
    print(f" RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Episodes:       {num_episodes}")
    print(f"  Pattern:        {pattern}")
    print(f"  Avg Steps:      {np.mean(all_steps):.0f}")
    print(f"  Max Steps:      {max(all_steps)}")
    print(f"  Min Steps:      {min(all_steps)}")
    print(f"  Avg Reward:     {np.mean(all_rewards):.1f}")
    print(f"  Survival Rate:  {sum(1 for s in all_steps if s >= 3000) / num_episodes * 100:.0f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
