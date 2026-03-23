import time
import cv2
import numpy as np
import mss
import keyboard
import pydirectinput
import ctypes
from ctypes.wintypes import RECT
import sys

# Disable pydirectinput failsafe to prevent crashes when moving mouse to corners
pydirectinput.FAILSAFE = False

# Define bullet color range (UnderTale bullets are usually white/light blue)
LOWER_WHITE = np.array([200, 200, 200], dtype=np.uint8)
UPPER_WHITE = np.array([255, 255, 255], dtype=np.uint8)

# Define heart color range in HSV space (Red has two ranges)
LOWER_RED1 = np.array([0, 150, 150], dtype=np.uint8)
UPPER_RED1 = np.array([10, 255, 255], dtype=np.uint8)
LOWER_RED2 = np.array([170, 150, 150], dtype=np.uint8)
UPPER_RED2 = np.array([180, 255, 255], dtype=np.uint8)

def get_undertale_window():
    """Uses Windows API to find the UNDERTALE window and return its bounding box."""
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, "UNDERTALE")
    if hwnd:
        rect = RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return {
            'top': rect.top,
            'left': rect.left,
            'width': rect.right - rect.left,
            'height': rect.bottom - rect.top
        }
    return None

def find_heart(img_bgr):
    """Locate the red player heart using color masks."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        valid_contours = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 10 and area < 5000: # Filter out noise and massive objects
                x, y, w, h = cv2.boundingRect(c)
                aspect_ratio = float(w) / max(1, h)
                # The player heart is relatively square. 
                # A health bar is very wide (aspect_ratio > 3)
                if 0.5 <= aspect_ratio <= 2.0:
                    valid_contours.append(c)
                    
        if valid_contours:
            # Out of the valid square-ish red objects, the heart is usually the largest
            best_contour = max(valid_contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(best_contour)
            center_x = x + w // 2
            center_y = y + h // 2
            return (center_x, center_y, w, h)
    return None

def extract_grid_threats_dynamic(img_bgr, white_mask, hx, hy, B, play_area, grid_size=13, show_ui=False):
    """
    Extract threat counts for 8 directions using a dynamic NxN grid topology centered on the heart.
    Uses a precomputed white_mask for massive performance gains.
    Returns a dictionary of rings, each containing a dictionary of 8 directional threats.
    """
    c = grid_size // 2
    
    # X and Y bounds (0 to grid_size)
    x_bounds = [int(hx + (i - c - 0.5) * B) for i in range(grid_size + 1)]
    y_bounds = [int(hy + (j - c - 0.5) * B) for j in range(grid_size + 1)]
    
    h, w = img_bgr.shape[:2]
    mask_left, mask_top, mask_right, mask_bottom = play_area
    dir_names = ['up', 'down', 'left', 'right', 'up_left', 'up_right', 'down_left', 'down_right']
    
    threats = { r: { d: 0 for d in dir_names } for r in range(1, c + 1) }
    
    for row in range(grid_size):
        for col in range(grid_size):
            dc = col - c
            dr = row - c
            ring = max(abs(dc), abs(dr))
            
            if ring == 0: continue # Skip heart center
            
            # Identify generalized direction for this box mathematically
            dir_name = None
            if dr == -ring and dc == 0: dir_name = 'up'
            elif dr == ring and dc == 0: dir_name = 'down'
            elif dc == -ring and dr == 0: dir_name = 'left'
            elif dc == ring and dr == 0: dir_name = 'right'
            elif (dr == -ring and dc < 0) or (dc == -ring and dr < 0): dir_name = 'up_left'
            elif (dr == -ring and dc > 0) or (dc == ring and dr < 0): dir_name = 'up_right'
            elif (dr == ring and dc < 0) or (dc == -ring and dr > 0): dir_name = 'down_left'
            elif (dr == ring and dc > 0) or (dc == ring and dr > 0): dir_name = 'down_right'
            
            if dir_name is None: continue # Mathematical edge case safeguard
            
            x1, x2 = x_bounds[col], x_bounds[col+1]
            y1, y2 = y_bounds[row], y_bounds[row+1]
            
            # Determine if this box is out of bounds or inside our purple masks
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            oob_penalty = 0
            if cx < mask_left or cx >= mask_right or cy < mask_top or cy >= mask_bottom:
                oob_penalty = 100 # Treat out of bounds as a solid block of bullets
            
            x1_c, y1_c = max(0, min(w, x1)), max(0, min(h, y1))
            x2_c, y2_c = max(0, min(w, x2)), max(0, min(h, y2))
            
            t_count = oob_penalty
            if x1_c < x2_c and y1_c < y2_c:
                # Use precomputed mask — just sum the region (ultra fast!)
                t_count += np.count_nonzero(white_mask[y1_c:y2_c, x1_c:x2_c])
                
            threats[ring][dir_name] += t_count
            
            if show_ui:
                # Dynamic color shift: Green (Inner) to Blue (Outer)
                ratio = (ring - 1) / max(1, (c - 1))
                color = (int(255 * ratio), int(255 * (1 - ratio)), 0)
                draw_c = (0, 0, 255) if t_count > 5 else color
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), draw_c, 1)

    return threats

def get_keys_for_direction(direction):
    """Maps a 8-way direction string to keys to press."""
    mapping = {
        'up': ['up'],
        'down': ['down'],
        'left': ['left'],
        'right': ['right'],
        'up_left': ['up', 'left'],
        'up_right': ['up', 'right'],
        'down_left': ['down', 'left'],
        'down_right': ['down', 'right'],
        'stay': []
    }
    return mapping.get(direction, [])

def main():
    show_ui = "-ui" in sys.argv
    
    print("Starting Advanced Procedural Dodger in 3 seconds. Please focus the game window.")
    print("Press 'q' to ENALBE/DISABLE the dodger.")
    print("Press 'esc' at any time to completely close the script.")
    if show_ui:
        print("\n[INFO] UI Mode enabled! Opening vision window when active.")
    time.sleep(2)
    
    sct = mss.mss()
    monitor = get_undertale_window()
    
    last_pressed_keys = []
    
    # Store threats from the previous frame to track movement across all rings
    prev_threats = None
    
    # Heart position memory: keep the last known position for up to N frames
    last_known_heart = None
    heart_lost_frames = 0
    MAX_HEART_LOST_FRAMES = 10 # Keep dodging for up to 10 frames after losing sight
    
    # FPS tracking
    fps_timer = time.time()
    frame_count = 0
    
    is_active = False
    print("\n--- Dodger is currently PAUSED ---")
    
    last_print_time = time.time()
    
    try:
        while True:
            if keyboard.is_pressed('esc'):
                print("Quit signal detected. Stopping.")
                break
                
            # Toggle Active State
            if keyboard.is_pressed('q'):
                is_active = not is_active
                status = "ACTIVE" if is_active else "PAUSED"
                print(f"[{time.strftime('%H:%M:%S')}] Agent {status}")
                
                if not is_active:
                    for key in last_pressed_keys:
                        pydirectinput.keyUp(key)
                    last_pressed_keys = []
                    # Reset memory so it doesn't think stagnant bullets are incoming when unpaused
                    prev_threats = None
                    
                    if show_ui:
                        cv2.destroyAllWindows()
                    
                time.sleep(0.3) # Debounce for the toggle button
                continue
                
            if not is_active:
                time.sleep(0.05)
                continue
                
            # Dynamically fetch window in case it moved
            window_rect = get_undertale_window()
            if window_rect:
                monitor = window_rect
            elif not hasattr(main, "warned"):
                print("WARNING: Could not find 'UNDERTALE' window. Grabbing entire primary monitor instead.")
                main.warned = True

            sct_img = sct.grab(monitor)
            # Use cv2 to convert from BGRA to BGR to ensure memory contiguity
            img_bgr = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)
            
            # Black out the top 1/3 of the window to ignore the red title bar icon
            # The actual battle box is always in the lower portion of the screen
            # You can fine-tune these to perfectly frame the Undertale battle box!
            h, w = img_bgr.shape[:2]
            mask_top = (h // 2) + 26
            mask_bottom = (h - (h // 6)) - 14
            mask_left = (w // 3) + 29
            mask_right = (w - (w // 3)) - 31
            
            # Draw purple masks over everything outside the battle box
            purple = (128, 0, 128) # OpenCV uses BGR format
            # Top mask
            cv2.rectangle(img_bgr, (0, 0), (w, mask_top), purple, -1)
            # Bottom mask
            cv2.rectangle(img_bgr, (0, mask_bottom), (w, h), purple, -1)
            # Left mask (between top and bottom masks)
            cv2.rectangle(img_bgr, (0, mask_top), (mask_left, mask_bottom), purple, -1)
            # Right mask (between top and bottom masks)
            cv2.rectangle(img_bgr, (mask_right, mask_top), (w, mask_bottom), purple, -1)
            
            # Pass this play area into the CV function
            play_area = (mask_left, mask_top, mask_right, mask_bottom)
            
            heart_pos = find_heart(img_bgr)
            
            if not heart_pos:
                heart_lost_frames += 1
                
                if heart_lost_frames > MAX_HEART_LOST_FRAMES or last_known_heart is None:
                    # Truly lost - give up and stop
                    if is_active and time.time() - last_print_time > 2.0:
                        print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Heart NOT FOUND on screen. Check color threshold or monitor grab.")
                        last_print_time = time.time()
                        
                    for key in last_pressed_keys:
                        pydirectinput.keyUp(key)
                    last_pressed_keys = []
                    
                    if show_ui:
                        cv2.putText(img_bgr, "HEART NOT FOUND", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow("Dodger AI Vision", img_bgr)
                        cv2.waitKey(1)
                    else:
                        time.sleep(0.005)
                    continue
                else:
                    # Use last known position and keep dodging!
                    heart_pos = last_known_heart
                
            else:
                # Heart found! Update memory
                last_known_heart = heart_pos
                heart_lost_frames = 0
                
            hx, hy, hw, hh = heart_pos
            
            if is_active and time.time() - last_print_time > 2.0:
                print(f"[{time.strftime('%H:%M:%S')}] DEBUG: Heart tracked at ({hx}, {hy})")
                last_print_time = time.time()
                
            if show_ui:
                # Target square for the heart itself
                cv2.rectangle(img_bgr, (hx - hw//2, hy - hh//2), (hx + hw//2, hy + hh//2), (255, 0, 0), 2)
                
            # Precompute the white bullet mask ONCE for the entire frame (1 call instead of 169!)
            white_mask = cv2.inRange(img_bgr, LOWER_WHITE, UPPER_WHITE)
            
            # Using an 11x11 Grid with 5 rings of depth
            B = 10
            grid_size = 11
            curr_threats = extract_grid_threats_dynamic(img_bgr, white_mask, hx, hy, B, play_area, grid_size, show_ui)
            
            if prev_threats is None:
                prev_threats = curr_threats
            
            # Heuristic calculation:
            direction_scores = {d: 0 for d in curr_threats[1].keys()}
            c = grid_size // 2
            
            # Map opposite directions to reward escaping
            opposite_dirs = {
                'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left',
                'up_left': 'down_right', 'up_right': 'down_left',
                'down_left': 'up_right', 'down_right': 'up_left'
            }
            
            # Linear weights: Inner ring matters most, outer ring least
            for dir_name in curr_threats[1].keys():
                score = 0
                for r in range(1, c + 1):
                    # Linear decay: ring 1 = weight c, ring c = weight 1
                    weight = (c - r + 1)
                    score += curr_threats[r][dir_name] * weight
                direction_scores[dir_name] = score
                
            for dir_name in curr_threats[1].keys():
                # INCOMING CHECK:
                # Iterate through all rings checking if bullet shifted closer
                is_incoming = False
                for r in range(1, c):
                    if prev_threats[r+1][dir_name] > 5 and curr_threats[r][dir_name] > prev_threats[r][dir_name]:
                        is_incoming = True
                        break
                        
                if is_incoming:
                    # Strong penalty but not astronomically absurd
                    direction_scores[dir_name] += 500
                    opp_dir = opposite_dirs[dir_name]
                    direction_scores[opp_dir] -= 50
                    print(f"Danger! Incoming from {dir_name}")
                    
            prev_threats = curr_threats
            
            # --- GLOBAL AWARENESS (FLEE INSTINCT) ---
            # Calculate the Center of Mass of ALL white pixels in the playable arena
            arena_roi = img_bgr[mask_top:mask_bottom, mask_left:mask_right]
            white_mask = cv2.inRange(arena_roi, LOWER_WHITE, UPPER_WHITE)
            y_coords, x_coords = np.nonzero(white_mask)
            
            if len(y_coords) > 0:
                avg_x = np.mean(x_coords) + mask_left
                avg_y = np.mean(y_coords) + mask_top
                
                if show_ui:
                    # Draw an Orange dot for the Global Threat Center, and link it to the Heart
                    cv2.circle(img_bgr, (int(avg_x), int(avg_y)), 5, (0, 165, 255), -1)
                    cv2.line(img_bgr, (int(hx), int(hy)), (int(avg_x), int(avg_y)), (0, 165, 255), 1)
                
                # Apply a global flee-instinct to the direction scores
                bullet_dx = avg_x - hx
                bullet_dy = avg_y - hy
                distance = max(1, np.sqrt(bullet_dx**2 + bullet_dy**2))
                norm_bx = bullet_dx / distance
                norm_by = bullet_dy / distance
                
                # Geometric vectors for the 8 directions
                dir_vectors = {
                    'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0),
                    'up_left': (-0.707, -0.707), 'up_right': (0.707, -0.707),
                    'down_left': (-0.707, 0.707), 'down_right': (0.707, 0.707)
                }
                
                # Add a persistent mathematical "gravity" forcing the agent away from the mass
                flee_multiplier = 80
                for dir_name, (vx, vy) in dir_vectors.items():
                    # Dot product: Positive = moving TOWARDS the mass, Negative = moving AWAY
                    dot_product = (vx * norm_bx) + (vy * norm_by)
                    direction_scores[dir_name] += (dot_product * flee_multiplier)
            
            # Also consider the current position threat (the center)
            # If center is perfectly safe, maybe don't move
            cx1, cy1 = max(0, hx - 15), max(0, hy - 15)
            cx2, cy2 = min(w, hx + 15), min(h, hy + 15)
            center_roi = img_bgr[cy1:cy2, cx1:cx2]
            # --- CENTER BIAS (ANTI-CORNERING) ---
            # Pull the agent toward the center of the arena so it doesn't corner itself
            arena_cx = (mask_left + mask_right) / 2
            arena_cy = (mask_top + mask_bottom) / 2
            center_dx = arena_cx - hx
            center_dy = arena_cy - hy
            center_dist = max(1, np.sqrt(center_dx**2 + center_dy**2))
            
            # The further from center, the stronger the pull
            center_pull = min(center_dist * 0.5, 60) # Caps at 60 to not override real threats
            norm_cx = center_dx / center_dist
            norm_cy = center_dy / center_dist
            
            dir_vectors = {
                'up': (0, -1), 'down': (0, 1), 'left': (-1, 0), 'right': (1, 0),
                'up_left': (-0.707, -0.707), 'up_right': (0.707, -0.707),
                'down_left': (-0.707, 0.707), 'down_right': (0.707, 0.707)
            }
            for dir_name, (vx, vy) in dir_vectors.items():
                # Reward moving TOWARD center (negative dot = away from center = penalty)
                dot = (vx * norm_cx) + (vy * norm_cy)
                direction_scores[dir_name] -= (dot * center_pull)
            
            # Only commit to moving if inner rings (1-2) actually have threats
            inner_threat_total = sum(curr_threats[1].values()) + sum(curr_threats[2].values())
            
            if inner_threat_total > 10:
                # Real danger nearby — dodge!
                action = min(direction_scores, key=direction_scores.get)
            else:
                # No immediate danger — stay put or gently drift to center
                action = 'stay'

            # Actually press the keys
            target_keys = get_keys_for_direction(action)
            
            # Release keys that are no longer needed
            for key in last_pressed_keys:
                if key not in target_keys:
                    pydirectinput.keyUp(key)
            
            # Press new keys
            for key in target_keys:
                if key not in last_pressed_keys:
                    pydirectinput.keyDown(key)
                    
            last_pressed_keys = target_keys
            
            if show_ui:
                # FPS counter
                frame_count += 1
                if time.time() - fps_timer >= 1.0:
                    fps = frame_count / (time.time() - fps_timer)
                    fps_timer = time.time()
                    frame_count = 0
                cv2.putText(img_bgr, f"ACTION: {action}", (11, 41), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(img_bgr, f"ACTION: {action}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                try:
                    cv2.putText(img_bgr, f"FPS: {fps:.0f}", (11, 81), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    cv2.putText(img_bgr, f"FPS: {fps:.0f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except:
                    pass
                cv2.imshow("Dodger AI Vision", img_bgr)
                cv2.waitKey(1)
            else:
                time.sleep(0.005)
            
    except KeyboardInterrupt:
        pass
    finally:
        for key in last_pressed_keys:
            pydirectinput.keyUp(key)
        if "-ui" in sys.argv:
            cv2.destroyAllWindows()
        print("Dodger stopped safely.")

if __name__ == '__main__':
    main()
