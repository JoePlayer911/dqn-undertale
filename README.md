# Undertale AI Dodger & Gym Environment
Activate the virtual environment first!

### 1. How to use the Procedural Dodger logic directly in-game:
1. Start Undertale and enter a battle.
2. Run `python procedural_dodger.py -ui`. (`-ui` gives you a window to see what the AI sees)
3. Focus back on the Undertale window.
4. Press **`Q`** to activate the AI dodger. Watch it dodge!
5. Press **`Q`** again if you want to take back control, or **`ESC`** to quit.

### 2. Testing the Procedural Dodger in the Gym Environment:
This bypasses the game and runs the procedural CV logic against the custom Gym engine.
```bash
python test_procedural_in_gym.py -ui -d 30 -n 50 rain_down 
```
- **`pattern`**: `rain_down`, `aimed`, `mixed`, `random`, `rain_sides`
- **`-ui`**: Open a GUI window
- **`-d X`**: Difficulty from 0 to 100
- **`-n X`**: Number of episodes to run

---

### 3. Training the DQN Agent
Run `train_dqn.py` to train a Deep Q-Network on the Undertale Gym. It automatically saves checkpoints to `models/` and resumes from the newest one.
```bash
python train_dqn.py --render --difficulty 50 --pattern random --episodes 500
```
- **`--render`**: Watch the AI in a GUI window while it trains (*warning: slows down training*).
- **`--episodes X`**: Train for X episodes (default: 500).
- **`--difficulty X`**: Set the bullet difficulty (0-100) (default: 50).
- **`--pattern NAME`**: Choose the bullet pattern (`random`, `rain_down`, `aimed`, `mixed`, or `rain_sides`).
- **`--no-resume`**: Start training from scratch instead of auto-loading the latest checkpoint.
- **`--model PATH`**: Explicitly specify a checkpoint `.pth` file to load.

### 4. Evaluating the Trained DQN Agent
Run `eval_dqn.py` to watch your trained AI play at its absolute best (Forces purely greedy actions; zero random exploration).
```bash
python eval_dqn.py --episodes 10
```
- **`--episodes X`**: Number of rounds to play (default: 10).
- **`--delay X`**: Delay between frames to make it easier to watch (default: 0.033s).
- **`--model PATH`**: Path to a specific `.pth` file, or directory to auto-load the latest (default: `models`).