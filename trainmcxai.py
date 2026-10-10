import os
# Force CPU-only execution and activate oneDNN vectorization before TF loads
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import sys
import json
import random
import signal
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers  # type: ignore

# ─── CPU OPTIMIZATION & THREADING ────────────────────────────────────────────
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)
print("[HARDWARE] Operating in CPU Mode with oneDNN Vectorization.")


# ─── MUNEX CONFIGURATION & PATHS ─────────────────────────────────────────────
BASE_MAZE_SIZE = 5     # Starting Munex grid size (5x5x5)
MIN_MAZE_SIZE = 5      # Minimum boundary size after hunter shrinkage
MAX_MAZE_SIZE = 216000      # Maximum boundary size (Hard-capped by 64-byte ESP32 bitfield limit: 8x8x8 = 512 bits)

STATE_SIZE = 15        # [mx, my, mz, gx, gy, gz, hx, hy, hz, wall_e, wall_w, wall_n, wall_s, wall_u, wall_d]
ACTION_SIZE = 6        # 0: East (+X), 1: West (-X), 2: North (+Y), 3: South (-Y), 4: Up (+Z), 5: Down (-Z)

NUM_PARALLEL_MAZES = 16  # Optimized parallel workers for CPU core saturation
BATCH_SIZE = 256

MODEL_CHECKPOINT_FILE = "munex_miner_model.keras"
META_CHECKPOINT_FILE = "munex_miner_meta.json"
HEADER_OUTPUT_FILE = "model_weights.h"

DIR_DX = np.array([1, -1, 0, 0, 0, 0], dtype=np.int8)
DIR_DY = np.array([0, 0, 1, -1, 0, 0], dtype=np.int8)
DIR_DZ = np.array([0, 0, 0, 0, 1, -1], dtype=np.int8)

STOP_REQUESTED = False


def signal_handler(sig, frame):
    global STOP_REQUESTED
    print("\n\n[SIGNAL] Interrupt received! Completing step & saving checkpoint...")
    STOP_REQUESTED = True


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ─── 3D BITFIELD & MUNEX SPATIAL MAZE GENERATOR ──────────────────────────────
def coord_to_idx(x: int, y: int, z: int) -> int:
    return (int(z) << 6) | (int(y) << 3) | int(x)


class MunexSpatialMaze:
    """ Generates scalable 3D spatial mazes matching the 64-byte ESP32 bitfield layout """
    def __init__(self, size=BASE_MAZE_SIZE):
        self.size = min(size, MAX_MAZE_SIZE)
        self.wall_bitmap = np.zeros(64, dtype=np.uint8)
        self.generate_solvable_maze()

    def set_wall(self, x: int, y: int, z: int):
        if 0 <= x < 8 and 0 <= y < 8 and 0 <= z < 8:
            idx = coord_to_idx(x, y, z)
            self.wall_bitmap[idx >> 3] |= (1 << (idx & 7))

    def clear_wall(self, x: int, y: int, z: int):
        if 0 <= x < 8 and 0 <= y < 8 and 0 <= z < 8:
            idx = coord_to_idx(x, y, z)
            self.wall_bitmap[idx >> 3] &= (0xFF ^ (1 << (idx & 7)))

    def is_wall(self, x: int, y: int, z: int) -> bool:
        if x < 0 or x >= self.size or y < 0 or y >= self.size or z < 0 or z >= self.size:
            return True
        idx = coord_to_idx(x, y, z)
        return bool(self.wall_bitmap[idx >> 3] & (1 << (idx & 7)))

    def generate_solvable_maze(self):
        self.wall_bitmap.fill(0)
        
        for x in range(self.size):
            for y in range(self.size):
                for z in range(self.size):
                    self.set_wall(x, y, z)

        visited = set()
        stack = [(0, 0, 0)]
        self.clear_wall(0, 0, 0)
        visited.add((0, 0, 0))

        while stack:
            cx, cy, cz = stack[-1]
            neighbors = []

            for d in range(6):
                nx = cx + DIR_DX[d] * 2
                ny = cy + DIR_DY[d] * 2
                nz = cz + DIR_DZ[d] * 2
                if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                    if (nx, ny, nz) not in visited:
                        neighbors.append((d, nx, ny, nz))

            if neighbors:
                d, nx, ny, nz = random.choice(neighbors)
                mx, my, mz = cx + DIR_DX[d], cy + DIR_DY[d], cz + DIR_DZ[d]
                self.clear_wall(mx, my, mz)
                self.clear_wall(nx, ny, nz)
                visited.add((nx, ny, nz))
                stack.append((nx, ny, nz))
            else:
                stack.pop()


# ─── MUNEX SPATIAL ENVIRONMENT WORKER ────────────────────────────────────────
class MunexEnvironment:
    """ Manages persistent spatial state, dynamic endpoint mutation, and hunter entity """
    def __init__(self, initial_size=BASE_MAZE_SIZE):
        self.maze_size = initial_size
        self.reset()

    def reset(self):
        self.maze = MunexSpatialMaze(size=self.maze_size)
        self.mx = random.randint(0, self.maze_size - 1)
        self.my = random.randint(0, self.maze_size - 1)
        self.mz = random.randint(0, self.maze_size - 1)
        self.maze.clear_wall(self.mx, self.my, self.mz)

        # Dynamic endpoint
        self.gx = random.randint(0, self.maze_size - 1)
        self.gy = random.randint(0, self.maze_size - 1)
        self.gz = random.randint(0, self.maze_size - 1)
        while (self.gx, self.gy, self.gz) == (self.mx, self.my, self.mz):
            self.gx = random.randint(0, self.maze_size - 1)
            self.gy = random.randint(0, self.maze_size - 1)
            self.gz = random.randint(0, self.maze_size - 1)
        self.maze.clear_wall(self.gx, self.gy, self.gz)

        # Hunter entity
        self.hx = random.randint(0, self.maze_size - 1)
        self.hy = random.randint(0, self.maze_size - 1)
        self.hz = random.randint(0, self.maze_size - 1)

        self.steps = 0
        self.done = False
        self.total_reward = 0.0

    def get_state_vector(self) -> np.ndarray:
        s = float(self.maze_size)
        return np.array([
            self.mx / s, self.my / s, self.mz / s,
            self.gx / s, self.gy / s, self.gz / s,
            self.hx / s, self.hy / s, self.hz / s,
            1.0 if self.maze.is_wall(self.mx + 1, self.my, self.mz) else 0.0,
            1.0 if self.maze.is_wall(self.mx - 1, self.my, self.mz) else 0.0,
            1.0 if self.maze.is_wall(self.mx, self.my + 1, self.mz) else 0.0,
            1.0 if self.maze.is_wall(self.mx, self.my - 1, self.mz) else 0.0,
            1.0 if self.maze.is_wall(self.mx, self.my, self.mz + 1) else 0.0,
            1.0 if self.maze.is_wall(self.mx, self.my, self.mz - 1) else 0.0
        ], dtype=np.float32)

    def compute_spatial_pow(self) -> int:
        """ Munex Spatial PoW Challenge tied to current coordinates and bitfield state """
        seed_data = f"{self.mx},{self.my},{self.mz}:{self.maze.wall_bitmap.tobytes().hex()}"
        nonce = 0
        while nonce < 500:
            h = hashlib.sha256(f"{seed_data}:{nonce}".encode()).hexdigest()
            if h.startswith("00"):
                return nonce
            nonce += 1
        return nonce

    def step(self, action: int):
        self.steps += 1
        state = self.get_state_vector()

        nx = self.mx + DIR_DX[action]
        ny = self.my + DIR_DY[action]
        nz = self.mz + DIR_DZ[action]

        if nx < 0 or nx >= self.maze_size or ny < 0 or ny >= self.maze_size or nz < 0 or nz >= self.maze_size or self.maze.is_wall(nx, ny, nz):
            pow_nonce = self.compute_spatial_pow()
            reward = -15.0 - (5.0 * (pow_nonce / 100.0))
            next_state = state
        elif nx == self.gx and ny == self.gy and nz == self.gz:
            # Successfully reaching endpoint expands territory up to 8x8x8 limit
            reward = 120.0
            if self.maze_size < MAX_MAZE_SIZE:
                self.maze_size += 1
            self.done = True
            next_state = state
        else:
            dist_curr = abs(self.mx - self.gx) + abs(self.my - self.gy) + abs(self.mz - self.gz)
            dist_next = abs(nx - self.gx) + abs(ny - self.gy) + abs(nz - self.gz)
            reward = 6.0 if dist_next < dist_curr else -2.5

            self.mx, self.my, self.mz = nx, ny, nz
            next_state = self.get_state_vector()

        # Hunter pursuit
        if random.random() < 0.6:
            if self.hx < self.mx: self.hx += 1
            elif self.hx > self.mx: self.hx -= 1
            if self.hy < self.my: self.hy += 1
            elif self.hy > self.my: self.hy -= 1
            if self.hz < self.mz: self.hz += 1
            elif self.hz > self.mz: self.hz -= 1

        # Hunter capture shrinks territory
        if abs(self.mx - self.hx) + abs(self.my - self.hy) + abs(self.mz - self.hz) <= 1:
            reward -= 25.0
            if self.maze_size > MIN_MAZE_SIZE:
                self.maze_size -= 1
            self.done = True

        # Dynamic endpoint mutation
        if self.steps % 10 == 0 and not self.done:
            self.gx = random.randint(0, self.maze_size - 1)
            self.gy = random.randint(0, self.maze_size - 1)
            self.gz = random.randint(0, self.maze_size - 1)

        if self.steps >= 60:
            self.done = True

        self.total_reward += reward
        return state, action, reward, next_state, self.done


# ─── MICRO-MLP MODEL BUILDER ─────────────────────────────────────────────────
def build_dqn_model() -> tf.keras.Model:
    with tf.device("/CPU:0"):
        inputs = layers.Input(shape=(STATE_SIZE,), dtype=tf.float32)
        x = layers.Dense(32, activation='relu', name='fc1')(inputs)
        x = layers.Dense(16, activation='relu', name='fc2')(x)
        outputs = layers.Dense(ACTION_SIZE, activation='linear', name='fc3')(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="Munex_CPU_Brain")
        model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='huber')
    return model


# ─── C++ PROGMEM WEIGHT EXPORTER ──────────────────────────────────────────────
def export_weights_to_header(model: tf.keras.Model, filename=HEADER_OUTPUT_FILE):
    weights = model.get_weights()
    
    fc1_w = weights[0].T.flatten()
    fc1_b = weights[1]
    fc2_w = weights[2].T.flatten()
    fc2_b = weights[3]
    fc3_w = weights[4].T.flatten()
    fc3_b = weights[5]

    with open(filename, "w") as f:
        f.write("/* Munex Spatial Contract Header - Auto-generated for ESP32C3 */\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n#include <Arduino.h>\n\n")

        def write_cpp_array(arr_name, data):
            f.write(f"const float {arr_name}[{len(data)}] PROGMEM = {{\n  ")
            formatted = [f"{v:.6f}f" for v in data]
            for i in range(0, len(formatted), 8):
                line = ", ".join(formatted[i:i+8])
                f.write(line + (",\n  " if i + 8 < len(formatted) else "\n"))
            f.write("};\n\n")

        write_cpp_array("fc1_weight", fc1_w)
        write_cpp_array("fc1_bias", fc1_b)
        write_cpp_array("fc2_weight", fc2_w)
        write_cpp_array("fc2_bias", fc2_b)
        write_cpp_array("fc3_weight", fc3_w)
        write_cpp_array("fc3_bias", fc3_b)

        f.write("#endif // MODEL_WEIGHTS_H\n")

    print(f"[EXPORT] Embedded Munex spatial header written to '{filename}'")


# ─── CHECKPOINT MANAGEMENT ────────────────────────────────────────────────────
def save_checkpoint(model: tf.keras.Model, episode: int, epsilon: float, maze_size: int):
    print(f"\n[CHECKPOINT] Saving Munex state at Episode {episode} (Size: {maze_size}x{maze_size}x{maze_size}, Epsilon: {epsilon:.4f})...")
    model.save(MODEL_CHECKPOINT_FILE)
    
    meta = {
        "episode": episode,
        "epsilon": epsilon,
        "maze_size": maze_size
    }
    with open(META_CHECKPOINT_FILE, "w") as f:
        json.dump(meta, f, indent=4)
        
    export_weights_to_header(model)
    print("[CHECKPOINT] Checkpoint & C++ header updated successfully!")


def load_checkpoint():
    if os.path.exists(MODEL_CHECKPOINT_FILE) and os.path.exists(META_CHECKPOINT_FILE):
        print(f"[AUTO-RESUME] Found existing checkpoint '{MODEL_CHECKPOINT_FILE}'. Resuming...")
        model = tf.keras.models.load_model(MODEL_CHECKPOINT_FILE)
        
        with open(META_CHECKPOINT_FILE, "r") as f:
            meta = json.load(f)
            
        start_episode = meta.get("episode", 0) + 1
        epsilon = meta.get("epsilon", 1.0)
        curr_size = meta.get("maze_size", BASE_MAZE_SIZE)
        return model, start_episode, epsilon, curr_size
    else:
        print("[INIT] No previous checkpoint found. Starting fresh training session...")
        return build_dqn_model(), 1, 1.0, BASE_MAZE_SIZE


# ─── OPTIMIZED CPU TRAINING LOOP ──────────────────────────────────────────────
def train():
    global STOP_REQUESTED

    model, start_episode, epsilon, current_maze_size = load_checkpoint()
    target_model = build_dqn_model()
    target_model.set_weights(model.get_weights())

    gamma = 0.95
    epsilon_min = 0.05
    epsilon_decay = 0.99997

    replay_buffer = []
    max_buffer = 100000

    print(f"=== Munex AI CPU Training Active ===")

    envs = [MunexEnvironment(initial_size=current_maze_size) for _ in range(NUM_PARALLEL_MAZES)]
    episodes_completed = start_episode
    recent_rewards = []

    try:
        while not STOP_REQUESTED:
            states = np.array([env.get_state_vector() for env in envs], dtype=np.float32)
            
            with tf.device("/CPU:0"):
                q_values_batch = model(states, training=False).numpy()

            for idx, env in enumerate(envs):
                if random.random() <= epsilon:
                    action = random.randint(0, ACTION_SIZE - 1)
                else:
                    action = int(np.argmax(q_values_batch[idx]))

                st, act, rew, nst, done = env.step(action)
                replay_buffer.append((st, act, rew, nst, done))

                if done:
                    recent_rewards.append(env.total_reward)
                    episodes_completed += 1

                    if epsilon > epsilon_min:
                        epsilon *= epsilon_decay

                    # Track current average maze size across workers for checkpointing
                    current_maze_size = env.maze_size
                    env.reset()

            if len(replay_buffer) > max_buffer:
                replay_buffer = replay_buffer[-max_buffer:]

            if len(replay_buffer) >= BATCH_SIZE:
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                b_states = np.array([b[0] for b in minibatch], dtype=np.float32)
                b_actions = np.array([b[1] for b in minibatch])
                b_rewards = np.array([b[2] for b in minibatch], dtype=np.float32)
                b_next_states = np.array([b[3] for b in minibatch], dtype=np.float32)
                b_dones = np.array([b[4] for b in minibatch], dtype=np.float32)

                with tf.device("/CPU:0"):
                    target_q = model.predict_on_batch(b_states)
                    next_q = target_model.predict_on_batch(b_next_states)

                    for i in range(BATCH_SIZE):
                        if b_dones[i]:
                            target_q[i][b_actions[i]] = b_rewards[i]
                        else:
                            target_q[i][b_actions[i]] = b_rewards[i] + gamma * np.amax(next_q[i])

                    model.train_on_batch(b_states, target_q)

            if episodes_completed % 500 == 0 and recent_rewards:
                target_model.set_weights(model.get_weights())
                avg_reward = np.mean(recent_rewards[-100:])
                print(f"Episode {episodes_completed:8d} (CPU Munex) | Avg Reward: {avg_reward:6.1f} | Territory: {current_maze_size}x{current_maze_size}x{current_maze_size} | Epsilon: {epsilon:.4f} | Buffer: {len(replay_buffer)}")

            if episodes_completed % 5000 == 0 and episodes_completed > start_episode:
                save_checkpoint(model, episodes_completed, epsilon, current_maze_size)

    except (KeyboardInterrupt, EOFError):
        print("\n[INTERRUPT] Exit request received from user terminal.")

    save_checkpoint(model, episodes_completed, epsilon, current_maze_size)


if __name__ == "__main__":
    train()