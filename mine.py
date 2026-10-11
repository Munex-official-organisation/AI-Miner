#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION & VARIABLES (TUNE HERE)
# ═══════════════════════════════════════════════════════════════════════════════
NODE_URL = "ws://127.0.0.1:8080/ws"
PRIVATE_KEY = ""                                 # Paste your 64-char hex private key here (or leave blank for auto-key)
MODEL_FILENAME = "model.keras"
DEFAULT_MAZE_SIZE = 5
DEBUG_MODE = True                                # Set to True to print raw incoming packets

import sys
import os
import time
import json
import hashlib
import base64
import asyncio
import websockets
import numpy as np
import keras

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric.utils import decode_dss_signature

SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
CHAIN_ID = 6620

ACTION_MAP = {
    0: "east",
    1: "west",
    2: "north",
    3: "south",
    4: "up",
    5: "down"
}

try:
    from Crypto.Hash import keccak as _keccak
    def keccak256(data: bytes) -> bytes:
        k = _keccak.new(digest_bits=256)
        k.update(data)
        return k.digest()
except ImportError:
    def keccak256(data: bytes) -> bytes:
        return hashlib.sha3_256(data).digest()

def sign_bytes(priv_hex: str, message: bytes) -> str:
    priv_int = int(priv_hex, 16)
    priv_key = ec.derive_private_key(priv_int, ec.SECP256K1())
    sig = priv_key.sign(message, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(sig)
    if s > SECP256K1_N // 2:
        s = SECP256K1_N - s
    return f"{r:064x}{s:064x}"

def privkey_to_address(priv_hex: str) -> str:
    priv_key = ec.derive_private_key(int(priv_hex, 16), ec.SECP256K1())
    pub = priv_key.public_key()
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )
    return '0x' + keccak256(pub_bytes[1:])[-20:].hex()

def get_public_key_hex(priv_hex: str) -> str:
    priv_key = ec.derive_private_key(int(priv_hex, 16), ec.SECP256K1())
    pub = priv_key.public_key()
    pub_bytes = pub.public_bytes(
        encoding=serialization.Encoding.X962,
        format=serialization.PublicFormat.UncompressedPoint
    )
    return pub_bytes.hex()

def derive_miner_id(wallet_address: str, pub_key_hex: str) -> str:
    hash_part = hashlib.sha256(f"{wallet_address}_mcx_miner".encode("utf-8")).hexdigest()[:32].upper()
    return f"PC_{hash_part}"

def canonical_json_bytes(data: dict) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

def load_miner_brain():
    if os.path.exists(MODEL_FILENAME):
        print(f"[AI] Loading spatial brain from '{MODEL_FILENAME}'...")
        model = keras.saving.load_model(MODEL_FILENAME)
        print("[AI] 'model.keras' loaded successfully.")
        return model
    else:
        print(f"[ERROR] Required model file '{MODEL_FILENAME}' not found in current directory!", file=sys.stderr)
        sys.exit(1)

model = load_miner_brain()

def maze_grid_to_walls(maze_3d) -> list:
    walls = []
    for z, slice_2d in enumerate(maze_3d):
        for y, row in enumerate(slice_2d):
            for x, is_wall in enumerate(row):
                if is_wall:
                    walls.append([x, y, z])
    return walls

def construct_state_vector(pos, goal, ghost, walls, size) -> np.ndarray:
    mx, my, mz = pos[:3] if len(pos) >= 3 else (pos[0], pos[1], 0)
    gx, gy, gz = goal[:3] if len(goal) >= 3 else (goal[0], goal[1], 0)
    hx, hy, hz = ghost[:3] if len(ghost) >= 3 else (ghost[0], ghost[1], 0)
    s = float(size) if size > 0 else float(DEFAULT_MAZE_SIZE)

    wall_set = set(tuple(w) for w in walls)

    def is_w(wx, wy, wz):
        if wx < 0 or wx >= size or wy < 0 or wy >= size or wz < 0 or wz >= size:
            return 1.0
        return 1.0 if (wx, wy, wz) in wall_set else 0.0

    return np.array([
        mx / s, my / s, mz / s,
        gx / s, gy / s, gz / s,
        hx / s, hy / s, hz / s,
        is_w(mx + 1, my, mz),
        is_w(mx - 1, my, mz),
        is_w(mx, my + 1, mz),
        is_w(mx, my - 1, mz),
        is_w(mx, my, mz + 1),
        is_w(mx, my, mz - 1)
    ], dtype=np.float32)

def solve_wall_pow(seed_hex: str, diff_bits: int = 4) -> int:
    nonce = 0
    target_zeros = diff_bits // 4
    while True:
        candidate = f"{seed_hex}{nonce}".encode("ascii")
        h = hashlib.sha256(candidate).hexdigest()
        if h.startswith("0" * target_zeros):
            return nonce
        nonce += 1

async def make_and_send_move(websocket, privkey, miner_id, session):
    state_vec = construct_state_vector(session["pos"], session["goal"], session["ghost"], session["walls"], session["size"])
    state_tensor = np.array([state_vec], dtype=np.float32)

    q_values = model.predict(state_tensor, verbose=0)[0]
    best_action_idx = int(np.argmax(q_values))
    step_dir = ACTION_MAP.get(best_action_idx, "east")

    session["movenonce"] += 1
    sent_nonce = session["movenonce"]

    auth_msg = canonical_json_bytes({
        "domain": "MCXPOGMOVEV1",
        "chainid": CHAIN_ID,
        "minerid": miner_id,
        "mazeid": session["maze_id"],
        "mazeversion": session["mazeversion"],
        "movenonce": sent_nonce,
        "direction": step_dir,
    })
    sig = sign_bytes(privkey, auth_msg)

    await websocket.send(json.dumps({
        "type": "maze_move",
        "miner_id": miner_id,
        "maze_id": session["maze_id"],
        "direction": step_dir,
        "signature": sig,
        "move_nonce": sent_nonce,
    }))
    print(f"[MINER] Sent move -> {step_dir} (nonce: {sent_nonce}, pos: {session['pos']})")

async def start_desktop_miner():
    privkey = PRIVATE_KEY.strip() or os.getenv("MUNEX_PRIVATE_KEY", "").strip()
    if not privkey:
        privkey = hashlib.sha256(b"munex_desktop_ai_miner_01").hexdigest()
        print(f"[WARNING] No private key provided. Generated fallback key.")

    addr = privkey_to_address(privkey)
    pub_key_hex = get_public_key_hex(privkey)
    miner_id = derive_miner_id(addr, pub_key_hex)

    session = {
        "maze_id": pub_key_hex,
        "size": DEFAULT_MAZE_SIZE,
        "pos": (0, 0, 0),
        "goal": (4, 4, 4),
        "ghost": (2, 2, 2),
        "walls": [],
        "movenonce": 0,
        "mazeversion": 0
    }

    print(f"[MINER] Miner ID: {miner_id}")
    print(f"[MINER] Miner Address: {addr}")
    print(f"[MINER] Connecting to Node at {NODE_URL}...")

    while True:
        try:
            async with websockets.connect(NODE_URL, ping_interval=15, ping_timeout=30) as websocket:
                print("[MINER] Connected to CTM Node successfully! Registering...")
                
                timestamp = int(time.time())
                reg_payload = f"{miner_id}{addr}{timestamp}".encode("utf-8")
                reg_signature = sign_bytes(privkey, reg_payload)

                await websocket.send(json.dumps({
                    "type": "register",
                    "miner_id": miner_id,
                    "wallet": addr,
                    "address": addr,
                    "pub_key": pub_key_hex,
                    "public_key": pub_key_hex,
                    "signature": reg_signature,
                    "timestamp": timestamp,
                    "role": "miner"
                }))

                async for raw_message in websocket:
                    if DEBUG_MODE:
                        print(f"[DEBUG] Raw incoming packet: {raw_message}")

                    msg = json.loads(raw_message)
                    mtype = msg.get("type")

                    if mtype == "memory_challenge_request":
                        print("[MINER] Solving memory-hard challenge...")
                        seed_bytes = base64.b64decode(msg["seed"])
                        mem_size = msg.get("memory_size", 8 * 1024 * 1024)
                        
                        h_obj = hashlib.sha256()
                        counter = 0
                        while h_obj.digest_size * counter < mem_size:
                            h_obj.update(hashlib.sha256(seed_bytes + counter.to_bytes(8, "big")).digest())
                            counter += 1

                        await websocket.send(json.dumps({
                            "type": "memory_challenge_response",
                            "miner_id": miner_id,
                            "seed": msg["seed"],
                            "commitment": base64.b64encode(h_obj.digest()).decode(),
                        }))
                        print("[MINER] Memory challenge response sent.")

                    elif mtype == "registration_failed":
                        print(f"[MINER] Registration failed: {msg}")
                        break

                    elif mtype == "maze_init":
                        session["maze_id"] = msg.get("maze_root", pub_key_hex)
                        session["size"] = msg.get("size", DEFAULT_MAZE_SIZE)
                        session["pos"] = tuple(msg.get("start", (0, 0, 0)))
                        session["goal"] = tuple(msg.get("goal", (4, 4, 4)))
                        session["ghost"] = tuple(msg.get("ghost", (2, 2, 2)) or (2, 2, 2))
                        raw_maze = msg.get("maze")
                        session["walls"] = maze_grid_to_walls(raw_maze) if raw_maze else []
                        session["movenonce"] = int(msg.get("movenonce", 0))
                        session["mazeversion"] = int(msg.get("growthcount", 0))

                        print(f"[AI] Initialized maze {session['maze_id'][:12]}... at pos {session['pos']} towards goal {session['goal']}...")
                        await make_and_send_move(websocket, privkey, miner_id, session)

                    elif mtype == "maze_state":
                        st = msg.get("state")
                        if not st or not isinstance(st, dict):
                            print("[MINER] Received null maze state. Waiting before state resync...")
                            await asyncio.sleep(2.0)
                            await websocket.send(json.dumps({
                                "type": "get_maze_state",
                                "miner_id": miner_id,
                                "maze_id": session["maze_id"]
                            }))
                            continue

                        session["size"] = int(st.get("size", session["size"]))
                        session["pos"] = tuple(st.get("position", session["pos"]))
                        session["goal"] = tuple(st.get("goal", session["goal"]))
                        session["ghost"] = tuple(st.get("ghost", session["ghost"]) or (2, 2, 2))
                        raw_maze = st.get("maze")
                        if raw_maze:
                            session["walls"] = maze_grid_to_walls(raw_maze)
                        session["movenonce"] = int(st.get("movenonce", session["movenonce"]))
                        session["mazeversion"] = int(st.get("growthcount", session["mazeversion"]))
                        session["maze_id"] = st.get("maze_root", session["maze_id"])

                        print(f"[AI] Synced state at pos {session['pos']} towards goal {session['goal']}...")
                        await make_and_send_move(websocket, privkey, miner_id, session)

                    elif mtype == "maze_move_ack":
                        if msg.get("success"):
                            st = msg.get("state", {})
                            if st.get("movenonce") is not None:
                                session["movenonce"] = int(st["movenonce"])
                            if st.get("growthcount") is not None:
                                session["mazeversion"] = int(st["growthcount"])
                            if st.get("position"):
                                session["pos"] = tuple(st["position"])
                            if st.get("goal"):
                                session["goal"] = tuple(st["goal"])
                            print(f"[MINER] ⚡ Move accepted! Pos: {session['pos']} | Balance: {msg.get('balance', 0)} MCX")
                            
                            await make_and_send_move(websocket, privkey, miner_id, session)
                        else:
                            msg_text = msg.get("message") or ""
                            print(f"[MINER] Move rejected: {msg_text}")
                            if msg_text == "No maze state":
                                print("[MINER] Requesting maze state resync with backoff...")
                                await asyncio.sleep(1.5)
                                await websocket.send(json.dumps({
                                    "type": "get_maze_state",
                                    "miner_id": miner_id,
                                    "maze_id": session["maze_id"]
                                }))
                            elif msg_text.startswith("Nonce mismatch"):
                                try:
                                    expected = int(msg_text.split("expected")[1].split(",")[0].strip())
                                    session["movenonce"] = expected - 1
                                    print(f"[MINER] Nonce resynced -> {session['movenonce']}")
                                    await make_and_send_move(websocket, privkey, miner_id, session)
                                except Exception:
                                    pass
                            elif "wall_hit" in msg:
                                print("[MINER] Wall hit detected! Solving spatial PoW...")
                                wseed = msg.get("wall_seed", "0" * 64)
                                wdiff = msg.get("wall_diff", 4)
                                p_nonce = solve_wall_pow(wseed, wdiff)
                                await websocket.send(json.dumps({
                                    "type": "wall_pow_solution",
                                    "nonce": p_nonce,
                                    "address": addr
                                }))

        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK) as ce:
            print(f"[WARNING] Connection closed ({ce}). Reconnecting in 3 seconds...")
            await asyncio.sleep(3)
        except Exception as e:
            print(f"[ERROR] Exception encountered: {e}. Reconnecting in 5 seconds...")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(start_desktop_miner())
    except KeyboardInterrupt:
        print("\n[SHUTDOWN] Miner stopped by user.")
