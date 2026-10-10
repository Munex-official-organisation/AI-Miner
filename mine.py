#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import time
import json
import hashlib
import random
import asyncio
import websockets
import numpy as np
import keras

P_CURVE = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
N_ORDER = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
G_X = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
G_Y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

CHAIN_ID = 6620
ACTION_MAP = {
    0: "east",
    1: "west",
    2: "north",
    3: "south",
    4: "up",
    5: "down"
}

def mod_inv(a, n=P_CURVE):
    return pow(a, n - 2, n)

def point_add(p1, p2):
    if p1 is None: return p2
    if p2 is None: return p1
    x1, y1 = p1; x2, y2 = p2
    if x1 == x2:
        if (y1 + y2) % P_CURVE == 0: return None
        m = (3 * x1 * x1) * mod_inv(2 * y1, P_CURVE) % P_CURVE
    else:
        m = (y2 - y1) * mod_inv(x2 - x1, P_CURVE) % P_CURVE
    x3 = (m * m - x1 - x2) % P_CURVE
    y3 = (m * (x1 - x3) - y1) % P_CURVE
    return (x3, y3)

def point_mul(pt, k):
    res = None; curr = pt
    while k > 0:
        if k & 1: res = point_add(res, curr)
        curr = point_add(curr, curr)
        k >>= 1
    return res

def sign_bytes_secp256k1(priv_hex: str, message: bytes) -> str:
    d = int(priv_hex, 16)
    z = int.from_bytes(hashlib.sha256(message).digest(), "big")
    k_seed = hashlib.sha256(d.to_bytes(32, "big") + message).digest()
    k = int.from_bytes(k_seed, "big") % N_ORDER
    if k == 0: k = 1

    while True:
        pt = point_mul((G_X, G_Y), k)
        r = pt[0] % N_ORDER
        if r == 0: k = (k + 1) % N_ORDER; continue
        k_inv = mod_inv(k, N_ORDER)
        s = (k_inv * (z + r * d)) % N_ORDER
        if s == 0: k = (k + 1) % N_ORDER; continue
        if s > N_ORDER // 2: s = N_ORDER - s
        return f"{r:064x}{s:064x}"

_KECCAK_RC = [
    0x0000000000000001, 0x0000000000008082, 0x800000000000808A, 0x8000000080008000,
    0x000000000000808B, 0x0000000080000001, 0x8000000080008081, 0x8000000000008009,
    0x000000000000008A, 0x0000000000000088, 0x0000000080008009, 0x000000008000000A,
    0x000000008000808B, 0x800000000000008B, 0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080, 0x000000000000800A, 0x800000008000000A,
    0x8000000080008081, 0x8000000000008080, 0x0000000080000001, 0x8000000080008008
]
_KECCAK_ROTC = [
    [0, 36, 3, 41, 18], [1, 44, 10, 45, 2], [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56], [27, 20, 39, 8, 14]
]

def _rol64(x, n):
    return ((x << (n % 64)) & 0xFFFFFFFFFFFFFFFF) | (x >> ((64 - (n % 64)) % 64))

def _keccak_f1600(state):
    A = [row[:] for row in state]
    for round_idx in range(24):
        C = [A[x][0] ^ A[x][1] ^ A[x][2] ^ A[x][3] ^ A[x][4] for x in range(5)]
        D = [C[(x - 1) % 5] ^ _rol64(C[(x + 1) % 5], 1) for x in range(5)]
        A = [[A[x][y] ^ D[x] for y in range(5)] for x in range(5)]
        B = [[0] * 5 for _ in range(5)]
        for x in range(5):
            for y in range(5):
                B[y][(2 * x + 3 * y) % 5] = _rol64(A[x][y], _KECCAK_ROTC[x][y])
        for x in range(5):
            for y in range(5):
                A[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])
        A[0][0] ^= _KECCAK_RC[round_idx]
    return A

def keccak_256(data: bytes) -> bytes:
    state = [[0] * 5 for _ in range(5)]
    rate = 136
    padded = bytearray(data)
    padded.append(0x01)
    while len(padded) % rate != (rate - 1): padded.append(0x00)
    padded.append(0x80)
    for block_start in range(0, len(padded), rate):
        for i in range(17):
            val = int.from_bytes(padded[block_start + i * 8: block_start + (i + 1) * 8], "little")
            state[i % 5][i // 5] ^= val
        state = _keccak_f1600(state)
    out = bytearray()
    for i in range(4):
        out.extend(state[i % 5][i // 5].to_bytes(8, "little"))
    return bytes(out)

def privkey_to_address(priv_hex: str) -> str:
    pt = point_mul((G_X, G_Y), int(priv_hex, 16))
    pub_raw = pt[0].to_bytes(32, "big") + pt[1].to_bytes(32, "big")
    return "0x" + keccak_256(pub_raw)[12:].hex()

def sign_pog_move(priv_hex: str, miner_addr: str, direction: str, movenonce: int) -> str:
    payload = f"MCXPOGMOVEV1:{CHAIN_ID}:{miner_addr.lower()}:{direction.lower()}:{movenonce}".encode("utf-8")
    return sign_bytes_secp256k1(priv_hex, payload)

MODEL_FILENAME = "model.keras"

def load_miner_brain():
    if os.path.exists(MODEL_FILENAME):
        model = keras.saving.load_model(MODEL_FILENAME)
        return model
    else:
        sys.exit(1)

model = load_miner_brain()

def construct_state_vector(pos, goal, ghost, walls, size) -> np.ndarray:
    mx, my, mz = pos
    gx, gy, gz = goal
    hx, hy, hz = ghost
    s = float(size) if size > 0 else 5.0

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

async def start_desktop_miner(node_url="ws://127.0.0.1:8080/ws", privkey=None):
    if not privkey:
        privkey = hashlib.sha256(b"munex_desktop_ai_miner_01").hexdigest()
    addr = privkey_to_address(privkey)
    
    while True:
        try:
            async with websockets.connect(node_url) as websocket:
                await websocket.send(json.dumps({"type": "register", "address": addr, "role": "miner"}))

                async for raw_message in websocket:
                    msg = json.loads(raw_message)
                    mtype = msg.get("type")

                    if mtype == "maze_state":
                        size = msg.get("dimension", 5)
                        pos = tuple(msg.get("miner_pos", (0, 0, 0)))
                        goal = tuple(msg.get("goal_pos", (4, 4, 4)))
                        ghost = tuple(msg.get("ghost_pos", (2, 2, 2)))
                        walls = msg.get("walls", [])
                        movenonce = msg.get("movenonce", 0)

                        state_vec = construct_state_vector(pos, goal, ghost, walls, size)
                        state_tensor = np.array([state_vec], dtype=np.float32)

                        q_values = model.predict(state_tensor, verbose=0)[0]
                        best_action_idx = int(np.argmax(q_values))
                        step_dir = ACTION_MAP.get(best_action_idx, "east")

                        sig = sign_pog_move(privkey, addr, step_dir, movenonce)
                        await websocket.send(json.dumps({
                            "type": "maze_move",
                            "direction": step_dir,
                            "movenonce": movenonce,
                            "signature": sig,
                            "address": addr
                        }))

                    elif mtype == "maze_move_ack":
                        if msg.get("success"):
                            pass
                        else:
                            if "wall_hit" in msg:
                                wseed = msg.get("wall_seed", "0" * 64)
                                wdiff = msg.get("wall_diff", 4)
                                p_nonce = solve_wall_pow(wseed, wdiff)
                                await websocket.send(json.dumps({
                                    "type": "wall_pow_solution",
                                    "nonce": p_nonce,
                                    "address": addr
                                }))

        except (websockets.ConnectionClosedError, websockets.ConnectionClosedOK):
            await asyncio.sleep(3)
        except Exception:
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        asyncio.run(start_desktop_miner())
    except KeyboardInterrupt:
        pass
