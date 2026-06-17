"""Heuristic Crawl agent with deterministic movement and A* pathfinding.

The bot keeps the logic intentionally local and lightweight:
- factories build workers/scouts/miners based on simple thresholds
- workers clear walls and keep lanes open
- miners transform on visible mining nodes
- scouts explore toward unseen territory

The movement logic is mostly deterministic now, with A* used to route toward
targets instead of relying on random drift.
"""

from heapq import heappop, heappush
from collections import deque


NORTH = "NORTH"
EAST = "EAST"
SOUTH = "SOUTH"
WEST = "WEST"
IDLE = "IDLE"

DIRS = (NORTH, EAST, SOUTH, WEST)
DIR_BIT = {
    NORTH: 1,
    EAST: 2,
    SOUTH: 4,
    WEST: 8,
}

TYPE_FACTORY = 0
TYPE_SCOUT = 1
TYPE_WORKER = 2
TYPE_MINER = 3


MEMORY = {
    "seen_cells": set(),
    "seen_crystals": set(),
    "seen_nodes": set(),
    "turn": 0,
    "last_south_bound": None,
}


def wall_at(obs, config, col, row):
    """Return the wall bitfield for a board cell, or 0 if unknown/out of range."""
    idx = (row - obs.southBound) * config.width + col
    if idx < 0 or idx >= len(obs.walls):
        return 0
    wall = obs.walls[idx]
    return 0 if wall == -1 else wall


def has_wall(wall, direction):
    return bool(wall & DIR_BIT[direction])


def my_robots(obs):
    return {
        uid: data
        for uid, data in obs.robots.items()
        if data[4] == obs.player
    }


def adjacent(col, row, direction):
    if direction == NORTH:
        return col, row + 1
    if direction == EAST:
        return col + 1, row
    if direction == SOUTH:
        return col, row - 1
    return col - 1, row


def inside_board(obs, config, col, row):
    return 0 <= col < config.width and obs.southBound <= row <= obs.northBound


def can_move(obs, config, col, row, direction):
    wall = wall_at(obs, config, col, row)
    if has_wall(wall, direction):
        return False
    nc, nr = adjacent(col, row, direction)
    return inside_board(obs, config, nc, nr)


def valid_moves(obs, config, col, row):
    moves = []
    for direction in DIRS:
        if can_move(obs, config, col, row, direction):
            moves.append(direction)
    return moves


def occupied_cells(obs):
    occupied = {}
    for data in obs.robots.values():
        occupied[(data[1], data[2])] = data
    return occupied


def robot_on_cell(obs, col, row):
    return occupied_cells(obs).get((col, row))


def is_danger_cell(obs, col, row):
    robot = robot_on_cell(obs, col, row)
    return robot is not None


def move_risk(obs, config, col, row, direction):
    """Lower is better. Penalize edge cases and moves that keep us low."""
    if not can_move(obs, config, col, row, direction):
        return 10**9

    nc, nr = adjacent(col, row, direction)
    risk = 0

    if direction == SOUTH:
        risk += 12
    if direction == NORTH:
        risk -= 2

    # Staying near the south boundary is dangerous because of scrolling.
    if nr - obs.southBound <= 2:
        risk += 8
    elif nr - obs.southBound <= 4:
        risk += 3

    # Prefer climbing upward over lingering.
    risk += max(0, 6 - (nr - obs.southBound))

    # Small penalty for hugging board edges.
    if nc == 0 or nc == config.width - 1:
        risk += 2

    return risk


def open_side_move(obs, config, col, row):
    """Use a cheap open side lane when forward is clear or a side path is available."""
    moves = valid_moves(obs, config, col, row)
    safe = []
    for direction in moves:
        nc, nr = adjacent(col, row, direction)
        if not is_danger_cell(obs, nc, nr):
            safe.append(direction)

    if NORTH in safe:
        return NORTH

    side_moves = [d for d in (EAST, WEST) if d in safe]
    if side_moves:
        return min(side_moves, key=lambda d: move_risk(obs, config, col, row, d))

    return None


def cell_key(col, row):
    return f"{col},{row}"


def parse_cell(key):
    c, r = key.split(",")
    return int(c), int(r)


def init_memory():
    if "seen_cells" not in MEMORY:
        MEMORY["seen_cells"] = set()
    if "seen_crystals" not in MEMORY:
        MEMORY["seen_crystals"] = set()
    if "seen_nodes" not in MEMORY:
        MEMORY["seen_nodes"] = set()
    if "turn" not in MEMORY:
        MEMORY["turn"] = 0
    if "last_south_bound" not in MEMORY:
        MEMORY["last_south_bound"] = None


def reset_memory():
    MEMORY["seen_cells"] = set()
    MEMORY["seen_crystals"] = set()
    MEMORY["seen_nodes"] = set()
    MEMORY["turn"] = 0
    MEMORY["last_south_bound"] = None


def vision_range(rtype, config):
    if rtype == TYPE_FACTORY:
        return config.visionFactory
    if rtype == TYPE_SCOUT:
        return config.visionScout
    if rtype == TYPE_WORKER:
        return config.visionWorker
    if rtype == TYPE_MINER:
        return config.visionMiner
    return 3


def is_walkable(obs, config, col, row):
    return inside_board(obs, config, col, row)


def update_memory(obs, config):
    init_memory()
    if MEMORY["last_south_bound"] is not None and obs.southBound < MEMORY["last_south_bound"]:
        reset_memory()
    MEMORY["turn"] += 1
    MEMORY["last_south_bound"] = obs.southBound

    for data in obs.robots.values():
        if data[4] != obs.player:
            continue
        rtype, col, row = data[0], data[1], data[2]
        vision = vision_range(rtype, config)
        for dr in range(-vision, vision + 1):
            remaining = vision - abs(dr)
            for dc in range(-remaining, remaining + 1):
                vc, vr = col + dc, row + dr
                if inside_board(obs, config, vc, vr):
                    MEMORY["seen_cells"].add(cell_key(vc, vr))

    for key in obs.crystals.keys():
        MEMORY["seen_crystals"].add(key)

    for key in obs.miningNodes.keys():
        MEMORY["seen_nodes"].add(key)


def frontier_candidates(obs, config):
    """Return seen cells that border at least one unseen neighbor."""
    frontier = []
    for key in MEMORY["seen_cells"]:
        col, row = parse_cell(key)
        if not inside_board(obs, config, col, row):
            continue
        if row < obs.southBound + 2:
            continue
        for direction in DIRS:
            nc, nr = adjacent(col, row, direction)
            if not inside_board(obs, config, nc, nr):
                continue
            if cell_key(nc, nr) not in MEMORY["seen_cells"]:
                frontier.append((col, row))
                break
    return frontier


def nearest_memory_target(obs, config, col, row, keys):
    best = None
    best_dist = 10**9
    for key in keys:
        tc, tr = parse_cell(key)
        if not inside_board(obs, config, tc, tr):
            continue
        dist = abs(tc - col) + abs(tr - row)
        if dist < best_dist:
            best = (tc, tr)
            best_dist = dist
    return best


def lane_target(obs, config, uid, col, row, spread=4):
    """Pick a deterministic side lane so units do not all march in one column."""
    digits = "".join(ch for ch in str(uid) if ch.isdigit())
    seed = int(digits[-2:]) if digits else sum(ord(ch) for ch in str(uid))
    offset = ((seed % (spread * 2 + 1)) - spread)
    if offset == 0:
        offset = 2 if seed % 2 == 0 else -2

    target_col = max(1, min(config.width - 2, col + offset))
    target_row = min(obs.northBound - 1, row + 5)
    return target_col, target_row


def astar_next_direction(obs, config, start, goal, avoid_south=False):
    """Return the first step on an A* path from start to goal."""
    if start == goal:
        return None

    def heuristic(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_heap = []
    heappush(open_heap, (0, 0, start, None))
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_heap:
        _, _, current, first_dir = heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            return first_dir

        col, row = current
        for direction in DIRS:
            if avoid_south and direction == SOUTH and current != start:
                continue
            if not can_move(obs, config, col, row, direction):
                continue

            nc, nr = adjacent(col, row, direction)
            if not is_walkable(obs, config, nc, nr):
                continue

            step_cost = 1
            if direction == SOUTH:
                step_cost += 8
            if nr - obs.southBound <= 2:
                step_cost += 4
            elif nr - obs.southBound <= 4:
                step_cost += 1

            tentative = g_score[current] + step_cost
            neighbor = (nc, nr)
            if tentative < g_score.get(neighbor, 10**9):
                g_score[neighbor] = tentative
                came_from[neighbor] = current
                next_first = direction if current == start else came_from.get(current, None)
                priority = tentative + heuristic(neighbor, goal)
                heappush(open_heap, (priority, tentative, neighbor, direction if current == start else first_dir))

    return None


def bfs_next_direction(obs, config, start, goal, avoid_south=False, avoid_occupied=True):
    """Return the first step on a BFS path from start to goal."""
    if start == goal:
        return None

    queue = deque([start])
    came_from = {start: None}
    first_step = {start: None}

    while queue:
        col, row = queue.popleft()
        if (col, row) == goal:
            return first_step[(col, row)]

        for direction in DIRS:
            if avoid_south and direction == SOUTH and (col, row) != start:
                continue
            if not can_move(obs, config, col, row, direction):
                continue

            nc, nr = adjacent(col, row, direction)
            nxt = (nc, nr)
            if nxt in came_from:
                continue
            if avoid_occupied and nxt != goal and is_danger_cell(obs, nc, nr):
                continue

            came_from[nxt] = (col, row)
            first_step[nxt] = direction if (col, row) == start else first_step[(col, row)]
            queue.append(nxt)

    return None


def is_southern_risk(obs, row):
    """Rows close to the southern boundary are risky as the map scrolls."""
    return row - obs.southBound <= 2


def is_northern_pressure(obs, row):
    return obs.northBound - row <= 2


def nearest_known_target(obs, col, row, predicate):
    """Find a visible target cell that matches predicate."""
    best = None
    best_dist = 10**9

    for key in obs.crystals.keys():
        c, r = parse_cell(key)
        if predicate("crystal", c, r):
            dist = abs(c - col) + abs(r - row)
            if dist < best_dist:
                best = (c, r)
                best_dist = dist

    for key in obs.miningNodes.keys():
        c, r = parse_cell(key)
        if predicate("node", c, r):
            dist = abs(c - col) + abs(r - row)
            if dist < best_dist:
                best = (c, r)
                best_dist = dist

    return best


def unexplored_target(obs, config, col, row, uid=None):
    frontier = frontier_candidates(obs, config)
    if frontier:
        best = None
        best_score = 10**9
        lane = lane_target(obs, config, uid, col, row) if uid is not None else (col, row + 5)
        for tc, tr in frontier:
            dist = abs(tc - col) + abs(tr - row)
            lane_dist = abs(tc - lane[0])
            north_bonus = max(0, tr - row)
            score = dist + lane_dist * 2 - north_bonus
            if score < best_score:
                best = (tc, tr)
                best_score = score
        return best

    return lane_target(obs, config, uid, col, row) if uid is not None else (col, min(obs.northBound, row + 4))


def pick_move_toward(obs, config, col, row, preferred_dirs, avoid_south=False):
    moves = valid_moves(obs, config, col, row)
    safe_moves = []
    for direction in moves:
        nc, nr = adjacent(col, row, direction)
        if not is_danger_cell(obs, nc, nr):
            safe_moves.append(direction)
    if safe_moves:
        moves = safe_moves

    if not moves:
        return IDLE

    ordered = []
    for d in preferred_dirs:
        if d in moves:
            ordered.append(d)

    if avoid_south and SOUTH in ordered and len(ordered) > 1:
        ordered = [d for d in ordered if d != SOUTH]

    if ordered:
        return min(ordered, key=lambda d: move_risk(obs, config, col, row, d))

    if avoid_south and SOUTH in moves and len(moves) > 1:
        moves = [d for d in moves if d != SOUTH]

    return min(moves, key=lambda d: move_risk(obs, config, col, row, d))


def act_factory(obs, config, uid, data, state):
    _, col, row, energy = data[0], data[1], data[2], data[3]
    jump_cd = data[6] if len(data) > 6 else 0
    build_cd = data[7] if len(data) > 7 else 0
    wall = wall_at(obs, config, col, row)

    south_risk = is_southern_risk(obs, row)
    enemy_near = state["enemy_pressure"] > 0
    scroll_buffer = row - obs.southBound

    if not has_wall(wall, NORTH) and can_move(obs, config, col, row, NORTH):
        if scroll_buffer <= 8:
            return NORTH
        if build_cd == 0 and MEMORY["turn"] < 80 and scroll_buffer >= 12:
            spawn_robot = robot_on_cell(obs, col, row + 1)
            spawn_clear = spawn_robot is None
            if spawn_clear and energy >= config.scoutCost and state["scout_count"] == 0:
                return "BUILD_SCOUT"
            if spawn_clear and energy >= config.workerCost and state["worker_count"] == 0 and scroll_buffer >= 10:
                return "BUILD_WORKER"
            if spawn_clear and energy >= config.minerCost and state["mining_node_seen"] and state["miner_count"] == 0:
                return "BUILD_MINER"
            if spawn_clear and energy >= config.scoutCost and state["scout_count"] < 2 and not state["map_is_open"]:
                return "BUILD_SCOUT"
            if spawn_clear and energy >= config.workerCost and (enemy_near or south_risk or state["factory_blocked"]) and state["worker_count"] < 1:
                return "BUILD_WORKER"
        return NORTH

    # A north wall is deadly for the factory. Jump if available; otherwise sidestep.
    if has_wall(wall, NORTH):
        if jump_cd == 0:
            nc, nr = col, row + 2
            if inside_board(obs, config, nc, nr):
                return "JUMP_NORTH"
        side = open_side_move(obs, config, col, row)
        if side:
            return side
        return IDLE

    side = open_side_move(obs, config, col, row)
    if side:
        return side

    return IDLE


def act_worker(obs, config, uid, data, state):
    _, col, row, energy = data[0], data[1], data[2], data[3]
    wall = wall_at(obs, config, col, row)

    direct = open_side_move(obs, config, col, row)
    if direct:
        return direct

    if (state["factory_blocked"] or (state["factory_north_blocked"] and not state["factory_jump_ready"])) and has_wall(wall, NORTH) and energy >= config.wallRemoveCost:
        return "REMOVE_NORTH"

    target = nearest_known_target(
        obs,
        col,
        row,
        lambda kind, c, r: kind == "node" or kind == "crystal",
    )
    if target:
        step = bfs_next_direction(obs, config, (col, row), target, avoid_south=is_southern_risk(obs, row))
        if step:
            return step

    explore = unexplored_target(obs, config, col, row, uid)
    if explore:
        step = bfs_next_direction(obs, config, (col, row), explore, avoid_south=True)
        if step:
            return step

    if is_southern_risk(obs, row):
        return pick_move_toward(obs, config, col, row, [NORTH, EAST, WEST], avoid_south=True)

    return pick_move_toward(obs, config, col, row, [NORTH, EAST, WEST, SOUTH], avoid_south=True)


def act_miner(obs, config, uid, data, state):
    _, col, row, energy = data[0], data[1], data[2], data[3]

    if f"{col},{row}" in obs.miningNodes:
        return "TRANSFORM" if energy >= config.transformCost else IDLE

    target = nearest_known_target(
        obs,
        col,
        row,
        lambda kind, c, r: kind == "node",
    )
    if target is None:
        target = nearest_memory_target(obs, config, col, row, MEMORY["seen_nodes"])
    if target:
        step = bfs_next_direction(obs, config, (col, row), target, avoid_south=is_southern_risk(obs, row))
        if step:
            return step

    direct = open_side_move(obs, config, col, row)
    if direct:
        return direct

    if is_southern_risk(obs, row):
        return pick_move_toward(obs, config, col, row, [NORTH, EAST, WEST], avoid_south=True)

    return pick_move_toward(obs, config, col, row, [NORTH, EAST, WEST, SOUTH], avoid_south=True)


def act_scout(obs, config, uid, data, state):
    _, col, row, energy = data[0], data[1], data[2], data[3]
    wall = wall_at(obs, config, col, row)

    preferred = []
    if not has_wall(wall, NORTH):
        preferred.append(NORTH)
    if not has_wall(wall, EAST):
        preferred.append(EAST)
    if not has_wall(wall, WEST):
        preferred.append(WEST)
    if not has_wall(wall, SOUTH):
        preferred.append(SOUTH)

    if is_southern_risk(obs, row):
        preferred = [d for d in preferred if d != SOUTH] or preferred

    if obs.crystals:
        target = nearest_known_target(
            obs,
            col,
            row,
            lambda kind, c, r: kind == "crystal",
        )
        if target:
            step = bfs_next_direction(obs, config, (col, row), target, avoid_south=is_southern_risk(obs, row))
            if step:
                return step

    explore = unexplored_target(obs, config, col, row, uid)
    if explore:
        step = bfs_next_direction(obs, config, (col, row), explore, avoid_south=True)
        if step:
            return step

    return pick_move_toward(obs, config, col, row, preferred, avoid_south=True)


def build_state(obs, config):
    state = {
        "scout_count": 0,
        "worker_count": 0,
        "miner_count": 0,
        "factory_count": 0,
        "enemy_pressure": 0,
        "mining_node_seen": bool(obs.miningNodes) or bool(MEMORY["seen_nodes"]),
        "factory_blocked": False,
        "factory_north_blocked": False,
        "factory_jump_ready": False,
        "map_is_open": False,
    }

    for data in obs.robots.values():
        if data[4] == obs.player:
            rtype = data[0]
            if rtype == TYPE_FACTORY:
                state["factory_count"] += 1
                col, row = data[1], data[2]
                jump_cd = data[6] if len(data) > 6 else 0
                moves = valid_moves(obs, config, col, row)
                safe_moves = []
                for direction in moves:
                    nc, nr = adjacent(col, row, direction)
                    if not is_danger_cell(obs, nc, nr):
                        safe_moves.append(direction)
                state["factory_blocked"] = not safe_moves
                state["factory_north_blocked"] = has_wall(wall_at(obs, config, col, row), NORTH)
                state["factory_jump_ready"] = jump_cd == 0
                state["map_is_open"] = NORTH in safe_moves or EAST in safe_moves or WEST in safe_moves
            elif rtype == TYPE_SCOUT:
                state["scout_count"] += 1
            elif rtype == TYPE_WORKER:
                state["worker_count"] += 1
            elif rtype == TYPE_MINER:
                state["miner_count"] += 1
        else:
            # Simple pressure proxy: any visible enemy near the north half
            if data[2] >= obs.southBound + 5:
                state["enemy_pressure"] += 1

    return state


def agent(obs, config):
    update_memory(obs, config)
    actions = {}
    robots = my_robots(obs)
    state = build_state(obs, config)

    # Handle factories first so build actions are decided before movement.
    for uid, data in robots.items():
        if data[0] == TYPE_FACTORY:
            actions[uid] = act_factory(obs, config, uid, data, state)

    for uid, data in robots.items():
        if uid in actions:
            continue

        rtype = data[0]
        if rtype == TYPE_WORKER:
            actions[uid] = act_worker(obs, config, uid, data, state)
        elif rtype == TYPE_MINER:
            actions[uid] = act_miner(obs, config, uid, data, state)
        elif rtype == TYPE_SCOUT:
            actions[uid] = act_scout(obs, config, uid, data, state)
        else:
            # Fallback for anything unusual.
            col, row = data[1], data[2]
            actions[uid] = pick_move_toward(obs, config, col, row, [NORTH, EAST, WEST], avoid_south=is_southern_risk(obs, row))

    return actions
