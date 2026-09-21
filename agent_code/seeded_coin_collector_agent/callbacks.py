"""Supplied coin-collector policy with an explicit evaluation seed."""

from __future__ import annotations

import os
import random
import re

import numpy as np
import settings as s


def _agent_index(name: str) -> int:
    match = re.search(r"_(\d+)(?:_code)?$", name)
    return int(match.group(1)) if match else 0


def look_for_targets(free_space, start, targets, rng, logger=None):
    """Return the supplied BFS policy's next target step using ``rng``."""
    if len(targets) == 0:
        return None
    frontier = [start]
    parent_dict = {start: start}
    dist_so_far = {start: 0}
    best = start
    best_dist = np.sum(np.abs(np.subtract(targets, start)), axis=1).min()
    while frontier:
        current = frontier.pop(0)
        distance = np.sum(np.abs(np.subtract(targets, current)), axis=1).min()
        if distance + dist_so_far[current] <= best_dist:
            best = current
            best_dist = distance + dist_so_far[current]
        if distance == 0:
            best = current
            break
        x, y = current
        neighbors = [
            position for position in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
            if free_space[position]
        ]
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in parent_dict:
                frontier.append(neighbor)
                parent_dict[neighbor] = current
                dist_so_far[neighbor] = dist_so_far[current] + 1
    if logger:
        logger.debug("Suitable target found at %s", best)
    current = best
    while True:
        if parent_dict[current] == start:
            return current
        current = parent_dict[current]


def setup(self):
    seed_text = os.environ.get("TASK3_OPPONENT_SEED")
    if seed_text is None:
        raise ValueError("seeded_coin_collector_agent requires TASK3_OPPONENT_SEED")
    callback_name = getattr(self, "name", self.logger.name)
    sequence = np.random.SeedSequence(
        [int(seed_text), _agent_index(callback_name), 0xC013]
    ).generate_state(2)
    self.rng = random.Random((int(sequence[0]) << 32) | int(sequence[1]))


def act(self, game_state):
    arena = game_state["field"]
    _, _, bombs_left, (x, y) = game_state["self"]
    bombs = game_state["bombs"]
    bomb_xys = [xy for xy, _ in bombs]
    others = [other[3] for other in game_state["others"]]
    coins = game_state["coins"]
    bomb_map = np.ones(arena.shape) * 5
    for (bomb_x, bomb_y), timer in bombs:
        blast_line = (
            [(bomb_x + offset, bomb_y) for offset in range(-3, 4)]
            + [(bomb_x, bomb_y + offset) for offset in range(-3, 4)]
        )
        for i, j in blast_line:
            if 0 < i < bomb_map.shape[0] and 0 < j < bomb_map.shape[1]:
                bomb_map[i, j] = min(bomb_map[i, j], timer)

    directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
    valid_tiles = [
        position for position in directions
        if arena[position] == 0
        and game_state["explosion_map"][position] < 1
        and bomb_map[position] > 0
        and position not in others
        and position not in bomb_xys
    ]
    valid_actions = []
    if (x - 1, y) in valid_tiles:
        valid_actions.append("LEFT")
    if (x + 1, y) in valid_tiles:
        valid_actions.append("RIGHT")
    if (x, y - 1) in valid_tiles:
        valid_actions.append("UP")
    if (x, y + 1) in valid_tiles:
        valid_actions.append("DOWN")
    if (x, y) in valid_tiles:
        valid_actions.append("WAIT")
    if bombs_left > 0:
        valid_actions.append("BOMB")

    action_ideas = ["UP", "DOWN", "LEFT", "RIGHT"]
    self.rng.shuffle(action_ideas)
    cols = range(1, arena.shape[0] - 1)
    rows = range(1, arena.shape[1] - 1)
    dead_ends = [
        (column, row) for column in cols for row in rows
        if arena[column, row] == 0
        and [arena[column + 1, row], arena[column - 1, row],
             arena[column, row + 1], arena[column, row - 1]].count(0) == 1
    ]
    crates = [
        (column, row) for column in cols for row in rows if arena[column, row] == 1
    ]
    targets = [target for target in coins + dead_ends + crates if target not in bomb_xys]
    free_space = arena == 0
    for opponent in others:
        free_space[opponent] = False
    destination = look_for_targets(free_space, (x, y), targets, self.rng, self.logger)
    if destination == (x, y - 1):
        action_ideas.append("UP")
    if destination == (x, y + 1):
        action_ideas.append("DOWN")
    if destination == (x - 1, y):
        action_ideas.append("LEFT")
    if destination == (x + 1, y):
        action_ideas.append("RIGHT")
    if destination is None:
        action_ideas.append("WAIT")
    if (x, y) in dead_ends:
        action_ideas.append("BOMB")
    adjacent = [arena[x + 1, y], arena[x - 1, y], arena[x, y + 1], arena[x, y - 1]]
    if destination == (x, y) and adjacent.count(1) > 0:
        action_ideas.append("BOMB")

    for (bomb_x, bomb_y), _ in bombs:
        if bomb_x == x and abs(bomb_y - y) <= s.BOMB_POWER:
            if bomb_y > y:
                action_ideas.append("UP")
            if bomb_y < y:
                action_ideas.append("DOWN")
            action_ideas.extend(["LEFT", "RIGHT"])
        if bomb_y == y and abs(bomb_x - x) <= s.BOMB_POWER:
            if bomb_x > x:
                action_ideas.append("LEFT")
            if bomb_x < x:
                action_ideas.append("RIGHT")
            action_ideas.extend(["UP", "DOWN"])
    for bomb_position, _ in bombs:
        if bomb_position == (x, y):
            action_ideas.extend(action_ideas[:4])

    while action_ideas:
        action = action_ideas.pop()
        if action in valid_actions:
            return action
