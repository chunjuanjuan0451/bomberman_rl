"""Auditable official rewards and bounded curriculum shaping."""

from __future__ import annotations

import numpy as np

import events as e
from agent_code.model_a_dqn.features import INF_TIME, danger_time_map, legal_action_mask
from .features import idle_wait, resource_distance


OFFICIAL = {
    e.COIN_COLLECTED: 1.0,
    e.KILLED_OPPONENT: 5.0,
    e.SURVIVED_ROUND: 1.0,
    e.GOT_KILLED: -5.0,
    e.KILLED_SELF: -8.0,
    e.CRATE_DESTROYED: 0.2,
    e.INVALID_ACTION: -0.2,
    e.BOMB_DROPPED: 0.0,
}
SHAPING_NAMES = (
    "IDLE_WAIT", "ENTER_IMMEDIATE_DANGER", "ESCAPE_IMMEDIATE_DANGER",
    "RESOURCE_PROGRESS", "RESOURCE_REGRESS", "ATTACK_PROGRESS", "ATTACK_REGRESS",
)


def official_components(events: list[str] | tuple[str, ...]) -> dict[str, float]:
    counts = {name: events.count(name) for name in OFFICIAL}
    if counts[e.KILLED_SELF]:
        counts[e.GOT_KILLED] = 0  # self-kill has one exclusive total, never -8 plus -5
    return {name: OFFICIAL[name] * count for name, count in counts.items() if count}


def _danger_at(state: dict | None) -> int:
    if state is None:
        return INF_TIME
    return int(danger_time_map(state)[tuple(state["self"][3])])


def _free_exits(state: dict | None) -> int | None:
    if state is None or not state["others"]:
        return None
    own = tuple(state["self"][3])
    nearest = min(
        state["others"],
        key=lambda other: abs(other[3][0] - own[0]) + abs(other[3][1] - own[1]),
    )
    x, y = nearest[3]
    field = np.asarray(state["field"])
    occupied = {tuple(item[3]) for item in state["others"]} | {own}
    occupied.update(tuple(position) for position, _ in state["bombs"])
    return sum(
        0 <= x + dx < field.shape[0] and 0 <= y + dy < field.shape[1]
        and field[x + dx, y + dy] == 0 and (x + dx, y + dy) not in occupied
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0))
    )


def shaping_components(old: dict | None, new: dict | None, action: str, events, stage: str) -> dict[str, float]:
    if old is None or new is None:
        return {}
    result: dict[str, float] = {}
    if idle_wait(old, action):
        result["IDLE_WAIT"] = -0.03
    old_danger, new_danger = _danger_at(old), _danger_at(new)
    if old_danger > 2 and new_danger <= 1:
        result["ENTER_IMMEDIATE_DANGER"] = -0.08
    elif old_danger <= 2 and new_danger > 2:
        result["ESCAPE_IMMEDIATE_DANGER"] = 0.05
    if e.COIN_COLLECTED not in events and e.CRATE_DESTROYED not in events:
        old_distance, new_distance = resource_distance(old), resource_distance(new)
        if old_distance is not None and new_distance is not None:
            if new_distance < old_distance:
                result["RESOURCE_PROGRESS"] = 0.03
            elif new_distance > old_distance:
                result["RESOURCE_REGRESS"] = -0.02
    if stage in ("task3_peaceful", "task3_coin", "task4"):
        old_exits, new_exits = _free_exits(old), _free_exits(new)
        self_safe = _danger_at(new) > 2 and bool(np.any(legal_action_mask(new)[:4]))
        if self_safe and old_exits is not None and new_exits is not None:
            if new_exits < old_exits:
                result["ATTACK_PROGRESS"] = 0.03
            elif new_exits > old_exits:
                result["ATTACK_REGRESS"] = -0.02
    multiplier = {"task1": 1.0, "task2": 1.0, "task3_peaceful": 0.5,
                  "task3_coin": 0.5, "task4": 0.25}[stage]
    return {name: value * multiplier for name, value in result.items()}


def reward_components(old, new, action: str, events, stage: str) -> dict[str, float]:
    result = official_components(tuple(events))
    result.update(shaping_components(old, new, action, events, stage))
    shaping_total = sum(abs(value) for name, value in result.items() if name in SHAPING_NAMES)
    if shaping_total > 0.2 + 1e-12:
        raise RuntimeError(f"shaping bound violated: {shaping_total}")
    return result


def reward_total(old, new, action: str, events, stage: str) -> tuple[float, dict[str, float]]:
    components = reward_components(old, new, action, events, stage)
    return float(sum(components.values())), components
