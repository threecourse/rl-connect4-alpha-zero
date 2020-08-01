"""
ゲームの全般にかかるユーティリティ
"""

import enum
from dataclasses import dataclass
from typing import Optional

BH = 6
BW = 7
A = 4

MOVES = BH * BW

BLACK = 1
WHITE = -1
EMPTY = 0
BLACK_STR = "o"
WHITE_STR = "x"
EMPTY_STR = "."

O_WIN = 1
E_WIN = -1
ON_GOING = 9
DRAW = 0
NOT_EXIST = -99

# noinspection PyArgumentList
Player = enum.Enum("Player", "black white")
# noinspection PyArgumentList
Winner = enum.Enum("Winner", "black white draw")


@dataclass(eq=True, frozen=True)
class CounterKey:
    """盤面を表すキー"""
    black: int
    white: int
    next_player: enum


@dataclass
class ActionWithEvaluation:
    """行動および評価"""
    action: Optional[int]
    n: float
    q: float


def another_player(player: Player) -> Player:
    """反対側のプレイヤーを返す"""

    return Player.white if player == Player.black else Player.black
