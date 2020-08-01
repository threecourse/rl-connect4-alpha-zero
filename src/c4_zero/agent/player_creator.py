from typing import Optional

from c4_zero.agent import player_alphazero
from c4_zero.agent import player_solver
from c4_zero.agent.model.model import C4Model
from c4_zero.config import Config, PlayerConfig


def create_player(config: Config, player_config: PlayerConfig, model: Optional[C4Model],
                  enable_resign=True, mtcs_info=None, api=None, solver=None):
    """プレイヤーを作成する"""

    if player_config.player_type == "alphazero":
        return player_alphazero.AlphaZeroPlayer(config, player_config, model, enable_resign, mtcs_info, api)
    elif player_config.player_type == "solver":
        return player_solver.SolverPlayer(config, player_config, solver)
    else:
        raise NotImplementedError
