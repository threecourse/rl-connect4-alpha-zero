from logging import getLogger
from typing import Optional

from c4_zero.agent.model.model import C4Model
from c4_zero.agent.model.model_util import ModelUtil
from c4_zero.agent.player import BasePlayer
from c4_zero.agent.player_alphazero import ThoughtItem, EvaluationItem
from c4_zero.agent.player_creator import create_player
from c4_zero.config import Config
from c4_zero.env.c4_environment import C4Env
from c4_zero.env.c4_util import BW, MOVES, Player, BLACK, WHITE
from c4_zero.env.environment import Env

logger = getLogger(__name__)


class GUIGameManager:
    """人間と対戦する場合のゲームマネージャー"""
    config: Config
    human_color: Player
    env: Env
    model: C4Model
    ai: BasePlayer
    last_history: Optional[ThoughtItem]
    last_evaluation: Optional[EvaluationItem]

    def __init__(self, config: Config):
        # envの初期化
        C4Env.initialize(config.env)

        self.config = config
        self.human_color = None
        self.env = C4Env.create_init()
        self.env.reset()
        self.model = self._load_model()
        self.last_evaluation = None
        self.last_history = None

    def start_game(self):
        """ゲームを開始する"""
        self.env.reset()
        self.ai = create_player(self.config, self.config.play_gui.player, self.model)

    def start_game_with_state(self, state_string):
        """ゲームをある状態から開始する"""
        try:
            env = C4Env.from_string(state_string)
        except Exception as ex:
            import traceback
            print(ex)
            print(traceback.format_exc())
            return False
        self.env = env
        self.ai = create_player(self.config, self.config.play_gui.player, self.model)
        return True

    @property
    def turn(self):
        return self.env.turn

    @property
    def over(self):
        return self.env.done

    @property
    def winner(self):
        return self.env.winner

    @property
    def is_next_human(self):
        return self.next_player == self.human_color

    @property
    def next_player(self):
        return self.env.next_player

    def stone(self, y, x):
        value = self.env.board.get_value(y, x)
        if value == BLACK:
            return Player.black
        elif value == WHITE:
            return Player.white
        return None

    def available(self, y, x):
        legal_moves = self.env.legal_moves()
        return legal_moves[y * BW + x] > 0

    def legal_moves_str(self):
        ret = ""
        for i in range(MOVES):
            y, x = i // BW, i % BW
            if self.available(y, x):
                ret += "o"
            else:
                ret += "."
        return ret

    def move(self, y, x):
        """次の手を打つ"""
        action = int(y * BW + x)
        assert 0 <= action < MOVES
        self.env.step(action)
        self.last_history = None
        self.last_evaluation = None

    def _load_model(self):
        return ModelUtil.load_model(self.config, self.config.play_gui.player.use_newest)

    def think_by_ai(self):
        """aiによる思考を行う"""
        env_black = self.env.get_env_black()
        action = self.ai.action(env_black)

        # show evaluations
        self.last_history = self.ai.ask_thought(env_black)
        self.last_evaluation = self.ai.ask_evaluation(env_black)

    def move_by_ai(self):
        """aiによる思考を行い、一手指す"""
        env_black = self.env.get_env_black()
        action = self.ai.action(env_black)

        self.env.step(action)

        # show evaluations
        self.last_history = self.ai.ask_thought(env_black)
        self.last_evaluation = self.ai.ask_evaluation(env_black)
