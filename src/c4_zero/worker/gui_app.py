import json
import os

import tornado.escape
import tornado.ioloop
import tornado.web
from tornado.options import define

from c4_zero.config import Config
from c4_zero.env.c4_util import BH, BW, Player
from c4_zero.worker.gui_game_manager import GUIGameManager


def start(config: Config):
    """PlayWithHumanモデルを用いて、GUIによる対戦を行う"""
    play_with_human = GUIGameManager(config)
    play_with_human.start_game()

    define("port", default=8888, help="run on the given port", type=int)
    app = Application(play_with_human)
    port = int(os.environ.get("PORT", 8888))
    app.listen(port)
    print("Run server on port: {}".format(port))

    tornado.ioloop.IOLoop.current().start()


class IndexHandler(tornado.web.RequestHandler):
    """index.htmlの描画"""

    def get(self):
        self.render("index.html")


class GameHandler(tornado.web.RequestHandler):
    """htmlからリクエストを受ける"""
    game_manager: GUIGameManager

    def initialize(self, game_manager: GUIGameManager):
        # RequestHandlerの共通の初期化メソッド
        self.game_manager = game_manager

    def post(self):
        # get posted data from javascript
        data = tornado.escape.json_decode(self.request.body)
        cmd_type = data["cmd_type"]
        cmd_data = data["cmd_data"]
        cmd_data2 = data["cmd_data2"]
        evaluated = 0

        # command
        if cmd_type == "new_game":
            self.new_game()
            evaluated = 0

        if cmd_type == "load_game":
            state_string = cmd_data2
            self.new_game_with_state(state_string)
            evaluated = 0

        if cmd_type == "try_move":
            y, x = int(cmd_data[0]), int(cmd_data[2])
            self.try_move(y, x)
            evaluated = 0

        if cmd_type == "think_ai":
            self.think_ai()
            evaluated = 1

        if cmd_type == "move_ai":
            self.move_ai()
            evaluated = 1

        # return values to javascript

        board_str = self.board_str()
        messages = []
        messages.append(f"turn: {self.game_manager.turn}")
        messages.append(f"next_player: {self.game_manager.next_player}")
        if self.game_manager.over:
            messages.append(f"game is over: winner {self.game_manager.winner}")
        legal_moves_str = self.game_manager.legal_moves_str()

        last_history = self.game_manager.last_history
        last_evaluation = self.game_manager.last_evaluation

        if evaluated and last_history is not None:
            q_values = last_history.ary_q.reshape((BH, BW)).tolist()
            n_values = last_history.ary_n.reshape((BH, BW)).tolist()
        else:
            q_values = ""
            n_values = ""

        if evaluated and last_evaluation is not None:
            p_values = last_evaluation.curr_policy
            v_current = last_evaluation.curr_value
            v_values = last_evaluation.values_after_move
            p_values = p_values.reshape((BH, BW)).tolist()
            v_current = float(v_current)
            v_values = v_values.reshape((BH, BW)).tolist()
        else:
            p_values = ""
            v_current = 0.0
            v_values = ""

        dic = {"evaluated": evaluated,
               "board": board_str,
               "message": "\n".join(messages),
               "legal_moves": legal_moves_str,
               "q_values": q_values,
               "n_values": n_values,
               "p_values": p_values,
               "v_current": v_current,
               "v_values": v_values}
        self.write(json.dumps(dic))

    def new_game(self):
        self.game_manager.start_game()

    def new_game_with_state(self, state_string):
        success = self.game_manager.start_game_with_state(state_string)
        if not success:
            print("failed to read state")

    def try_move(self, y, x):
        print(f"try_move {y} {x}")
        if self.game_manager.over:
            print("game is already over")
            return
        if not self.game_manager.available(y, x):
            print("action is unavailable")
            return
        self.game_manager.move(y, x)

    def think_ai(self):
        if not self.game_manager.over:
            self.game_manager.think_by_ai()

    def move_ai(self):
        if not self.game_manager.over:
            self.game_manager.move_by_ai()

    def board_str(self) -> str:
        ret = ""
        for y in range(BH):
            for x in range(BW):
                stone = self.game_manager.stone(y, x)
                if stone == Player.black:
                    ret += "o"
                elif stone == Player.white:
                    ret += "x"
                else:
                    ret += "."
        return ret


class Application(tornado.web.Application):

    def __init__(self, game_manager: GUIGameManager):
        handlers = [
            (r"/", IndexHandler),
            (r"/game", GameHandler, dict(game_manager=game_manager)),
        ]

        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "gui_app/templates"),
            static_path=os.path.join(os.path.dirname(__file__), "gui_app/static"),
            cookie_secret=os.environ.get("SECRET_TOKEN", "__GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__"),
            debug=True,
        )

        super(Application, self).__init__(handlers, **settings)
