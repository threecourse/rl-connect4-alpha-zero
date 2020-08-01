import os
from typing import Optional

from moke_config import ConfigBase


def _project_dir():
    d = os.path.dirname
    return d(d(d(os.path.abspath(__file__))))


def _data_dir():
    return os.path.join(_project_dir(), "data")


class Config(ConfigBase):
    def __init__(self):
        self.type = "default"
        self.env = EnvironmentConfig()
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.self_play = SelfPlayConfig()
        self.optimize = OptimizeConfig()
        self.evaluate = EvaluateConfig()
        self.play_gui = PlayGuiConfig()


class EnvironmentConfig(ConfigBase):

    def __init__(self):
        self.env_solver_turn: Optional[int] = None


class ResourceConfig(ConfigBase):
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        self.model_best_config_path = os.path.join(self.model_dir, "model_best_config.json")
        self.model_best_weight_path = os.path.join(self.model_dir, "model_best_weight.h5")

        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_config_filename = "model_config.json"
        self.next_generation_model_weight_filename = "model_weight.h5"

        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"
        self.self_play_ggf_data_dir = os.path.join(self.data_dir, "self_play-ggf")
        self.eval_ggf_data_dir = os.path.join(self.data_dir, "eval_play-ggf")
        self.ggf_filename_tmpl = "self_play-%s.ggf"

        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.tensorboard_log_dir = os.path.join(self.log_dir, 'tensorboard')
        self.self_play_log_dir = os.path.join(self.tensorboard_log_dir, "self_play")
        self.optimize_log_dir = os.path.join(self.tensorboard_log_dir, "optimize")
        self.evaluate_log_dir = os.path.join(self.tensorboard_log_dir, "evaluate")
        self.force_learing_rate_file = os.path.join(self.data_dir, ".force-lr")
        self.self_play_status_file = os.path.join(self.data_dir, ".self-play_status.json")
        self.optimize_status_file = os.path.join(self.data_dir, ".optimize_status.json")

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir, self.self_play_log_dir, self.self_play_ggf_data_dir,
                self.eval_ggf_data_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class SelfPlayConfig(ConfigBase):
    def __init__(self):
        self.player = PlayerConfig()
        self.multi_process_num = 16
        self.nb_game_in_file = 2
        self.max_file_num = 800
        self.save_policy_of_tau_1 = True
        self.enable_ggf_data = True
        self.drop_draw_game_rate = 0
        self.share_mtcs_info_in_self_play = True
        self.reset_mtcs_info_per_game = 1

        # debugging
        self.use_self_play_worker = True


class OptimizeConfig(ConfigBase):
    def __init__(self):
        self.wait_after_save_model_ratio = 1  # モデル保存後の待機
        self.batch_size = 256  # バッチサイズ
        self.min_data_size_to_learn = 100000  # 学習を開始する最小の学習データのレコード数
        self.use_tensorboard = True    # TensorBoardを使用するか
        self.epoch_to_checkpoint = 1  # 学習ごとに何epochの学習を行うか
        self.save_model_steps = 200  # 何stepごとにモデルの保存（および評価）を行うか
        self.logging_per_steps = 100  # 何stepごとにtensorflowに学習ログの出力を行うか
        self.delete_self_play_after_number_of_training = 0  # 何回の学習後に学習データを削除するか（0は削除しない）
        self.lr_schedules = [
            (0, 0.01),
            (150000, 0.001),
            (300000, 0.0001),
        ]
        self.max_save_model_count = 20  # メモリリーク対策のため、一定回数のモデル保存ごとにプロセスを再起動する


class EvaluateConfig(ConfigBase):
    def __init__(self):
        self.game_num = 20  # 1回のevaluateでの対戦数
        self.replace_rate = 0.55  # not used
        self.evaluate_per_steps = 200  # 何stepごとにevaluateするか
        self.evaluate_random_rates = [0, 10, 30, 100]  # solverのランダム打ちの割合
        self.player1 = PlayerConfig()
        self.player1.simulation_num_per_move = 400
        self.player1.thinking_loop = 1
        self.player1.change_tau_turn = 0
        self.player1.noise_eps = 0
        self.player1.disable_resignation_rate = 0
        self.player2 = PlayerConfig()
        self.player2.simulation_num_per_move = 400
        self.player2.thinking_loop = 1
        self.player2.change_tau_turn = 0
        self.player2.noise_eps = 0
        self.player2.disable_resignation_rate = 0


class PlayGuiConfig(ConfigBase):
    def __init__(self):
        self.player = PlayerConfig()
        self.player.parallel_search_num = 8
        self.player.noise_eps = 0
        self.player.change_tau_turn = 0
        self.player.resign_threshold = None
        self.player.use_newest = True


class PlayerConfig(ConfigBase):
    """プレイヤーコンフィグ"""

    def __init__(self):
        self.player_type = "alphazero"

        self.thinking_loop = 10  # 思考ループの回数
        # 思考ループごとのプレイアウト回数（下の変数で上書きされるため、使用されない）
        self.simulation_num_per_move = 200
        # 思考ループごとのプレイアウト回数（総ゲーム回数でスケジューリングされる）
        self.schedule_of_simulation_num_per_move = [
            (0, 8),
            (300, 50),
            (2000, 200),
        ]
        self.required_visit_to_decide_action = 400  # 着手選択で、Q値の差が小さいときに何回以上の手の選択を必要とするか
        self.change_tau_turn = 4  # 着手選択を最大にするか、回数に比例にするか
        self.start_rethinking_turn = 8  # 思考ループを2回以上行うようにするターン
        self.c_puct = 1  # プレイアウトの着手選択のQ値とP値のバランスのパラメータ
        self.noise_eps = 0.25  # プレイアウトのルートノードの着手選択のノイズの比率
        self.dirichlet_alpha = 0.5  # プレイアウトのルートノードの着手選択のディリクレ分布のパラメータ
        self.policy_decay_turn = 60  # プレイアウトの着手選択の温度に関連するパラメータ - このターンを超えると温度が1となる
        self.policy_decay_power = 3  # プレイアウトの着手選択の温度に関連するパラメータ
        self.virtual_loss = 3  # バーチャルロスの数

        self.prediction_queue_size = 16  # プレイヤーごとの予測用キューの最大数（それを超える場合は待つ）
        self.parallel_search_num = 8  # プレイヤー内で並行に実行する盤面数
        self.prediction_worker_sleep_sec = 0.0001  # プレイアウトの推論中、キューが空のときに待つ時間
        self.wait_for_expanding_sleep_sec = 0.00001  # 対象盤面がExpand中である場合にasyncio.sleepする時間

        self.allowed_resign_turn = 20  # 投了を許可するターン
        self.disable_resignation_rate = 0.1  # 投了を不許可にする確率
        self.resign_threshold = -0.9  # 評価がどの程度であれば投了するか
        self.resign_threshold_delta = 0.01  # 誤投了率が閾値を超えたとき、それより少ないときにどの程度動かすか
        self.false_positive_threshold = 0.05  # 誤投了率の閾値

        self.use_newest = True  # 最新モデルを使うかどうか
        self.solver_player_random_ratio = 0.0  # ソルバーのランダム着手率（SolverPlayerの場合）

        # デバッグ用
        self.complete_random = False  # プレイアウトでの着手を完全ランダムにする場合


class ModelConfig(ConfigBase):
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 10
        self.l2_reg = 1e-4
        self.value_fc_size = 256
