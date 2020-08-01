import asyncio
from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict
from logging import getLogger
from typing import Any, Dict, List, Optional, Set

import numpy as np

from c4_zero.agent.api import InferenceAPIBase, InferenceSimpleAPI
from c4_zero.agent.model.model import C4Model
from c4_zero.agent.player import BasePlayer
from c4_zero.agent.player_util import AlphaZeroPlayerUtil
from c4_zero.agent.player_util import QueueItem, ThoughtItem, CallbackInMCTS, MCTSInfo, EvaluationItem
from c4_zero.config import Config, PlayerConfig
from c4_zero.env.c4_util import MOVES, Player, Winner, CounterKey, ActionWithEvaluation
from c4_zero.env.data import MoveData, InferenceData
from c4_zero.env.environment import Env

logger = getLogger(__name__)


class AlphaZeroPlayer(BasePlayer):
    """AlphaZero方式によるプレイヤー

    Attributes:
        config: コンフィグ
        model: モデル
        player_config: プレイヤーコンフィグ
        enable_resign: 投了を許可するかどうか
        api: 推論API
        moves_data: 学習用データの保持

        # 非同期処理関連
        loop: asyncioのイベントループ
        sem: asyncioのセマフォ
        prediction_queue: 推論処理用のキュー

        # 以下は探索開始時に更新する
        callback_in_mtcs: MTCSの情報

        # 以下は探索中に更新する
        var_n: N -- キーとなる盤面で、あるactionを取った回数
        var_w: W -- キーとなる盤面で、あるactionを取った場合の勝数
        var_p: P -- キーとなる盤面で、ポリシーネットワークによる各actionの値
        expanded: Expandを行ったキーのSet
        now_expanding: Expandを行っているキーのSet
        running_simulation_num: 非同期実行の探索進行数
        thinking_history: 探索した各局面でのaction, policy, Q, N および次の手からの探索のQ, N
        _resigned: 投了したかどうか
        requested_stop_thinking: 思考を中断する指示が設定されたかどうか
    """

    config: Config
    model: C4Model
    player_config: PlayerConfig
    enable_resign: bool
    api: InferenceAPIBase

    _moves_data: List[MoveData]
    loop: Any
    sem: asyncio.Semaphore
    prediction_queue: Queue

    callback_in_mtcs: Any

    var_n: Dict[CounterKey, np.ndarray]
    var_w: Dict[CounterKey, np.ndarray]
    var_p: Dict[CounterKey, np.ndarray]
    expanded: Set[CounterKey]
    now_expanding: Set[CounterKey]
    running_simulation_num: int
    thinking_history: Dict[CounterKey, ThoughtItem]
    _resigned: bool
    requested_stop_thinking: bool

    def __init__(self, config: Config, player_config: PlayerConfig, model: Optional[C4Model], enable_resign=True,
                 mtcs_info=None, api=None):

        super().__init__()
        self.config = config
        self.model = model
        self.player_config = player_config
        self.enable_resign = enable_resign
        self.api = api or InferenceSimpleAPI(self.config, self.model)
        self._moves_data = []
        mtcs_info = mtcs_info or self.create_mtcs_info()

        # 非同期処理関連
        self.prediction_queue = Queue(self.player_config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.player_config.parallel_search_num)

        # MCTS
        self.var_n, self.var_w, self.var_p = mtcs_info.var_n, mtcs_info.var_w, mtcs_info.var_p
        self.expanded = set(self.var_p.keys())
        self.now_expanding = set()
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        self.thinking_history = {}  # for fun
        self._resigned = False
        self.requested_stop_thinking = False

    def var_q(self, key) -> np.array:
        """ある盤面における各手のQ値"""
        return self.var_w[key] / (self.var_n[key] + 1e-5)

    def var_n_normalized(self, key) -> np.array:
        """探索回数に比例する確率"""
        s = np.sum(self.var_n[key])
        if s == 0:
            return self.var_n[key]
        return self.var_n[key] / s

    def action(self, env_black: Env, callback_in_mtcs: CallbackInMCTS = None) -> int:
        """黒番の盤面が与えられたときの行動を返す

        :return action=move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right)
        """
        assert (env_black.next_player == Player.black)
        action_with_eval = self.action_with_evaluation(env_black, callback_in_mtcs=callback_in_mtcs)
        return action_with_eval.action

    def action_with_evaluation(self, env_black: Env, callback_in_mtcs: CallbackInMCTS = None) -> ActionWithEvaluation:
        """黒番の盤面が与えられたときの行動と評価を返す

        :return ActionWithEvaluation(
                    action=move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right),
                    n=N of the action,
                    q=W/N of the action,
                )
        """
        assert (env_black.next_player == Player.black)
        key = env_black.counter_key()
        self.callback_in_mtcs = callback_in_mtcs
        pc = self.player_config

        # 探索ソルバーは省略

        # 探索ループを複数回行う
        for tl in range(self.player_config.thinking_loop):
            self._search_moves(env_black)

            policy = self._calc_policy(env_black)
            action = int(np.random.choice(range(MOVES), p=policy))
            action_by_value = int(np.argmax(self.var_q(key) + (self.var_n[key] > 0) * 100))
            value_diff = self.var_q(key)[action] - self.var_q(key)[action_by_value]

            # start_rethinking_turnより前では1回しかsearchしない
            if env_black.turn <= pc.start_rethinking_turn or self.requested_stop_thinking or \
                    (value_diff > -0.01 and self.var_n[key][action] >= pc.required_visit_to_decide_action):
                break

        # 思考の結果のアップデート（学習には不要）
        self._update_thinking_history(env_black, action, policy)

        # 投了関連
        if self.player_config.resign_threshold is not None and \
                np.max(self.var_q(key) - (self.var_n[key] == 0) * 10) <= self.player_config.resign_threshold:
            self._resigned = True
            if self.enable_resign:
                if env_black.turn >= self.player_config.allowed_resign_turn:
                    return ActionWithEvaluation(None, 0, 0)  # means resign
                else:
                    logger.debug(
                        f"Want to resign but disallowed turn {env_black.turn} < {self.player_config.allowed_resign_turn}")

        # 学習用にデータを保存する
        # ポリシーは探索回数に比例する確率とする
        saved_policy = self.var_n_normalized(key) if self.config.self_play.save_policy_of_tau_1 else policy
        self._add_data_to_move_buffer(env_black, saved_policy)

        return ActionWithEvaluation(action=action, n=self.var_n[key][action], q=self.var_q(key)[action])

    def ask_evaluation(self, env_black: Env) -> EvaluationItem:
        """現時点のP、Vと次の局面のそれぞれのVを出力する（デバッグ用）"""
        inference_data_list = []

        # 現在の状態
        inference_data = env_black.to_inference()
        inference_data_list.append(inference_data)

        # 1手進めた状態（合法手のみ）
        legal_moves = env_black.legal_moves()
        for action, flag in enumerate(legal_moves):
            if flag == 1:
                # 一度コピーする
                _env = env_black.copy()
                _env.step(action)
                _inference_data = _env.to_inference()
                inference_data_list.append(_inference_data)

        inference_result = self.api.predict_inference_data_list(inference_data_list)
        pred_policy_ary, pred_value_ary = inference_result.p_ary, inference_result.v_ary

        # 次の手番から見た評価値のため、逆にする必要がある
        pred_value_ary[1:] *= -1
        curr_policy = pred_policy_ary[0]
        curr_value = pred_value_ary[0]
        values_after_move = np.zeros(MOVES)
        i = 1
        for action, flag in enumerate(legal_moves):
            if flag == 1:
                values_after_move[action] = pred_value_ary[i]
                i += 1

        eval_debug = EvaluationItem(curr_policy, curr_value, values_after_move)

        return eval_debug

    def ask_thought(self, env_black: Env) -> Optional[ThoughtItem]:
        """その局面についての思考の経過を返す"""
        key = env_black.counter_key()
        return self.thinking_history.get(key)

    def finish_game(self, z: float) -> None:
        """ゲームの終了処理

        :param z: プレイヤーにとっての勝敗

        学習データに結果を追記する
        """
        for move in self._moves_data:
            move.z = z

    @staticmethod
    def create_mtcs_info() -> MCTSInfo:
        """MCTSの情報を作成する"""
        return MCTSInfo(defaultdict(lambda: np.zeros((MOVES,))),
                        defaultdict(lambda: np.zeros((MOVES,))),
                        defaultdict(lambda: np.zeros((MOVES,))))

    @property
    def moves_data(self) -> List[MoveData]:
        return self._moves_data

    @property
    def resigned(self) -> bool:
        return self._resigned

    def _update_thinking_history(self, env_black: Env, action: int, policy: np.array) -> None:
        """思考の経過をアップデートする"""
        # その局面でのaction, policy, Q, N
        key = env_black.counter_key()
        self.thinking_history[key] = ThoughtItem(action, policy, self.var_q(key), self.var_n[key].copy())

    def _stop_thinking(self) -> None:
        """思考の停止を指示する"""
        self.requested_stop_thinking = True

    def _add_data_to_move_buffer(self, env_black: Env, policy: np.array) -> None:
        """学習用のデータを保存する"""
        self._moves_data += env_black.moves_data(policy)

    def _search_moves(self, env_black: Env) -> None:
        """探索を行う

        * プレイアウトを一定回数行う
        * 非同期処理により実行する
        """
        loop = self.loop
        self.running_simulation_num = 0
        self.requested_stop_thinking = False

        coroutine_list = []
        for it in range(self.player_config.simulation_num_per_move):
            cor = self._start_search_my_move(env_black)
            coroutine_list.append(cor)

        coroutine_list.append(self._prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    def _calc_policy(self, env_black: Env) -> np.array:
        """探索結果に基づき、次の手を決める

        :param env_black: 黒番の盤面
        :return: 次の手の着手確率
        """
        key = env_black.counter_key()
        if env_black.turn < self.player_config.change_tau_turn:
            # tau=1で選択する（探索回数に比例する確率）
            return self.var_n_normalized(key)
        else:
            # tau=0で選択する（もっともnが大きい手）
            action = np.argmax(self.var_n[key])
            ret = np.zeros(MOVES)
            ret[action] = 1
            return ret

    async def _start_search_my_move(self, env_black: Env) -> Optional[float]:
        """各非同期処理ごとの探索の開始

        :param env_black: 黒番の盤面
        :return: 対象のプレイヤーの勝敗

        探索においては、黒番プレイヤーの視点で行う（白番プレイヤーの場合には、反転済みの盤面となっている）
        ルートノードにおいては、黒番であるとする
        """
        self.running_simulation_num += 1
        root_key = env_black.counter_key()

        with await self.sem:  # 同時実行数を制限
            if self.requested_stop_thinking:
                self.running_simulation_num -= 1
                return None
            env = env_black.copy()
            leaf_v = await self._search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            if self.callback_in_mtcs and self.callback_in_mtcs.per_sim > 0 and \
                    self.running_simulation_num % self.callback_in_mtcs.per_sim == 0:
                self.callback_in_mtcs.callback(list(self.var_q(root_key)), list(self.var_n[root_key]))
            return leaf_v

    async def _search_my_move(self, env: Env, is_root_node=False) -> float:
        """盤面での探索

        QとVは探索を行うプレイヤーから見た値（常に黒とする）
        Pは次の手番のプレイヤーから見た値（白も黒もありうる）
        :param env: 盤面
        :param is_root_node: ルートノードか否か
        :return: 対象のプレイヤーの勝敗
        """
        if env.done:
            if env.winner == Winner.black:
                return 1
            elif env.winner == Winner.white:
                return -1
            else:
                return 0

        key = env.counter_key()
        another_side_key = env.another_side_counter_key()

        # ソルバー使用の場合の処理は省略

        # 対象局面がExpand中である場合、一定時間待つ
        while key in self.now_expanding:
            await asyncio.sleep(self.player_config.wait_for_expanding_sleep_sec)

        # 対象局面がExpand中されていない場合、つまりリーフノードの場合
        if key not in self.expanded:
            leaf_v = await self._expand_and_evaluate(env)
            if env.next_player == Player.black:
                return leaf_v  # Value for black
            else:
                return -leaf_v  # Value for white == -Value for black

        # バーチャルロスの設定
        virtual_loss = self.player_config.virtual_loss
        virtual_loss_for_w = virtual_loss if env.next_player == Player.black else -virtual_loss

        # 探索における次の手の選択を行う
        action_t = self._select_action_q_and_u(env, is_root_node)
        env.step(action_t)

        # バーチャルロスを戻す
        self.var_n[key][action_t] += virtual_loss
        self.var_w[key][action_t] -= virtual_loss_for_w
        leaf_v = await self._search_my_move(env)

        # 探索から戻ってくるときに更新を行う
        # N, Wの更新（反転した局面も含む）
        self.var_n[key][action_t] += - virtual_loss + 1
        self.var_w[key][action_t] += virtual_loss_for_w + leaf_v
        self.var_n[another_side_key][action_t] += 1
        self.var_w[another_side_key][action_t] -= leaf_v  # must flip the sign.
        return leaf_v

    async def _expand_and_evaluate(self, env: Env) -> float:
        """ノードのExpandを行う

        * 推論APIに状態を送付し、PとVを取得する
        * 対象盤面のPをアップデートする

        :param env: 盤面
        :return: 手番におけるVの値
        """

        key = env.counter_key()
        another_side_key = env.another_side_counter_key()
        self.now_expanding.add(key)

        # 予測を行う（変換を考慮することもできる）
        # is_flip_vertical = random() < 0.5
        # rotate_right_num = int(random() * 4)
        inference_data = env.to_inference()

        if self.player_config.complete_random:
            leaf_p, leaf_v = np.ones(MOVES) / float(MOVES), 0.5
        else:
            future = await self._predict(inference_data)  # type: Future
            await future
            leaf_p, leaf_v = future.result()

        # 変換がある場合は元に戻す
        leaf_p = env.leaf_p_restore_transform(leaf_p)

        # Pのアップデート
        self.var_p[key] = leaf_p  # P is value for next_player (black or white)
        self.var_p[another_side_key] = leaf_p

        self.expanded.add(key)
        self.now_expanding.remove(key)

        return float(leaf_v)

    async def _predict(self, inference_data: InferenceData) -> Future:
        """非同期処理により推論を行うため、入力値をキューにセットする"""
        future = self.loop.create_future()
        item = QueueItem(inference_data, future)
        await self.prediction_queue.put(item)
        return future

    async def _prediction_worker(self) -> None:
        """非同期処理により推論の実行を行う

          探索用のコルーチンとともに、この推論を行うコルーチンも含めている
        """
        q = self.prediction_queue

        margin = 10  # 他の探索が開始される前に終了することを防ぐ
        # 探索が走っている間は継続する
        while self.running_simulation_num > 0 or margin > 0:
            # キューが空であれば待つ
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.player_config.prediction_worker_sleep_sec)
                continue

            # 推論サーバへの送付を行う
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: List[QueueItem]
            inference_data_list = [item.data for item in item_list]
            inference_result = self.api.predict_inference_data_list(inference_data_list)
            policy_ary, value_ary = inference_result.p_ary, inference_result.v_ary
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    def _select_action_q_and_u(self, env: Env, is_root_node) -> int:
        """Q値とu値に基づいてプレイアウト中の手の選択を行う"""
        key = env.counter_key()
        legal_moves = env.legal_moves()

        # 合法手に制限して標準化する
        p_ = self.var_p[key]
        p_ = p_ * legal_moves
        if np.sum(p_) > 0:
            # decay policy gradually in the end phase
            _pc = self.player_config
            temperature = min(np.exp(1 - np.power(env.turn / _pc.policy_decay_turn, _pc.policy_decay_power)), 1)
            p_ = AlphaZeroPlayerUtil.normalize_by_temparature(p_, temperature)

        # ルートノードの場合、ノイズを与える
        if is_root_node and self.player_config.noise_eps > 0:
            noise = AlphaZeroPlayerUtil.dirichlet_noise_of_mask(legal_moves, self.player_config.dirichlet_alpha)
            p_ = (1 - self.player_config.noise_eps) * p_ + self.player_config.noise_eps * noise

        # u値を求める
        xx_ = np.sqrt(np.sum(self.var_n[key]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        u_ = self.player_config.c_puct * p_ * xx_ / (1 + self.var_n[key])

        # Q値を求める（相手の手を求めている場合、反転する必要がある）
        if env.next_player == Player.black:
            q_ = self.var_q(key)
        else:
            q_ = -self.var_q(key)

        # Q値とu値を統合した値が最大のものを手として選択する
        v_ = (q_ + u_ + 1000) * legal_moves
        action_t = int(np.argmax(v_))
        return action_t
