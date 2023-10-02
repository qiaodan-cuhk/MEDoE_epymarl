from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
import torch as th


class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        self.batch_size = 256 # TODO

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        episode_returns = th.zeros(self.batch_size)
        track_terminated = th.zeros(self.batch_size, dtype=bool)
        episode_lengths = th.zeros(self.batch_size, dtype=int)
        self.mac.init_hidden(batch_size=self.batch_size)
        actions = th.zeros((self.batch_size, 4), dtype=int) # TODO hardcoded n_agents

        while not all(track_terminated):

            pre_transition_data = {
                "state": self.env.get_state()[~track_terminated],
                "avail_actions": self.env.get_avail_actions()[~track_terminated],
                "obs": self.env.get_obs()[~track_terminated],
            }

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(track_terminated) if not termed]
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            active_actions = self.mac.select_actions(self.batch, bs=envs_not_terminated, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
            actions[~track_terminated] = active_actions
            # need to pad actions for finished environments
            reward, terminated, env_info = self.env.step(actions)

            post_transition_data = {
                "actions": actions[~track_terminated],
                "reward": reward[~track_terminated],
                "terminated": terminated[~track_terminated].byte(),
            }

            if test_mode and self.args.render:
                self.env.render()

            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t)

            self.t += 1
            if not test_mode:
                self.t_env += sum(~track_terminated).item()
            episode_lengths[~track_terminated] += 1

            episode_returns += reward * (~track_terminated)
            track_terminated = track_terminated | terminated


        """
        last_data = {
            "state": self.env.get_state()[~track_terminated],
            "avail_actions": self.env.get_avail_actions()[~track_terminated],
            "obs": self.env.get_obs()[~track_terminated],
        }

        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        """

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns.tolist())

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
