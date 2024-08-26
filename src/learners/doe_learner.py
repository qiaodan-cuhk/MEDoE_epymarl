# Modified from fst doe_ac.py
import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry
from modules.doe import REGISTRY as doe_resigtry
from modules.doe import doe_classifier_config_loader
from components.standarize_stream import RunningMeanStd


class DoEIA2C:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.logger = logger

        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)
    
        # Copy from doe agents
        self.n_train_envs = n_train_envs
        self.freeze_critic = freeze_critic/self.n_train_envs
        self.freeze_actor = freeze_actor/self.n_train_envs

        # Initialise update counters
        self.c_update = 0
        self.p_update = self.n_steps
        self.total_critic_updates = 0
        self.total_actor_updates = 0

        # adopt from doe agents
        self.ent_coef = 1.0 # needed to override the ent_coef called elsewhere

        self.base_temp = self.args.get("base_temp", 1.0)
        self.base_lr = self.args.get("base_lr", 1.0)
        self.base_ent = self.args.get("base_ent", 1.0)

        self.boost_temp_coef = self.args.get("boost_temp", 1.0)
        self.boost_lr_coef = self.args.get("boost_lr", 1.0)
        self.boost_ent_coef = self.args.get("boost_ent", 1.0)

        # self.ids = Iterable[AgentID]

        # mlp/joint_mlp

        self.doe_classifier = doe_classifier_config_loader(
                cfg=self.args.get("doe_classifier_cfg"),
                ids=self.ids
                )
        # self.ids = Iterable[AgentID] 
        # self.doe_classifier = doe_resigtry[args.doe_type].from_config(cfg=self.args.get("doe_classifier_cfg"), ids=self.ids)

    def _compute_gae(self, obs, agent_id=None):
        if agent_id is None:
            As, Gs, Vs = zip(*(self._compute_gae(obs, agent_id=agent_id)
                           for agent_id in self.ids))
            As = dict(zip(self.ids, As))
            Gs = dict(zip(self.ids, Gs))
            Vs = dict(zip(self.ids, Vs))
            return As, Gs, Vs
        else:
            V = self.critic_nets[agent_id](obs[agent_id])
            with torch.no_grad():
                G = torch.zeros_like(V)
                A = torch.zeros_like(V)
                V_n = self.critic_nets[agent_id](self.last_transition.n_obs[agent_id])
                G_n = V_n.clone()
                A_n = 0
                for i, t in enumerate(reversed(self.last_n_transitions)):
                    term = t.terminated.unsqueeze(1)
                    trunc = t.truncated.unsqueeze(1)
                    r = t.reward.unsqueeze(1)
                    if self.ignore_trunc:
                        V_n = ~(term|trunc) * V_n
                        G_n = ~(term|trunc) * G_n
                    else:
                        V_n = (~trunc * (V_n * ~term)
                               + trunc * (V_n))
                        G_n = (~trunc * (G_n * ~term)
                               + trunc * (G_n))
                    d = r + self.gamma * V_n - V[-i-1, :, :]
                    A[-i-1, :, :] = d + self.gamma * self.gae_lambda * A_n
                    G[-i-1, :, :] = r + self.gamma * G_n
                    A_n = A[-i-1, :, :]
                    G_n = G[-i-1, :, :]
                    V_n = V[-i-1, :, :]
            return A, G, V

    def _update_critic(self, G, V, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            critic_metrics = {}
            for agent_id in self.ids:
                critic_loss, critic_metrics[agent_id] = self._update_critic(G[agent_id], V[agent_id],
                                                                            agent_id=agent_id,
                                                                            frozen=frozen)
                total_loss += critic_loss
            return total_loss, critic_metrics
        else:
            if frozen:
                with torch.no_grad():
                    critic_loss = torch.nn.functional.mse_loss(G, V)
            else:
                critic_loss = torch.nn.functional.mse_loss(G, V)
                if self.clip_grad.get("critic"):
                    torch.nn.utils.clip_grad_norm_(self.critic_nets[agent_id].parameters(), self.clip_grad["critic"])
            critic_metrics = {
                "critic_loss": critic_loss.item()
                }
            return critic_loss, critic_metrics

    def _update_actor(self, A, lp_chosen, entropy,
                      weights=1.0, agent_id=None, frozen=False):
        if agent_id is None:
            total_loss = 0
            actor_metrics = {}
            for agent_id in self.ids:
                actor_loss, actor_metrics[agent_id] = self._update_actor(A[agent_id],
                                                                         lp_chosen,
                                                                         entropy,
                                                                         weights,
                                                                         agent_id=agent_id,
                                                                         frozen=frozen)
                total_loss += actor_loss
            return total_loss, actor_metrics
        else:
            if frozen:
                with torch.no_grad():
                    actor_loss = - (lp_chosen[agent_id].mul(A)
                                    + self.ent_coef*entropy[agent_id]).mul(weights).mean()
            else:
                actor_loss = - (lp_chosen[agent_id].mul(A)
                                + self.ent_coef*entropy[agent_id]).mul(weights).mean()
                if self.clip_grad.get("actor"):
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_nets[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy[agent_id].mean().item(),
                }
            return actor_loss, actor_metrics


    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)

    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:

            """ 这里 actor nets需要从agents里调用"""
            logits = self.actor_nets[agent_id](obs[agent_id])  
            """ 修改 """

            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            # Boost quantities according to DoE
            b_lp_chosen = lp_chosen * self.boost_lr(obs, agent_id=agent_id)
            b_entropy = entropy * self.boost_ent(obs, agent_id=agent_id)
            return b_lp_chosen, b_entropy

    """ This part is encoded in mac.forward """
    # def policy(self, obs, explore=True, agent_id=None, temp=1.0):
    #     with torch.no_grad():
    #         if agent_id is None:
    #             p = {agent_id: self.policy(obs[agent_id], explore=explore, agent_id=agent_id)
    #                  for agent_id in self.ids}
    #         else:
    #             obs_torch = torch.FloatTensor(obs[agent_id])
    #             logits = self.actor_nets[agent_id](obs_torch)
    #             if explore:
    #                 p = torch.distributions.categorical.Categorical(logits=logits/temp)
    #             else:
    #                 p = torch.distributions.categorical.Categorical(logits=logits)
    #     return p

    # def act(self, obs, explore=True, agent_id=None):
    #     if agent_id is None:
    #         return {agent_id: self.act(obs, explore=explore, agent_id=agent_id)
    #                 for agent_id in self.ids}
    #     else:
    #         policy = self.policy(obs,
    #                              explore=explore,
    #                              agent_id=agent_id,
    #                              temp=self.boost_temp(obs, agent_id))
    #         return policy.sample().unsqueeze(1).numpy()
    """ Delete """

    def boost_lr(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_lr(obs, agent_id) for agent_id in self.ids}
        else:
            boost = self.boost_lr_coef
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_lr*(boost + (1-boost)*self.is_doe(obs[agent_id], agent_id))

    def boost_temp(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_temp(obs, agent_id) for agent_id in self.ids}
        else:
            doe = self.is_doe(obs[agent_id], agent_id)
            return self.base_temp*torch.pow(self.boost_temp_coef, 1-doe)

    def boost_ent(self, obs, agent_id=None):
        if agent_id is None:
            return {self.boost_ent(obs, agent_id) for agent_id in self.ids}
        else:
            boost = self.boost_ent_coef
            return self.base_ent*(boost + (1-boost)*self.is_doe(obs[agent_id], agent_id))

    # @classmethod
    # def from_config(cls, cfg, env):
    #     zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
    #     name_id_mapping = {agent["zoo_name"]: agent["agent_id"]
    #                        for agent in cfg.agents}
    #     use_medoe = ("medoe" in cfg.keys()) and cfg.medoe.get("enable_medoe", True)
    #     agents = IA2C.load_from_zoo(
    #         cfg,
    #         name_id_mapping,
    #         env,
    #         zoo_path,
    #         )
    #     if not use_medoe:
    #         return agents
    #     agents = cls(
    #         agents,
    #         base_temp=cfg.medoe.base_vals.temp,
    #         base_ent=cfg.medoe.base_vals.ent,
    #         base_clip=cfg.medoe.base_vals.clip,
    #         #
    #         boost_temp=cfg.medoe.boost_vals.temp,
    #         boost_ent=cfg.medoe.boost_vals.ent,
    #         boost_clip=cfg.medoe.boost_vals.clip,
    #         #
    #         doe_classifier_cfg=cfg.medoe.classifier,
    #         )
    #     return agents
    
    # epymarl module, cuda, save critic and actor
    # what about save and load doe? embodied in doe py?

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):

        # Adopt from DoEIA2C
        # Count Early Stopping
        self.c_update += 1
        critic_frozen = self.c_update <= self.freeze_critic
        actor_frozen = self.c_update <= self.freeze_actor
        if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
            return {}
        
        # generate G, V, lp_chosen, entropy

        """ Align """
        _, agent_obs, actions = batch  #需要stack对齐数据格式


        As, Gs, Vs = self._compute_gae(agent_obs)


        """ 这部分可以考虑在 mac 里修改，返回的是pi分布和entropy """
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)


        # update
        self.critic_optimiser.zero_grad()
        self.agent_optimiser.zero_grad()
        critic_loss, critic_metrics = self._update_critic(Gs, Vs, frozen=critic_frozen)
        actor_loss, actor_metrics = self._update_actor(As, lp_chosens, entropies, frozen=actor_frozen)
        total_loss = critic_loss + actor_loss
        total_loss.backward()


        if not critic_frozen:
            self.total_critic_updates += 1
            for agent_id in self.ids:
                """ critic_nets 换 critic module """
                for i, p in enumerate(self.critic_nets[agent_id].parameters()):
                    critic_metrics[agent_id][f"critic_grad_{i}"] = p.grad.detach().norm().item()
        if not actor_frozen:
            self.total_actor_updates += 1
            for agent_id in self.ids:
                """ actor_nets 换 mac module """
                for i, p in enumerate(self.actor_nets[agent_id].parameters()):
                    actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()

        self.critic_optimiser.step()
        self.agent_optimiser.step()

        # do not return

        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        
        """ To Be Verified """
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_metrics["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_metrics[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

        # return train_metrics

        """ Old Epymarl Code """
        # Get the relevant quantities

        # rewards = batch["reward"][:, :-1]
        # actions = batch["actions"][:, :]
        # terminated = batch["terminated"][:, :-1].float()
        # mask = batch["filled"][:, :-1].float()
        # mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # actions = actions[:, :-1]
        # if self.args.standardise_rewards:
        #     self.rew_ms.update(rewards)
        #     rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)


        # mask = mask.repeat(1, 1, self.n_agents)

        # critic_mask = mask.clone()

        # old_mac_out = []
        # self.old_mac.init_hidden(batch.batch_size)
        # for t in range(batch.max_seq_length - 1):
        #     agent_outs = self.old_mac.forward(batch, t=t)
        #     old_mac_out.append(agent_outs)
        # old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        # old_pi = old_mac_out
        # old_pi[mask == 0] = 1.0

        # old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        # old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        # for k in range(self.args.epochs):
        #     mac_out = []
        #     self.mac.init_hidden(batch.batch_size)
        #     for t in range(batch.max_seq_length - 1):
        #         agent_outs = self.mac.forward(batch, t=t)
        #         mac_out.append(agent_outs)
        #     mac_out = th.stack(mac_out, dim=1)  # Concat over time

        #     pi = mac_out
        #     advantages, critic_train_stats = self.train_critic_sequential(self.critic, self.target_critic, batch, rewards,
        #                                                                   critic_mask)
        #     advantages = advantages.detach()
        #     # Calculate policy grad with mask

        #     pi[mask == 0] = 1.0

        #     pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
        #     log_pi_taken = th.log(pi_taken + 1e-10)

        #     ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
        #     surr1 = ratios * advantages
        #     surr2 = th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip) * advantages

        #     entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
        #     pg_loss = -((th.min(surr1, surr2) + self.args.entropy_coef * entropy) * mask).sum() / mask.sum()

        #     # Optimise agents
        #     self.agent_optimiser.zero_grad()
        #     pg_loss.backward()
        #     grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        #     self.agent_optimiser.step()

        # self.old_mac.load_state(self.mac)

        # self.critic_training_steps += 1
        # if self.args.target_update_interval_or_tau > 1 and (
        #         self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
        #     self._update_targets_hard()
        #     self.last_target_update_step = self.critic_training_steps
        # elif self.args.target_update_interval_or_tau <= 1.0:
        #     self._update_targets_soft(self.args.target_update_interval_or_tau)

        # if t_env - self.log_stats_t >= self.args.learner_log_interval:
        #     ts_logged = len(critic_train_stats["critic_loss"])
        #     for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
        #         self.logger.log_stat(key, sum(critic_train_stats[key]) / ts_logged, t_env)

        #     self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
        #     self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
        #     self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
        #     self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
        #     self.log_stats_t = t_env

    # def train_critic_sequential(self, critic, target_critic, batch, rewards, mask):
    #     # Optimise critic
    #     with th.no_grad():
    #         target_vals = target_critic(batch)
    #         target_vals = target_vals.squeeze(3)

    #     if self.args.standardise_returns:
    #         target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

    #     target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
    #     if self.args.standardise_returns:
    #         self.ret_ms.update(target_returns)
    #         target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

    #     running_log = {
    #         "critic_loss": [],
    #         "critic_grad_norm": [],
    #         "td_error_abs": [],
    #         "target_mean": [],
    #         "q_taken_mean": [],
    #     }

    #     v = critic(batch)[:, :-1].squeeze(3)
    #     td_error = (target_returns.detach() - v)
    #     masked_td_error = td_error * mask
    #     loss = (masked_td_error ** 2).sum() / mask.sum()

    #     self.critic_optimiser.zero_grad()
    #     loss.backward()
    #     grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
    #     self.critic_optimiser.step()

    #     running_log["critic_loss"].append(loss.item())
    #     running_log["critic_grad_norm"].append(grad_norm.item())
    #     mask_elems = mask.sum().item()
    #     running_log["td_error_abs"].append((masked_td_error.abs().sum().item() / mask_elems))
    #     running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
    #     running_log["target_mean"].append((target_returns * mask).sum().item() / mask_elems)

    #     return masked_td_error, running_log

    # def nstep_returns(self, rewards, mask, values, nsteps):
    #     nstep_values = th.zeros_like(values[:, :-1])
    #     for t_start in range(rewards.size(1)):
    #         nstep_return_t = th.zeros_like(values[:, 0])
    #         for step in range(nsteps + 1):
    #             t = t_start + step
    #             if t >= rewards.size(1):
    #                 break
    #             elif step == nsteps:
    #                 nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
    #             elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
    #                 nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
    #                 nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
    #             else:
    #                 nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
    #         nstep_values[:, t_start, :] = nstep_return_t
    #     return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        torch.save(self.critic.state_dict(), "{}/critic.th".format(path))
        torch.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        torch.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            torch.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            torch.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
