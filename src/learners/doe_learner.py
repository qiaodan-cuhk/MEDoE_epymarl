# Modified from fst doe_ac.py
# 这个是最开始对齐版本，迁移fst到epymarl框架，暂时不用
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

        # Actor Network
        self.mac = mac
        self.old_mac = copy.deepcopy(mac)
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        # Critic Network
        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        # Target Critic Soft Update
        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents, ), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

        # Copy from doe agents
        # 暂时不考虑并行化训练，不设置，需要的话从 runner.env 得到信息
        # if hasattr(env, "num_vec_envs"):
        #     n_train_envs = env.num_vec_envs
        # elif hasattr(env, "vec_envs"):
        #     n_train_envs = len(env.vec_envs)
        # else:
        #     n_train_envs = 1

        # self.n_train_envs = args.n_train_envs     # n_train_env = 1
        # self.freeze_critic = args.freeze_critic/self.n_train_envs
        # self.freeze_actor = args.freeze_actor/self.n_train_envs

        # Initialise update counters
        # self.c_update = 0
        # self.p_update = self.n_steps
        # self.total_critic_updates = 0
        # self.total_actor_updates = 0

        # adopt from doe agents
        self.ent_coef = 1.0 # needed to override the ent_coef called elsewhere

        self.base_temp = self.args.get("base_temp", 1.0)
        self.base_lr = self.args.get("base_lr", 1.0)
        self.base_ent = self.args.get("base_ent", 1.0)

        self.boost_temp_coef = self.args.get("boost_temp", 1.0)
        self.boost_lr_coef = self.args.get("boost_lr", 1.0)
        self.boost_ent_coef = self.args.get("boost_ent", 1.0)


        """ ids 是一个可迭代list，装载每个agent的名字，用于指定 agent """
        # self.ids = Iterable[AgentID]

        # mlp/joint_mlp

        self.doe_classifier = doe_classifier_config_loader(
                cfg=self.args.get("doe_classifier_cfg"),
                ids=self.ids
                )
        
        # self.doe_classifier = doe_resigtry[args.doe_type].from_config(cfg=self.args.get("doe_classifier_cfg"), ids=self.ids)


    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # t_env 是记录环境运行步骤的，输入到learner.train中，一般不用 t 。 不参与训练，只用于logger

        # DoE 相对于 IA2C 的改动其实只在两个地方，第一是策略采样时多了tmp，另外是计算qty多了gain以后的entropy和sotfmax
        # update critic 不改，uodate actor 里entropy和clamp的ratio变为doe ratio
        # PPO 里 qty 也没变
        # update里多了个doe判断，输入给update actor用于计算 boosted coef
        # IA2C里 qty变了，返回的lp和entropy多了gain，feed in actor直接作为loss计算
        # PPO 的 gain 放在actor update，IA2C 的 entropy放在 qty


        # Adopt from DoEIA2C

        # Count Early Stopping
        # self.c_update += 1
        # critic_frozen = self.c_update <= self.freeze_critic
        # actor_frozen = self.c_update <= self.freeze_actor
        # if (critic_frozen and actor_frozen) or self.c_update % self.p_update != 0:
        #     return {}
        
        

        # _, agent_obs, actions = batch  #需要stack对齐数据格式
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / torch.sqrt(self.rew_ms.var)

        """ Align """
        agent_obs = batch["obs"][:, :-1]

        

        # generate G, V, lp_chosen, entropy
        # agent_outs = self.mac.forward(batch, t=t) 得到policy分布，其中 build_inputs 中取出 batch["obs"] 并处理
        # 要考虑是否使用 mac 的形式
        # agent_inputs = self._build_inputs(ep_batch, t)
        # agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        As, Gs, Vs = self._compute_gae(agent_obs, batch)


        """ 这部分可以考虑在 mac 里修改，返回的是pi分布和entropy,这里用到了doe的增益 """
        lp_chosens, entropies = self._compute_policy_qty(agent_obs, actions)


        """ Old Epymarl Code Start """
        # Get the relevant quantities

        
        mask = mask.repeat(1, 1, self.n_agents)   
        """ old Epymarl code Finished """ 

        # update
        self.critic_optimiser.zero_grad()
        self.agent_optimiser.zero_grad()
        
        # 这里frozen都设置为False
        critic_loss, critic_metrics = self._update_critic(Gs, Vs, frozen=False)
        actor_loss, actor_metrics = self._update_actor(As, lp_chosens, entropies, frozen=False)

        total_loss = critic_loss + actor_loss
        total_loss.backward()
        # critic_loss.backward()
        self.critic_optimiser.step()

        # actor_loss.backward()
        self.agent_optimiser.step()

        # Early Stopping for Critic and Actor
        # if not critic_frozen:
        #     self.total_critic_updates += 1
        #     for agent_id in self.ids:
        #         """ critic_nets 换 critic module """
        #         for i, p in enumerate(self.critic[agent_id].parameters()):
        #             critic_metrics[agent_id][f"critic_grad_{i}"] = p.grad.detach().norm().item()
        # if not actor_frozen:
        #     self.total_actor_updates += 1
        #     for agent_id in self.ids:
        #         """ actor_nets 换 mac module """
        #         for i, p in enumerate(self.mac[agent_id].parameters()):
        #             actor_metrics[agent_id][f"actor_grad_{i}"] = p.grad.detach().norm().item()
    
        
        
        

        # Target Critic Soft Update
        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        

        # do not return

        train_metrics = {
            **critic_metrics,
            **actor_metrics
            }
        
        self.logger.log_stat(train_metrics)
        
        """ To Be Verified, Logging """
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_metrics["critic_loss"])
            for key in ["critic_loss", "critic_grad_norm", "td_error_abs", "q_taken_mean", "target_mean"]:
                self.logger.log_stat(key, sum(critic_metrics[key]) / ts_logged, t_env)

            self.logger.log_stat("advantage_mean", (advantages * mask).sum().item() / mask.sum().item(), t_env)
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat("pi_max", (pi.max(dim=-1)[0] * mask).sum().item() / mask.sum().item(), t_env)
            self.log_stats_t = t_env

    def is_doe(self, obs, agent_id=None):
        return self.doe_classifier.is_doe(obs, agent_id=agent_id)


    def _compute_gae(self, obs, batch, agent_id=None):
        if agent_id is None:
            As, Gs, Vs = zip(*(self._compute_gae(obs, batch, agent_id=agent_id)
                           for agent_id in self.ids))
            As = dict(zip(self.ids, As))
            Gs = dict(zip(self.ids, Gs))
            Vs = dict(zip(self.ids, Vs))
            return As, Gs, Vs
        else:
            # 这里要检查 critic 是list还是单个的
            V = self.critic[agent_id](obs[agent_id])
            with torch.no_grad():
                G = torch.zeros_like(V)
                A = torch.zeros_like(V)
                """ 检查batch数据格式是不是batch size * agent nums """
                V_n = self.critic[agent_id](batch["obs"][:, agent_id])

                G_n = V_n.clone()
                A_n = 0

                """ GAE 实现要重新检查 """
                # last_n -> batch
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
        
    def _compute_policy_qty(self, obs, acts, agent_id=None):
        if agent_id is None:
            lp_chosens, entropies = zip(*(self._compute_policy_qty(obs, acts, agent_id)
                                          for agent_id in self.ids))
            lp_chosens = dict(zip(self.ids, lp_chosens))
            entropies = dict(zip(self.ids, entropies))
            return lp_chosens, entropies
        else:

            """ 这里 actor nets需要从agents里调用"""
            logits = self.mac[agent_id](obs[agent_id])  
            """ 修改 """

            lp = torch.nn.functional.log_softmax(logits, dim=2)
            lp_chosen = lp.gather(2, acts[agent_id])
            entropy = -lp.exp().mul(lp).sum(dim=2).unsqueeze(2)
            # Boost quantities according to DoE
            b_lp_chosen = lp_chosen * self.boost_lr(obs, agent_id=agent_id)
            b_entropy = entropy * self.boost_ent(obs, agent_id=agent_id)
            return b_lp_chosen, b_entropy

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
                    torch.nn.utils.clip_grad_norm_(self.critic[agent_id].parameters(), self.clip_grad["critic"])
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
                    # actor_net 换 mac
                    torch.nn.utils.clip_grad_norm_(
                        self.mac[agent_id].parameters(), self.clip_grad["actor"]
                        )

            actor_metrics = {
                "actor_loss": actor_loss.item(),
                "entropy": entropy[agent_id].mean().item(),
                }
            return actor_loss, actor_metrics


    # DoE模块，调节超参数，分别是lr，采样动作的tmp，entropy系数，lr和ent都在qty里，temp没出现，可能在别的文件中

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
    
    # epymarl module, cuda, save critic and actor
    # what about save and load doe? embodied in doe py?

    


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
        self.critic.load_state_dict(torch.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            torch.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            torch.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
