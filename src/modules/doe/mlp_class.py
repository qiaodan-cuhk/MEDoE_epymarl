import os
import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from fst.utils.network import fc_network
from omegaconf import OmegaConf
import hydra

from utils import SimpleListDataset


# ids 是一个Iterable list，用于适配zoo管理的字符串命名agent，如 Alice Bob等，在epymarl框架中不需要
# name_id_mapping = OrderedDict(((cfg.zoo_mapping[agent_id], agent_id) for agent_id in ids))
# zoo_mapping:
#   alice: ${agents.medoe.zoo_alice}
#   bob: ${agents.medoe.zoo_bob}
#   carol: ${agents.medoe.zoo_carol}
#   dave: ${agents.medoe.zoo_dave}


class MLPClassifier:
    def __init__(self,
                 n_agents,
                 train_dataloader, 
                 test_dataloader,
                 network_arch,
                 role_list,
                 learning_rate=1e-2,
                 batch_size=256,
                 test_period=5,
                 obs_mask=None,
                 ):
        self.n_agents = n_agents
        self.mlps = [fc_network(network_arch) for _ in range(n_agents)]
        self.learning_rates = [learning_rate] * n_agents
        self.network_arch = network_arch
        self.obs_mask = obs_mask


        if train_dataloader is not None:

            self.trained_agent_id = 0 # 要改成for循环list

            self.train_data_loader = train_dataloader
            self.test_data_loader = test_dataloader
            self.results = self.train_mlp(
                    train_dataloader=train_dataloader,
                    test_dataloader=test_dataloader,
                    role_list=role_list,
                    test_period=test_period,
                    obs_mask=self.obs_mask,
                    # agent_id=self.trained_agent_id
                    )

    def train_mlp(
            self,
            train_dataloader,
            test_dataloader,
            role_list, # 这是个字典，key是每个agent的名字，返回的是 0/1 这样的标签，代表进攻和防守
            test_period=5,
            obs_mask=None):
        
        results = []

        for agent_id in range(self.n_agents):

            loss_function = torch.nn.BCEWithLogitsLoss()
            optim = Adam(self.mlps[agent_id].parameters(),
                         lr=self.learning_rates[agent_id],
                         eps=1e-8)

            train_results = []
            test_results = []

            # mask某一层，network_arch = [32, 256, 10] 代表网络结构
            if obs_mask is None:
                mask = 1
            else:
                mask = torch.zeros(self.network_arch[0])
                for i in obs_mask:
                    mask[i] = 1

            for batch_idx, (s, label) in enumerate(train_dataloader):
                predicted_label = self.mlps[agent_id](s*mask).flatten()
                egocentric_label = (label == role_list[agent_id]).float()
                # 这里label提供的是0-1代表defend-attack角色的经验，需要经过跟当前agent的role list进行对比，
                # 来判断这个状态是不是符合当前的角色，如果是，那么egocentric_label就是1，否则就是0
                # 所以这里loss_function是BCEWithLogitsLoss，目标是让agent根据状态判断是否符合自身角色
                loss = loss_function(predicted_label, egocentric_label)
                optim.zero_grad()
                loss.backward()
                optim.step()
                train_results.append(loss.item())
                test_loss = 0.0
                if batch_idx % test_period == 0:
                    with torch.no_grad():
                        for s_test, label_test in test_dataloader:
                            predicted_label_test = self.mlps[agent_id](s_test).flatten()
                            egocentric_label_test = (label_test == role_list[agent_id]).float()
                            test_loss += loss_function(predicted_label_test, egocentric_label_test).item()
                        test_results.append(test_loss/len(test_dataloader))
                        # batch是一个递增整数，代表epoch的轮数，多个batch——size的数据

            results.append({ 
                "agent_index": agent_id,
                "train": train_results,
                "test": test_results,
            })
        return results

    # 返回的是obs属于当前角色的0-1概率，需要经过sigmoid函数
    def is_doe(self, obs, agent_id=None):
        if agent_id is None:
            return [self.mlps[i](torch.Tensor(obs[i])).sigmoid() for i in range(self.n_agents)]
        else:
            return self.mlps[agent_id](torch.Tensor(obs)).sigmoid()

    # 返回的是mlp输出值，不需要经过sigmoid函数，变成0-1的概率
    def is_doe_logits(self, obs, agent_id=None):
        if agent_id is None:
            return [self.mlps[i](torch.Tensor(obs[i])) for i in range(self.n_agents)]
        else:
            return self.mlps[agent_id](torch.Tensor(obs))

    def update(self):
        ...
    
    def save(self, pathname):
        torch.save(self.mlps, pathname)

    @classmethod
    def from_zoo(cls,
                 name_id_mapping,
                 role_list,
                 zoo_path,
                 cfg,
                 ):
        
        agent_ids = list(name_id_mapping.values())
        # [0 1 2 3]

        # 这段是为了load buffer,需要在run.py中添加一个save buffer的接口
        exp_buffers = {}
        for actor_name, agent_id in name_id_mapping.items():
            # load config
            actor_cfg_path = osp.normpath(
                osp.join(zoo_path, "configs", "actors", f"{actor_name}.yaml")
                )
            actor_cfg = OmegaConf.load(actor_cfg_path)
            # load experience
            exp_path = osp.normpath(
                osp.join(zoo_path, "configs", "experience", f"{actor_cfg.experience}.yaml")
                )
            exp_cfg = OmegaConf.load(exp_path)
            exp_buffers.update({agent_id: torch.load(exp_cfg.path_to_experience)})

        # Classifier training params
        batch_size = cfg.get("batch_size", 256)
        test_fraction = cfg.get("test_fraction", 0.1)
        hidden_sizes = cfg.get("hidden_sizes", [128])
        learning_rate = cfg.get("lr", 1e-2)
        test_period = cfg.get("test_period", 5)
        obs_mask = cfg.get("obs_mask", None)

        # Load & process the data
        states = []
        labels = []
        with torch.no_grad():
            for agent_id in agent_ids:
                #state = torch.concat(exp_buffers[agent_id])
                state = exp_buffers[agent_id]
                label = torch.full((len(exp_buffers[agent_id]),), agent_id_to_label[agent_id])
                states.append(state)
                labels.append(label)
            states = torch.concat(states)
            labels = torch.concat(labels)
            dataset = SimpleListDataset(states, labels)
            train_size = int(test_fraction * len(dataset))
            test_size = len(dataset) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                        [train_size, test_size])
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        network_arch = [states[0].size().numel(), *hidden_sizes, 1]
        return cls(
            n_agents=len(agent_ids),
            train_dataloader=train_dataloader, 
            test_dataloader=test_dataloader,
            network_arch=network_arch,
            learning_rate=learning_rate,
            batch_size=batch_size,
            test_period=test_period,
            obs_mask=obs_mask,
            )

    @classmethod
    def from_config(cls, n_agents, cfg):
        if cfg.load_mode == "train":
            classifier = cls.from_config_train(n_agents, cfg)
            if cfg.get("save_classifier", False):
                classifier.save(cfg.save_pathname)
            return classifier
        elif cfg.load_mode == "load":
            return cls.from_config_load(n_agents, cfg)

    @classmethod
    def from_config_train(cls, n_agents, cfg):

        name_id_mapping = OrderedDict(((cfg.zoo_mapping[agent_id], agent_id) for agent_id in range(n_agents)))
        # 这个没用，待删除

        zoo_path = os.path.abspath(cfg.zoo_path)
        # zoo_path = hydra.utils.to_absolute_path(cfg.get("zoo_path", "zoo"))
        # 这个没用，待删除  
        
        role_ids = cfg.role_ids
        # (0, 'defence', ['alice', 'bob'])) 和 (1, ('attack', ['carol', 'dave']))

        mlp_cfg = cfg.mlp
        role_list = [0] * n_agents
        for label, (source, source_agent_ids) in enumerate(role_ids.items()):
            for agent_id in source_agent_ids:
                role_list[agent_id] = label

        return cls.from_zoo(name_id_mapping, role_list, zoo_path, mlp_cfg)

    @classmethod
    def from_config_load(cls, n_agents, cfg):
        classifier = cls(
            n_agents,
            train_dataloader=None, 
            test_dataloader=None,
            network_arch=None,
            role_list=None  # 假设这个参数是必需的，如果不是，可以移除
        )
        # 使用os.path来处理路径
        absolute_path = os.path.abspath(cfg.path_to_classifier)
        classifier.mlps = torch.load(absolute_path)
        return classifier

    # @classmethod
    # def load_mlp(cls, n_agents, pathname):
    #     classifier = cls(
    #              n_agents,
    #              train_dataloader=None, 
    #              test_dataloader=None,
    #              network_arch=None,
    #              )
    #     classifier.mlps = torch.load(hydra.utils.to_absolute_path(pathname))
    #     return classifier




