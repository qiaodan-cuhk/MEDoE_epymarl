# LLM Subtask for GRF based on EPyMARL

This implementation aims for adapt MODoE into Google Research Football Environments and EPyMARL framework.

## Adopting MEDoE to Epymarl & GRF

Change Log
- Seperate fst/runner agents into learner and agents. 
- Add doe_a2c_learner in src/learners (agent.update), adopt doe coefficients as actor loss
- Add doe_classifier & LLM_doe in src/modules/doe
- Add doe_controller in src/controllers/mac, adopt doe coefficients as sample temperature
- Add gfootball in src/envs
- Add doe_ia2c.yaml in config/alg
- Zoo Agents are removed, doe_classifier experience are saved manually, code is in run.py/242-249, buffer will be saved in results/models/envs/map_name/buffer.pt
- MEDoE contains 2-stage training, first on sourse task, then on target task, the pretrained zoo agents are saved in [Edingburgh DataShare](https://datashare.ed.ac.uk/handle/10283/8778). Here we implemented this by learning from scratch & save models/experience in .py file, then load them in the medoe.py training.
- 移除 ids 姓名列表，在epymarl框架中不需要，改为用 [0 1 2 3] 表示4个agent
- 添加判断是source task预训练和target task DoE的参数，在yaml.args中，对应的是 args.use_doe = True
- 在 run.py 中，添加 args.if_doe 参数，在训练时，如果为True，则加载buffer并训练一个classifier，否则不加载
- 修改 doe_controller.py 和 doe_a2c_learner.py 中关于is_doe_classifier的代码，增加set_doe_classifier方法直接加载 self.doe_classifier = doe_classifier_config_loader(n_agents, cfg)，避免重复训练
- 移除 mlp_class.py 中 from zoo 的代码，omegaconf 的buffer加载方式，以及 agent_id_to_label 替换为role_list一个列表表示每个agent的角色“attack-defend”

ToDo
- GRF 数据结构需要测试
- buffer存储的数据结构需要测试，并与doe classifier对齐
- 先在SMAC测试epymarl的 source 和 target 训练

代码结构
- run.py 是整个训练流程
- mlp_class.py 是分类器，加载buffer数据和角色配置，训练一个 obs-> 是否属于自己角色的分类器
- doe_controller.py 是控制器 actor，初始化is_doe时加载分类器，并训练分类器，使用分类器结果来控制动作的采样温度
- doe_a2c_learner.py 是学习器，初始化is_doe时加载分类器，在训练中使用分类器结果作为 lr 和 entropy 的增益系数，调节critic loss
- 在mlp_class中，如果 train_dataloader 不为空，则使用 buffer 数据训练分类器，否则不训练 mlps


Notice:
- Epymarl has been updated to [Gymnasium](https://github.com/uoe-agents/epymarl) version in June 2024. For simplicity, we adopt old version Epymarl with Gym suits.
- Some tricks dependencies for Epymarl: gym==0.21.0 & setuptools==65.5.0
- GRF installation is tricky, try following codes, see [GRF](https://github.com/google-research/football) and [MARLlib_Env](https://marllib.readthedocs.io/en/latest/handbook/env.html#google-research-football)


```
sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip 
python3 -m pip install --upgrade pip setuptools psutil wheel
```

- (Tmp) For Chinese reader, here is a quickstart for Epymarl structures. [mathpretty](https://mathpretty.com/16924.html)

## Original Readme from Epymarl
EPyMARL is  an extension of [PyMARL](https://github.com/oxwhirl/pymarl), and includes
- Additional algorithms (IA2C, IPPO, MADDPG, MAA2C and MAPPO)
- Support for [Gym](https://github.com/openai/gym) environments (on top of the existing SMAC support)
- Option for no-parameter sharing between agents (original PyMARL only allowed for parameter sharing)
- Flexibility with extra implementation details (e.g. hard/soft updates, reward standarization, and more)
- Consistency of implementations between different algorithms (fair comparisons)

See our blog post here: https://agents.inf.ed.ac.uk/blog/epymarl/


## Update as of *15th July 2023*!
We have released our _Pareto Actor-Critic_ algorithm as part of the E-PyMARL source code. 

Find the preprint here: https://arxiv.org/abs/2209.14344

Pareto-AC (Pareto-AC), is an actor-critic algorithm that utilises a simple principle of no-conflict games (and, in turn, cooperative games with identical rewards): each agent can assume the others will choose actions that will lead to a Pareto-optimal equilibrium.
Pareto-AC works especially well in environments with multiple suboptimal equilibria (a problem is also known as relative over-generalisation). We have seen impressive results in a diverse set of multi-agent games with suboptimal equilibria, including the matrix games of the MARL benchmark, but also LBF variations with high penalties.

To run Pareto-AC in an environment, for example the Penalty game, you can run:
```
python main.py --config=pac_ns --env-config=gymma with env_args.time_limit=1 env_args.key=matrixgames:penalty-100-nostate-v0
```

# Table of Contents
- [Extended Python MARL framework - EPyMARL](#extended-python-marl-framework---epymarl)
- [Table of Contents](#table-of-contents)
- [Installation & Run instructions](#installation--run-instructions)
  - [Installing LBF, RWARE, and MPE](#installing-lbf-rware-and-mpe)
  - [Using A Custom Gym Environment](#using-a-custom-gym-environment)
- [Run an experiment on a Gym environment](#run-an-experiment-on-a-gym-environment)
- [Run a hyperparameter search](#run-a-hyperparameter-search)
- [Saving and loading learnt models](#saving-and-loading-learnt-models)
  - [Saving models](#saving-models)
  - [Loading models](#loading-models)
- [Citing PyMARL and EPyMARL](#citing-pymarl-and-epymarl)
- [License](#license)

# Installation & Run instructions

For information on installing and using this codebase with SMAC, we suggest visiting and reading the original [PyMARL](https://github.com/oxwhirl/pymarl) README. Here, we maintain information on using the extra features EPyMARL offers.
To install the codebase, clone this repo and install the `requirements.txt`.  

## Installing LBF, RWARE, and MPE

In [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869) we introduce and benchmark algorithms in Level-Based Foraging, Multi-Robot Warehouse and Multi-agent Particle environments.
To install these please visit:
- [Level Based Foraging](https://github.com/uoe-agents/lb-foraging) or install with `pip install lbforaging`
- [Multi-Robot Warehouse](https://github.com/uoe-agents/robotic-warehouse) or install with `pip install rware`
- [Our fork of MPE](https://github.com/semitable/multiagent-particle-envs), clone it and install it with `pip install -e .`

Example of using LBF:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
```
Example of using RWARE:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=500 env_args.key="rware:rware-tiny-2ag-v1"
```

For MPE, our fork is needed. Essentially all it does (other than fixing some gym compatibility issues) is i) registering the environments with the gym interface when imported as a package and ii) correctly seeding the environments iii) makes the action space compatible with Gym (I think MPE originally does a weird one-hot encoding of the actions).

The environments names in MPE are:
```
...
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
...
```
Therefore, after installing them you can run it using:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleSpeakerListener-v0"
```

The pretrained agents are included in this repo [here](https://github.com/uoe-agents/epymarl/tree/main/src/pretrained). You can use them with:
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleAdversary-v0" env_args.pretrained_wrapper="PretrainedAdversary"
```
and
```sh
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=25 env_args.key="mpe:SimpleTag-v0" env_args.pretrained_wrapper="PretrainedTag"
```


## Using A Custom Gym Environment

EPyMARL supports environments that have been registered with Gym. 
The only difference with the Gym framework would be that the returned rewards should be a tuple (one reward for each agent). In this cooperative framework we sum these rewards together.

Environments that are supported out of the box are the ones that are registered in Gym automatically. Examples are: [Level-Based Foraging](https://github.com/semitable/lb-foraging) and [RWARE](https://github.com/semitable/robotic-warehouse). 

To register a custom environment with Gym, use the template below (taken from Level-Based Foraging).
```python
from gym.envs.registration import registry, register, make, spec
register(
  id="Foraging-8x8-2p-3f-v1",                     # Environment ID.
  entry_point="lbforaging.foraging:ForagingEnv",  # The entry point for the environment class
  kwargs={
            ...                                   # Arguments that go to ForagingEnv's __init__ function.
        },
    )
```

# Run an experiment on a Gym environment

```shell
python3 src/main.py --config=qmix --env-config=gymma with env_args.time_limit=50 env_args.key="lbforaging:Foraging-8x8-2p-3f-v1"
```
 In the above command `--env-config=gymma` (in constrast to `sc2` will use a Gym compatible wrapper). `env_args.time_limit=50` sets the maximum episode length to 50 and `env_args.key="..."` provides the Gym's environment ID. In the ID, the `lbforaging:` part is the module name (i.e. `import lbforaging` will run automatically).


The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.

# Run a hyperparameter search

We include a script named `search.py` which reads a search configuration file (e.g. the included `search.config.example.yaml`) and runs a hyperparameter search in one or more tasks. The script can be run using
```shell
python search.py run --config=search.config.example.yaml --seeds 5 locally
```
In a cluster environment where one run should go to a single process, it can also be called in a batch script like:
```shell
python search.py run --config=search.config.example.yaml --seeds 5 single 1
```
where the 1 is an index to the particular hyperparameter configuration and can take values from 1 to the number of different combinations.

# Saving and loading learnt models

## Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

## Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

# Citing EPyMARL and PyMARL

The Extended PyMARL (EPyMARL) codebase was used in [Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](https://arxiv.org/abs/2006.07869).

*Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, & Stefano V. Albrecht. Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks, Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021*

In BibTeX format:

```tex
@inproceedings{papoudakis2021benchmarking,
   title={Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks},
   author={Georgios Papoudakis and Filippos Christianos and Lukas Schäfer and Stefano V. Albrecht},
   booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS)},
   year={2021},
   url = {http://arxiv.org/abs/2006.07869},
   openreview = {https://openreview.net/forum?id=cIrPX-Sn5n},
   code = {https://github.com/uoe-agents/epymarl},
}
```

If you use the original PyMARL in your research, please cite the [SMAC paper](https://arxiv.org/abs/1902.04043).

*M. Samvelyan, T. Rashid, C. Schroeder de Witt, G. Farquhar, N. Nardelli, T.G.J. Rudner, C.-M. Hung, P.H.S. Torr, J. Foerster, S. Whiteson. The StarCraft Multi-Agent Challenge, CoRR abs/1902.04043, 2019.*

In BibTeX format:

```tex
@article{samvelyan19smac,
  title = {{The} {StarCraft} {Multi}-{Agent} {Challenge}},
  author = {Mikayel Samvelyan and Tabish Rashid and Christian Schroeder de Witt and Gregory Farquhar and Nantas Nardelli and Tim G. J. Rudner and Chia-Man Hung and Philiph H. S. Torr and Jakob Foerster and Shimon Whiteson},
  journal = {CoRR},
  volume = {abs/1902.04043},
  year = {2019},
}
```

# License
All the source code that has been taken from the PyMARL repository was licensed (and remains so) under the Apache License v2.0 (included in `LICENSE` file).
Any new code is also licensed under the Apache License v2.0
