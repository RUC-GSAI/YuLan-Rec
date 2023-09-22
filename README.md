<div align=center>
<h1>RecAgent: A Novel Simulation Paradigm for Recommender Systems</h1>
<a href="https://pypi.org/project/recbole/">
        <img src="https://img.shields.io/pypi/v/recbole" alt="PyPi Latest Release">
    </a>
    <a href="https://anaconda.org/aibox/recbole">
        <img src="https://anaconda.org/aibox/recbole/badges/version.svg" alt="Conda Latest Release">
    </a>
    <img src="https://img.shields.io/badge/License-MIT-blue" alt="license">
    <img src="https://img.shields.io/github/stars/RUC-GSAI/YuLan-RecAgent" alt="license">

</div>

RecAgent is an LLM-based recommender simulator, which is expected to open a novel simulation based study paradigm in the recommendation domain. RecAgent is composed of two modules: (1) the user module and (2) the recommender module. The user module aims to simulate the real users, which can browse the recommendation website, communicate with the other users and broadcast messages on the social media. The recommender module is designed to provide the recommendation list to the users, and one can design different models to implement the recommender module. Each user maintains an individual memory, which can be automatically updated by the user behaviors. Different users take actions based on LLMs by retrieving their own memories, All the users can freely evolve in the simulator. RecAgent is highly flexible and customizable, where one can design different recommendation scenarios and algorithms. You can get the introduction video of RecAgent through the [Baidu Netdisk](https://pan.baidu.com/s/1ch8sinzFBi9XlEqp4xiYXQ?pwd=xsmq) or [Google Drive](https://drive.google.com/file/d/1IgsoWUipgZvnBnPSOUtbsZkm5yoyJskt/view?usp=sharing).

<p align="center">
  <img src="asset/img/framework.png" alt="RecAgnet Framework" width="100%">
  <br>
  <b>Figure 1</b>: RecAgent Framework
</p>


## 🔥 News
- [9/18/2023] The RecAgent `v2.0` is released on arXiv with following updates: 
  - <details>

    <summary>Latest Updates</summary>

    -  ****
        - Added 31 new works until 9/1/2023, to enrich the literature support.

    -  **📊 New Figures**
        - **Figure 3:** Added a new chart illustrating the differences and similarities between various planning approaches. This helps in gaining a clearer understanding of the comparisons between different planning methods.
        ![single-path and multi-path reasoning](assets/planning.png)
        - **Figure 4:** Introduced a new chart that describes the evolutionary path of model capability acquisition from the "Machine Learning era" to the "Large Language Model era" and then to the "Agent era." Specifically, a new concept, "mechanism engineering," has been introduced, which, along with "parameter learning" and "prompt engineering," forms part of this evolutionary path.
        ![Capabilities Acquisition](assets/capability.png)

    -  **🔍 Optimized Classification System**
        - Made appropriate modifications to the existing classification system to make it more logical and organized. This will assist in better organizing and presenting the content of the document.


  </details>

- [6/5/2023] The RecAgent `v1.0` is released on arXiv: [When Large Language Model based Agent Meets User Behavior Analysis: A Novel User Simulation Paradigm](https://arxiv.org/abs/2308.11432)


## Table of Contents


- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Example](#1-example)
  - [2. Website Demo](#2-website-demo)
- [Data](#data)
- [To-Do](#to-do)
- [Maintainers](#maintainers)
- [License](#license)
- [Citation](#citation)

## 💡 Features

- **Novel User Simulation Paradigm:** RecAgent is a novel user simulation paradigm that combines large language models (LLMs) with user behavior analysis. It is designed to simulate the behavior of real users in a recommender system. It also replicates users' social interactions, including chatting and posting. RecAgent can be used to evaluate the performance of recommender systems and to generate user behavior data for training and testing. 

- **Human-like Memory Mechanism:** RecAgent's memory mechanism mimics human cognitive processes, divided into sensory memory, short-term memory, and long-term memory. Through efficient compression and scoring of observations, it ensures the relevance and timeliness of information. By combining relevant memory with high-level insights, it achieves more realistic and consistent simulation of user behaviors. 

- **Large Scale Multi-Agent Simulation:** RecAgent is designed for robust, large-scale multi-agent parallel simulations. Seamlessly integrated with multiple OpenAI APIs, it support the capacity to simultaneously engage up to **1000** agents in its simulations. 

- **Multi-dimensional Evaluation:** We conducted an evaluation from two perspectives, that of the agent and that of the system. From the perspective of the agent, we assessed the effectiveness of various types of memory and evaluated whether the agent could access information-rich and relevant memories based on different memory structures. From the system's perspective, we focused on the reliability of the user behavior generated by our system and assessed the efficiency of the simulation.

- **High Extensibility:** RecAgent is designed to support a variety of recommendation algorithms and is equally compatible with multiple LLMs. Both aspects come with interfaces that are easy to customize.

- **Human-in-the-loop Simulation:** Individuals can create their own Agent to role-play and participate in the entire simulation process. Real people can take the same actions as other Agents, such as watching recommended movies, chatting with other Agents, or posting publicly.

- **Flexible Intervention and Control:** RecAgent provides diverse means to intervene with Agents and monitor the results. An Agent's profile can be freely modified. Moreover, people can intervene with Agents through dialogue. RecAgent also supports saving checkpoints, facilitating retrospective analysis and counterfactual studies.


## Installation

To install RecAgent, one can follow the following steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/RUC-GSAI/YuLan-Rec.git
    ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```
   Using `pip` to install `faiss` may report an error. One solution is to use `conda install faiss-cpu -c pytorch`.
3. Set your OpenAI API key in the `config/config.yaml` file.

## Usage

To start using RecAgent, follow these steps:

Configure the simulation parameters in the `config/config.yaml` file.

```yaml
# path for item data
item_path: data/item.csv
# path for user data
user_path: data/user_1000.csv
# path for relationship data
relationship_path: data/relationship_1000.csv
# path for save interaction records
interaction_path: data/interaction.csv
# directory name for faiss index
index_name: data/faiss_index
# simulator directory name for saving and restoring
simulator_dir: data/simulator
# simulator restoring name
simulator_restore_file_name: 
# recommender system model
rec_model: Random
# number of epochs
epoch: 30
# number of agents, which cannot exceed the number of user in user.csv
agent_num: 10
# number of items to be recommended in one page
page_size: 5
# temperature for LLM
temperature: 0.3
# maximum number of tokens for LLM
max_token: 1500
# execution mode, serial or parallel
execution_mode: serial
# time interval for action of agents. The form is a number + a single letter, and the letter may be s, m, h, d, M, y
interval: 5h
# name of LLM model
llm: gpt-3.5
# number of max retries for LLM API
max_retries: 100
# verbose mode
verbose: True
# threshold for agent num
active_agent_threshold: 100
# method for choose agent to be active, random, sample, or marginal
active_method: random
# propability for agent to be active
active_prob: 1
# memory for recagent, recagent or none
recagent_memory: recagent
# whether to add role play
play_role: False
# list of api keys for LLM API
api_keys:
  - xxxxx
  - xxxxx
```

The first `agent_num` users in `user.csv` will be loaded. Note that the `agent_num` in `config.yaml` cannot exceed the total number of users in `user.csv`.

### 1. Example

Run the simulation script:

```shell
python -u simulator.py --config_file config/config.yaml --output_file messages.json --log_file simulation.log
```

`config_file` is the path of the configuration file. `output_file` is the path of the output file, which is a JSON file containing all messages generated during the simulation. `log_file` set the path of the log file.

We will obtain some output like:
```
INFO:1422158:os.getpid()=1422158
INFO:1422158:
active_agent_threshold: 100
active_method: random
active_prob: 1
agent_num: 10
api_keys: ['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']
epoch: 30
execution_mode: serial
index_name: data/faiss_index
interaction_path: data/interaction.csv
interval: 5h
item_path: data/item.csv
llm: gpt-3.5
log_file: test.log
log_name: 1422158
max_retries: 100
max_token: 1500
output_file: output/message/test.json
page_size: 5
play_role: False
rec_model: Random
recagent_memory: recagent
relationship_path: data/relationship_1000.csv
simulator_dir: data/simulator_debug
simulator_restore_file_name: None
temperature: 0.3
user_path: data/user_1000.csv
verbose: True
Load faiss db from local

  0%|          | 0/10 [00:00<?, ?it/s]
100%|██████████| 10/10 [00:00<00:00, 3275.52it/s]
INFO:1422158:Simulator loaded.
INFO:82754:Round 1
  0%|                  | 0/10 [00:00<?, ?it/s]
INFO:1422158:Sarah Miller enters the recommender system.
INFO:1422158:Sarah Miller is recommended ['<Casper>;;The movie <Casper > is about a friendly ghost named Casper who lives in a haunted mansion with his three mischievous uncles.', '<The Goonies>;;<The Goonies> is a 1985 adventure-comedy film directed by Richard Donner and produced by Steven Spielberg.', '<One False Move>;;<One False Move> is a crime thriller movie released in 1991.', '<What About Bob?>;;<What About Bob?> is a comedy film about a man named Bob Wiley (played by Bill Murray) who has multiple phobias and anxiety disorders.', '<Phantasm III: Lord of the Dead>;;<Phantasm III: Lord of the Dead> is a horror movie that follows the story of Mike, who is on a mission to stop the Tall Man, a supernatural entity who is responsible for the death of his family.'].
INFO:1422158:Sarah Miller watches ['Casper']
```


### 2. Website Demo

We also have a local website demo.
Run the website demo script:

```shell
python -u run_demo.py --config_file config/config.yaml --output_file messages.json --log_file simulation.log
```
Then you can visit the demo at `http://127.0.0.1:7860`.
<p align="center">
  <img src="asset/img/interface.GIF" alt="RecAgnet Interface" width="100%">
  <br>
  <b>Figure 2</b>: RecAgent Interface
</p>

Click the *Play* button to start the simulation. The simulation will run for `epoch` rounds. You can click the *Reset* button to reset the simulation. Currently, the demo only supports the serial running model.

## Data

The item data used in the simulation is from the [MovieLens-1M](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset).
User profiles and relationships between users are fabricated for simulation purposes. We provide the FAISS index of the item data in directory `faiss_index`.

## To-Do

- [ ] Add documentation
- [x] ~~Large-scale user simulation~~
- [x] ~~Combining Real and Virtual Data~~
- [x] ~~Add support for various LLM~~
- [ ] Support for more recommendation algorithms
- [x] ~~Support for various LLM~~
- [ ] ...


## Maintainers
   
<div>
    <a href="https://github.com/Paitesanshi">@Lei Wang</a>
    <a href="https://github.com/JingsenZhang">@Jingsen Zhang</a>
    <a href="https://github.com/Paitesanshi">@Lei Wang</a>
    <a href="https://github.com/JingsenZhang">@Jingsen Zhang</a>
    <a href="https://github.com/Paitesanshi">@Lei Wang</a>
    <a href="https://github.com/JingsenZhang">@Jingsen Zhang</a>
    <a href="https://github.com/Paitesanshi">@Lei Wang</a>
    <a href="https://github.com/JingsenZhang">@Jingsen Zhang</a>

</div>



## License

RecAgent uses [MIT License](./LICENSE). All data and code in this project can only be used for academic purposes.

## Citation
Please cite the following paper as the reference if you use our code. [![Paper](https://img.shields.io/badge/arxiv-PDF-red)](https://arxiv.org/abs/2306.02552.pdf)

```
@misc{wang2023large,
      title={When Large Language Model based Agent Meets User Behavior Analysis: A Novel User Simulation Paradigm}, 
      author={Lei Wang and Jingsen Zhang and Hao Yang and Zhiyuan Chen and Jiakai Tang and Zeyu Zhang and Xu Chen and Yankai Lin and Ruihua Song and Wayne Xin Zhao and Jun Xu and Zhicheng Dou and Jun Wang and Ji-Rong Wen},
      year={2023},
      eprint={2306.02552},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```