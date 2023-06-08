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


## Table of Contents

- [Table of Contents](#table-of-contents)
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

## Features

- Simulation of the interactions between the users and recommender system
- Simulation of the chatting behaviors between different users
- Simulation of the broadcasting behaviors of the users on the social media
- Customizable recommender algorithm
- Flexible data import and export mechanisms
- Parallel ChatGPT API support


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
user_path: data/user.csv
# path for relationship data
relationship_path: data/relationship.csv
# path for save interaction records
interaction_path: data/interaction.csv
# directory name for faiss index
index_name: data/faiss_index
# recommender system model
model: Random
# number of epochs
epoch: 1
# number of agents, which cannot exceed the number of user in user.csv
num_agents: 1
# number of items to be recommended in one page
page_size: 5
# temperature for LLM
temperature: 0.8
# maximum number of tokens for LLM
max_token: 1500
# execution mode, serial or parallel
execution_mode: serial
# list of api keys for LLM API
api_keys:
  - xxxxx
  - xxxxx
```

The first `num_agents` users in `user.csv` will be loaded. Note that the `num_agents` in `config.yaml` cannot exceed the total number of users in `user.csv`.

### 1. Example

Run the simulation script:

```shell
python -u simulator.py --config_file config/config.yaml --output_file messages.json --log_file simulation.log
```

`config_file` is the path of the configuration file. `output_file` is the path of the output file, which is a JSON file containing all messages generated during the simulation. `log_file` set the path of the log file.

We will obtain some output like:
```
api_keys: ['xxxxx', 'xxxxx']
epoch: 1
execution_mode: serial
index_name: faiss_index
interaction_path: data/interaction.csv
item_path: data/item.csv
log_name: 82754
max_token: 1500
model: Random
num_agents: 1
page_size: 5
relationship_path: data/relationship.csv
temperature: 0.8
user_path: data/user.csv
Load faiss db from local
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:20<00:00,  6.99s/it]
INFO:82754:Simulator loaded.
INFO:82754:Round 1
  0%|                                                                                                                                                                                     | 0/3 [00:00<?, ?it/s]INFO:82754:Tommie is going to recommender system.
INFO:82754:Tommie is recommended ['<Kicked in the Head (1997)>', '<Goldfinger (1964)>', '<Clockwork Orange, A (1971)>', '<Picnic at Hanging Rock (1975)>', '<Courage Under Fire (1996)>'].
INFO:82754:Tommie watched  <Clockwork Orange, A (1971)>; <Courage Under Fire (1996)>
INFO:82754:Tommie feels:<Clockwork Orange, A (1971)>: Wow, that was intense and thought-provoking I'm not sure if I completely understand it, but it definitely left an impression on me; <Courage Under Fire (1996)>: That was a gripping and emotional story I was on the edge of my seat and felt a real connection to the characters
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
- [ ] Large-scale user simulation
- [ ] Combining Real and Virtual Data
- [ ] Add support for multiple LLM
- [ ] Support for more recommendation algorithms
- [ ] ...






## Maintainers
   
<div>
    <a href="https://github.com/Paitesanshi">@Lei Wang</a>
    <a href="https://github.com/JingsenZhang">@Jingsen Zhang</a>
</div>



## License

RecAgent uses [MIT License](./LICENSE). All data and code in this project can only be used for academic purposes.

## Citation
Please cite the following paper as the reference if you use our code. [![Paper](https://img.shields.io/badge/arxiv-PDF-red)](https://arxiv.org/abs/2306.02552.pdf)

```
@misc{wang2023recagent,
      title={RecAgent: A Novel Simulation Paradigm for Recommender Systems}, 
      author={Lei Wang and Jingsen Zhang and Xu Chen and Yankai Lin and Ruihua Song and Wayne Xin Zhao and Ji-Rong Wen},
      year={2023},
      eprint={2306.02552},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```