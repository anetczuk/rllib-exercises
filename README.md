# Exercises with *RLlib*

Exercises with *RLlib* -- an open-source library for reinforcement learning.

Exercises are done using *RLlib* version 1.6.0.


## Table of content

 1. [Project configuration](#project_configuration)
 6. [Solutions](#solutions)
     1. [acrobot](#solution_acrobot)
 7. [PyBrain limitations](#pybrain_limitations)
 8. [Alternative machine learning libraries](#alternative_libs)
 9. [References](#references)


## <a name="project_configuration"></a>Project configuration

Project's configuration can be easily done by execution of `tools/installvenv.sh` script. Script prepares virtual environemtn and installs necessary requirements (`requirements.txt`) inside `venv` subdirectory.

Starting environment can be done by execution of script `venv/activatevenv.sh`. 


## <a name="solutions"></a>Solutions

### <a name="solution_acrobot"></a>cart pole

[![CartPole-v1 PG](doc/solution/classic/cartpole-v1/cartpole-v1-pg-small.png "CartPole-v1 PG")](doc/solution/classic/cartpole-v1/cartpole-v1-pg.png)
[![CartPole-v1 PG](doc/solution/classic/cartpole-v1/cartpole-v1-pg-small.gif "CartPole-v1 PG")](doc/solution/classic/cartpole-v1/cartpole-v1-pg.mp4)

Meta parameters:
- algorithm: *Policy Gradients (PG)*
- model: neural network with two fully connected layers of size 64 and 32 respectively
- learning rate: 0.0006
- training steps: 658
- seed: 0xB36EF169

Algorithm took 0:03:15 to train.

Training input config: [input config](doc/solution/classic/cartpole-v1/cartpole-v1-pg.cfg)

Trained agent can be found [here](doc/solution/classic/cartpole-v1/cartpole-v1-pg-agent.zip)

Run command: `./src/rllibexercises/classic/cartpole_pg.py --seed=0xB36EF169`


## References:

- RLlib master (https://docs.ray.io/en/master/rllib/index.html)
- RLlib stable (https://docs.ray.io/en/stable/rllib.html)
- RLlib source code (https://github.com/ray-project/ray/tree/master/rllib)
- Gym environments (https://gym.openai.com/envs/)
- Review of popular reinforcement learning libraries (https://neptune.ai/blog/the-best-tools-for-reinforcement-learning-in-python)
