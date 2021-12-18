#!/usr/bin/env python3

try:
    ## following import success only when file is directly executed from command line
    ## otherwise will throw exception when executing as parameter for "python -m"
    # pylint: disable=W0611
    import __init__
except ImportError as error:
    ## when import fails then it means that the script was executed indirectly
    ## in this case __init__ is already loaded
    pass

import os
from datetime import datetime
import argparse

import matplotlib.pyplot as pyplot
# import plot

from rllibexercises import trainer


## disable import warning
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"


ENV_NAME = "MountainCar-v0"


def int_hex_dec(x):
    return int(x, 0)


def main():
    parser = argparse.ArgumentParser(description=ENV_NAME+' solution')
    parser.add_argument( '--seed', action="store", type=int_hex_dec, default=None, help='RNG seed (dec or hex)' )
    
    args = parser.parse_args()
    
    ## preparing and training
    start_time = datetime.now()

# "CQL"                  ## AttributeError: 'Categorical' object has no attribute 'sample_logp'
# "DDPG"                 ## ray.rllib.utils.error.UnsupportedSpaceException: Action space (Discrete(3)) of <ray.rllib.policy.tf_policy_template.DDPGTFPolicy object at 0x7f5558d985b0> is not supported for DDPG.
# "ApexDDPG"             ## ray.rllib.utils.error.UnsupportedSpaceException: Action space (Discrete(3)) of <ray.rllib.policy.tf_policy_template.DDPGTFPolicy object at 0x7fe7b86e84c0> is not supported for DDPG.
# "TD3"                  ## ray.rllib.utils.error.UnsupportedSpaceException: Action space (Discrete(3)) of <ray.rllib.policy.tf_policy_template.DDPGTFPolicy object at 0x7f5dd32284c0> is not supported for DDPG.
# "R2D2"                 ## AssertionError: R2D2 requires its model to be a recurrent one! Try using `model.use_lstm` or `model.use_attention` in your config to auto-wrap your model with an LSTM- or attention net.
# "MAML"                 ## AttributeError: 'MountainCarEnv' object has no attribute 'sample_tasks'
# "MBMPO"                ## ValueError: Env <TimeLimit<MountainCarEnv<MountainCar-v0>>> doest not have a `reward()` method, needed for MB-MPO!
# "QMix"                 ## ValueError: Obs space must be a Tuple, got Box([-1.2  -0.07], [0.6  0.07], (2,), float32). Use MultiAgentEnv.with_agent_groups() to group related agents for QMix.

# "Apex"                 weak results and long learn

    ## working with good results
    ## "DQN", "SimpleQ"
    
    specific_config = {
#         'batch_mode': 'complete_episodes',
    }
    
    best_seed = args.seed
    best_iters = 5000
    best_layers = [ 16, 4 ]
#     best_learning = 0.0005
    metrics_smooth_size=100
    metrics_stop_condition = {
        'limit': -110.0,
        'metrics': 'avg'
    }
    custom_params = ""

#     specific_config['lr'] = best_learning
#     custom_params = "lr: {:.5f}".format( best_learning )
    trainer.learn( ENV_NAME, "DQN", layers_size=best_layers, 
                   n_iter=best_iters, metrics_stop_condition=metrics_stop_condition, metrics_smooth_size=metrics_smooth_size,
                   seed=best_seed, specific_config=specific_config, draw_interval=3, custom_params=custom_params )

    execution_time = datetime.now() - start_time
    print( "total duration: {}\n".format( execution_time ) )
    
    pyplot.show( block=True )


if __name__ == '__main__':
    main()
