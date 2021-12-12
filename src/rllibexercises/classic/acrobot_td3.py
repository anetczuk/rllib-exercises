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

import matplotlib.pyplot as pyplot
# import plot

from rllibexercises import trainer


## disable import warning
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"


ENV_NAME = "Acrobot-v1"


def main():
    start_time = datetime.now()

# "ARS"                  ## RuntimeWarning: invalid value encountered in double_scalars
# "CQL"                  ## AttributeError: 'Categorical' object has no attribute 'sample_logp'
# "DDPG"                 ## ray.rllib.utils.error.UnsupportedSpaceException: Action space (Discrete(3)) of <ray.rllib.policy.tf_policy_template.DDPGTFPolicy object at 0x7f543ffe95b0> is not supported for DDPG.
# "ApexDDPG"             ## ray.rllib.utils.error.UnsupportedSpaceException: Action space (Discrete(3)) of <ray.rllib.policy.tf_policy_template.DDPGTFPolicy object at 0x7f07c78335b0> is not supported for DDPG.
# "TD3"                  ## ray.rllib.utils.error.UnsupportedSpaceException: Action space (Discrete(3)) of <ray.rllib.policy.tf_policy_template.DDPGTFPolicy object at 0x7f4b7e0774c0> is not supported for DDPG.
# "R2D2"                 ## AssertionError: R2D2 requires its model to be a recurrent one! Try using `model.use_lstm` or `model.use_attention` in your config to auto-wrap your model with an LSTM- or attention net.
# "MAML"                 ## AttributeError: 'AcrobotEnv' object has no attribute 'sample_tasks'
# "MBMPO"                ## ValueError: Env <TimeLimit<AcrobotEnv<Acrobot-v1>>> doest not have a `reward()` method, needed for MB-MPO!
# "QMix"                 ## ValueError: Obs space must be a Tuple, got Box([ -1.        -1.        -1.        -1.       -12.566371 -28.274334], [ 1.        1.        1.        1.       12.566371 28.274334], (6,), float32). Use MultiAgentEnv.with_agent_groups() to group related agents for QMix.

    ### works
    ## A2C                                episode takes  10 sec
    ## A3C                                episode takes   5 sec
    ## ARS                                episode takes  40 sec
    ## Apex                               episode takes  40 sec
    ## DQN                                episode takes   6 sec
    ## SimpleQ                            episode takes   3 sec
    ## BC                                 episode takes   1 sec
    ## MARWIL                             episode takes   0.5 sec
    ## PG                                 episode takes   0.5 sec
    ## PPO                                episode takes   7 sec
    ## APPO                               episode takes  10 sec 
    ## SAC                                episode takes  20 sec
    
    
    custom_configs = {}
    
    best_alg = "DQN"
    # best_layers = [ 16, 4 ]
    # best_layers = [ 32, 8 ]
    # best_layers = [ 64, 16 ]
    best_layers = [ 128, 32 ]
    best_iters = 200
    best_seed = None
    best_config = None
#     best_learning = 0.02
    
    specific_config = custom_configs.get( best_alg, {} ).copy()
#     specific_config['lr'] = best_learning
#     param = "lr: {:.5f}".format( best_learning )
    param = None
     
    # trainer.learn( ENV_NAME, best_alg, layers_size=best_layers, n_iter=best_iters, seed=best_seed, specific_config=specific_config, custom_params=param )


    specific_config = custom_configs.get( best_alg, {} ).copy()
    learning_rate = [ 0.00006, 0.0001, 0.0006, 0.001, 0.006 ]
#     learning_rate = [ 0.01, 0.06, 0.1, 0.6, 1.0, 6.0, 11.0 ]
    for learning in learning_rate:
        specific_config['lr'] = learning
        param = "lr: {:.5f}".format( learning )
        trainer.learn( ENV_NAME, best_alg, layers_size=best_layers, n_iter=best_iters, seed=best_seed, 
                       specific_config=specific_config, custom_params=param )
    


#     failed = []
#     for alg in trainer.ALGORITHMS_WORKING:
#         try:
#             trainer.learn( ENV_NAME, alg, layers_size=best_layers, n_iter=best_iters, seed=best_seed, specific_config=best_config )
#         except Exception as e:
#             print( "exception:", e )
#             failed.append( alg )
#     print( "failed algorithms:", failed )

    execution_time = datetime.now() - start_time
    print( "total duration: {}\n".format( execution_time ) )
    
    pyplot.show( block=True )


if __name__ == '__main__':
    main()
