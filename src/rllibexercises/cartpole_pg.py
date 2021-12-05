#!/usr/bin/env python3

import os
# import sys
from datetime import datetime
# import shutil
import pprint

import numpy

# import json
import pandas as pd
import matplotlib.pyplot as pyplot
# import plot

# import ray
from ray.rllib.agents.pg import DEFAULT_CONFIG as DEFAULT_CONFIG
from ray.rllib.agents.pg import PGTrainer as Trainer


## disable import warning
os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"


SCRIPT_DIR      = os.path.abspath( os.path.dirname(__file__) )
TMP_DIR         = os.path.abspath( os.path.join( SCRIPT_DIR, "..", "..", "tmp" ) )
checkpoint_root = os.path.abspath( os.path.join( TMP_DIR, "run", "cartpole" ) )

os.makedirs( checkpoint_root, exist_ok=True )


# n_iter -- number of training runs.
def learn( layers_size, n_iter=32, seed=None ):
    start_time = datetime.now()

    if seed is None:
        seed = numpy.random.randint( 0, 2**32 - 1 )
    print( "random seed:", seed )
    
    config = DEFAULT_CONFIG.copy()

    ## common parameters
    common_config = {
        "log_level": "WARN",                    # Suppress too many messages, but try "INFO" to see what can be printed.
        "seed": seed,
        "num_workers": 1,                       # Use > 1 for using more CPU cores, including over a cluster
        "model": {
          "fcnet_hiddens": layers_size,
#           "fcnet_activation": "tanh",
        },
        "num_cpus_per_worker": 0,               # This avoids running out of resources in the notebook environment when this cell is re-executed
#         "train_batch_size": 1,
#         "rollout_fragment_length": 1
    }
    config.update( common_config )

    eval_config = {
        # Evaluate once per training iteration.
        "evaluation_interval": n_iter,
        # Run evaluation on (at least) two episodes
        "evaluation_num_episodes": 1,
        # ... using one evaluation worker (setting this to 0 will cause
        # evaluation to run on the local evaluation worker, blocking
        # training until evaluation is done).
        "evaluation_num_workers": 1,
        # Special evaluation config. Keys specified here will override
        # the same keys in the main config, but only for evaluation.
        "evaluation_config": {
            # Store videos in this relative directory here inside
            # the default output dir (~/ray_results/...).
            # Alternatively, you can specify an absolute path.
            # Set to True for using the default output dir (~/ray_results/...).
            # Set to False for not recording anything.
            "record_env": checkpoint_root,
  
            # Render the env while evaluating.
            # Note that this will always only render the 1st RolloutWorker's
            # env and only the 1st sub-env in a vectorized env.
            "render_env": True,
            "explore": False
        }
    }
    config.update( eval_config )

    ## algorithm specific config
    specific_config = {
#         # Learning rate.
#         "lr": 0.0004,
    }
    config.update( specific_config )

    pprint.pprint( config )
    
    params = "layers: {} n_iter: {} seed: {}".format( layers_size, n_iter, seed )
    
    print( "\ncreating agent" )
    agent = Trainer( config=config, env="CartPole-v1" )
    
#     results = []
    episode_data = []
#     episode_json = []

    pyplot.ion()
    ## ax is 'Line2D'
    plot_fig, ax = pyplot.subplots()
    subplot = ax.axes                   ## of type 'AxesSubplot'
        
    print( "\ntraining:" )
    for n in range(1, n_iter+1):
        result = agent.train()
#         pprint.pprint( result )
#         results.append( result )
        
#         print( "ooooooo:", len( result['hist_stats']['episode_lengths'] ) )
        
        episode = {'n': n, 
                   'episode_reward_min':  result['episode_reward_min'], 
                   'episode_reward_mean': result['episode_reward_mean'], 
                   'episode_reward_max':  result['episode_reward_max']
#                     'episode_len_mean':   result['episode_len_mean']
                   }
        
        episode_data.append( episode )
#         episode_json.append( json.dumps(episode) )
#         file_name = agent.save(checkpoint_root)
        
#         print( "agent_timesteps_total:", result['agent_timesteps_total'], "done: ", result['done'], "episodes_total: ", result['episodes_total']  )
        print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}')
#         print(f'{n:3d}: Min/Mean/Max reward: {result["episode_reward_min"]:8.4f}/{result["episode_reward_mean"]:8.4f}/{result["episode_reward_max"]:8.4f}. Checkpoint saved to {file_name}')

        if len(episode_data) > 0:
            df = pd.DataFrame( data=episode_data )
            subplot.clear()
            df.plot( ax=subplot, title="cartpole:\n" + params, x="n", y=["episode_reward_min", "episode_reward_mean", "episode_reward_max"] )
#             plot_fig.canvas.draw()
            plot_fig.canvas.flush_events()
#             plot.process_events( 0.1 )

    execution_time = datetime.now() - start_time

#     policy = agent.get_policy()
#     model = policy.model
#     
#     print( "\nvariables:" )
#     pprint.pprint( model.variables() )
#     
#     print( "\nvalue function:" )
#     pprint.pprint( model.value_function() )
#     
#     print( "\nmodel summary:" )
#     print( model.base_model.summary() )
    
    print( params )
    print( "duration: {}\n".format( execution_time ) )
    
    fileName = params
    fileName = fileName.replace( ", ", "_" )
    fileName = fileName.replace( ": ", "-" )
    fileName = fileName.replace( ":", "-" )
    fileName = fileName.replace( " ", "_" )
    figOutput = os.path.abspath( os.path.join( checkpoint_root, fileName + ".png" ) )
    plot_fig.savefig( figOutput )


def main():
    seed = None
    
#     learn( layers_size=[100, 50], n_iter=100 )
#     learn( layers_size=[32, 16], n_iter=100 )
#     learn( layers_size=[16, 4], n_iter=100 )

    learn( layers_size=[64, 32], n_iter=2200, seed=seed )
    
    pyplot.show( block=True )


if __name__ == '__main__':
    main()
