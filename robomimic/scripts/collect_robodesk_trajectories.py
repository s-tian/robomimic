"""
Script to extract observations from low-dimensional simulation states in a robosuite dataset.

Args:
    dataset (str): path to input hdf5 dataset

    output_name (str): name of output hdf5 dataset

    n (int): if provided, stop after n trajectories are processed

    shaped (bool): if flag is set, use dense rewards

    camera_names (str or [str]): camera name(s) to use for image observations. 
        Leave out to not use image observations.

    camera_height (int): height of image observation.

    camera_width (int): width of image observation

    done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a success state.
        If 1, done is 1 at the end of each trajectory. If 2, both.

    copy_rewards (bool): if provided, copy rewards from source file instead of inferring them

    copy_dones (bool): if provided, copy dones from source file instead of inferring them

Example usage:
    
    # extract low-dimensional observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name low_dim.hdf5 --done_mode 2
    
    # extract 84x84 image observations
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image.hdf5 \
        --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84

    # use dense rewards, and only annotate the end of trajectories with done signal
    python dataset_states_to_obs.py --dataset /path/to/demo.hdf5 --output_name image_dense_done_1.hdf5 \
        --done_mode 1 --dense --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84
"""
import os
import json
import h5py
import argparse
import numpy as np
import torch
from copy import deepcopy
from PIL import Image

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_base import EnvBase
from robomimic.scripts.dataset_states_to_obs import create_dataset_with_compression, resize_tensor, resize_trajectory

from perceptual_metrics.envs.robodesk_env import RoboDeskEnv
from perceptual_metrics.envs.scripted_policies.robodesk_scripted_policies import TASK_TO_POLICY
from perceptual_metrics.envs.scripted_policies.noisy_policy_wrapper import NoisyPolicyWrapper
from robomimic.scripts.split_train_val import split_train_val_from_hdf5

def extract_trajectory(
    env, 
    policy,
    num_steps,
    done_mode,
):
    """
    Helper function to extract observations, rewards, and dones along a trajectory using
    the simulator environment.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load to extract information
        actions (np.array): array of actions
        done_mode (int): how to write done signal. If 0, done is 1 whenever s' is a 
            success state. If 1, done is 1 at the end of each trajectory. 
            If 2, do both.
    """
    # load the initial state
    obs = env.reset()
    policy.reset()
    initial_state = dict(state=env.get_state())

    traj = dict(
        obs=[],
        next_obs=[],
        object_positions=[],
        object_orientations=[],
        rewards=[],
        dones=[],
        actions=[],
        states=[],
        initial_state_dict=initial_state,
    )
    traj_len = num_steps

    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len+1):
        # get next observation
        action = policy.get_action(env)
        next_obs, rew, done, info = env.step(action)
        r = rew
        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
        traj["actions"].append(action)
        traj["states"].append(env.get_state())
        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])

    # list to numpy array
    for k in traj:
        if k == "initial_state_dict":
            continue
        if isinstance(traj[k], dict):
            for kp in traj[k]:
                traj[k][kp] = np.array(traj[k][kp])
        else:
            traj[k] = np.array(traj[k])

    return traj

def dataset_states_to_obs(args):
    # create environment to use for data processing

    assert args.task in TASK_TO_POLICY, "Task {} not supported, must be one of {}".format(args.task, TASK_TO_POLICY.keys())
    env = RoboDeskEnv(action_repeat=50, task=args.task, image_size=max(args.render_width, args.render_height))
    policy = NoisyPolicyWrapper(TASK_TO_POLICY[args.task], noise_std=args.policy_noise_std)

    print('Using env', env, 'with task ', args.task)

    # output file in same directory as input file
    output_path = args.output_name
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in range(args.n):
        ep = f"demo_{ind+1}"
        # prepare initial state to reload from
        traj = extract_trajectory(
            env=env,
            done_mode=args.done_mode,
            policy=policy,
            num_steps=args.num_steps,
        )

        if (args.render_height, args.render_width) != (args.camera_height, args.camera_width):
            traj = resize_trajectory(traj, target_size=(args.camera_height, args.camera_width))

        # store transitions
        # IMPORTANT: keep name of group the same as source file, to make sure that filter keys are
        #            consistent as well
        ep_data_grp = data_grp.create_group(ep)
        ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
        ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
        ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
        ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))
        for k in traj["obs"]:
            if 'segmentation' in k:
                # convert segmentations from uint32 to uint8, since we probably don't have that many segmentations
                traj["obs"][k] = traj["obs"][k].astype(np.uint8)
                traj["next_obs"][k] = traj["next_obs"][k].astype(np.uint8)
                if "goal_obs" in traj:
                    traj["goal_obs"][k] = traj["goal_obs"][k].astype(np.uint8)
            create_dataset_with_compression(ep_data_grp, "obs/{}".format(k), data=np.array(traj["obs"][k]))
            if "goal_obs" in traj:
                create_dataset_with_compression(ep_data_grp, "goal_obs/{}".format(k), data=np.array(traj["goal_obs"][k]))
            if args.store_next_obs:
                create_dataset_with_compression(ep_data_grp, "next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))
        # episode metadata
        # if is_robosuite_env:
        #     ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: Wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))

    # copy over all filter keys that exist in the original hdf5

    # global metadata
    data_grp.attrs["total"] = total_samples
    # data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(args.n, output_path))

    f_out.close()
    split_train_val_from_hdf5(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        required=True,
        help="number of steps per trajectory",
    )

    parser.add_argument(
        "--policy_noise_std",
        type=float,
        required=True,
        help="number of steps per trajectory",
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="name of task",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    parser.add_argument(
        "--render_height",
        type=int,
        default=256,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--render_width",
        type=int,
        default=256,
        help="(optional) width of image observations",
    )

    parser.add_argument(
        "--camera_height",
        type=int,
        default=256,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=256,
        help="(optional) width of image observations",
    )

    # specifies how the "done" signal is written. If "0", then the "done" signal is 1 wherever 
    # the transition (s, a, s') has s' in a task completion state. If "1", the "done" signal 
    # is one at the end of every trajectory. If "2", the "done" signal is 1 at task completion
    # states for successful trajectories and 1 at the end of all trajectories.
    parser.add_argument(
        "--done_mode",
        type=int,
        default=0,
        help="how to write done signal. If 0, done is 1 whenever s' is a success state.\
            If 1, done is 1 at the end of each trajectory. If 2, both.",
    )


    # Whether or not to save next obs, setting this to False will almost halve file size
    parser.add_argument(
        "--store_next_obs",
        action='store_true',
        help="(optional) whether to store next step observations",
    )


    args = parser.parse_args()
    dataset_states_to_obs(args)
