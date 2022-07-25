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

def extract_trajectory_actions(
    env, 
    initial_state, 
    states, 
    actions,
    done_mode,
    keep_textures=True,
    task_start_index=0,
):
    
    # Extract a trajectory, but instead of directly setting future MuJoCo states, take each action in sequence.
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state, reset_from_xml=keep_textures)

    # compute the goal with the arm at the initial state but the objects at final state
    print('task start idx', task_start_index)

    env.reset_to({'states': states[-1]}, reset_from_xml=keep_textures)
    env.env.sim.forward()
    final_object_positions = [deepcopy(env.env.sim.data.get_joint_qpos(obj.joints[0])) for obj in env.env.objects]
    initial_task_state = states[task_start_index]
    env.reset_to({'states': initial_task_state}, reset_from_xml=keep_textures)
    for obj, pos in zip(env.env.objects, final_object_positions):
        env.env.sim.data.set_joint_qpos(obj.joints[0], pos)
    env.step(np.zeros(actions.shape[1]))
    goal_obs = env.get_observation()
    goal_obs['state'] = env.get_state()['states']
    obs = env.reset_to(initial_state, reset_from_xml=keep_textures)
    if not keep_textures:
        initial_state = deepcopy(initial_state)
        initial_state['model'] = env.env.sim.model.get_xml()

    traj = dict(
        obs=[], 
        next_obs=[], 
        rewards=[], 
        dones=[], 
        actions=np.array(actions), 
        states=np.array(states), 
        initial_state_dict=initial_state,
        goal_obs=[goal_obs],
    )

    traj_len = states.shape[0]
    obs["state"] = env.get_state()["states"]
    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):
        next_obs, _, _, _ = env.step(actions[t - 1])

        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        next_obs['state'] = env.get_state()["states"]
        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)

        # update for next iter
        obs = deepcopy(next_obs)

    # convert list of dict to dict of list for obs dictionaries (for convenient writes to hdf5 dataset)
    traj["obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["obs"])
    traj["next_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["next_obs"])
    traj["goal_obs"] = TensorUtils.list_of_flat_dict_to_dict_of_list(traj["goal_obs"])

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


def extract_trajectory(
    env, 
    initial_state, 
    states, 
    actions,
    done_mode,
    keep_textures=True,
    task_start_index=0,
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
    assert isinstance(env, EnvBase)
    assert states.shape[0] == actions.shape[0]

    # load the initial state
    env.reset()
    obs = env.reset_to(initial_state, reset_from_xml=keep_textures)

    if not keep_textures:
        initial_state = deepcopy(initial_state)
        initial_state['model'] = env.env.sim.model.get_xml()

    traj = dict(
        obs=[],
        next_obs=[],
        object_positions=[],
        object_orientations=[],
        rewards=[],
        dones=[],
        actions=np.array(actions),
        states=np.array(states),
        initial_state_dict=initial_state,
    )
    traj_len = states.shape[0]

    # iteration variable @t is over "next obs" indices
    for t in range(1, traj_len + 1):
        # get next observation
        if t == traj_len:
            # play final action to get next observation for last timestep
            next_obs, _, _, _ = env.step(actions[t - 1])
        else:
            # reset to simulator state to get observation
            # env.step(np.zeros_like(actions[0])) # step env in order to trigger domain randomization
            next_obs = env.reset_to({"states": states[t]})
        # infer reward signal
        # note: our tasks use reward r(s'), reward AFTER transition, so this is
        #       the reward for the current timestep
        r = env.get_reward()

        # infer done signal
        done = False
        if (done_mode == 1) or (done_mode == 2):
            # done = 1 at end of trajectory
            done = done or (t == traj_len)
        if (done_mode == 0) or (done_mode == 2):
            # done = 1 when s' is task success state
            done = done or env.is_success()["task"]
        done = int(done)

        # collect transition
        traj["obs"].append(obs)
        traj["next_obs"].append(next_obs)
        traj["rewards"].append(r)
        traj["dones"].append(done)
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


def create_dataset_with_compression(group, name, data, compression="gzip", compression_opts=4):
    """
    Helper function to create a dataset with compression.

    Args:
        group (h5py.Group): group to create dataset in
        name (str): name of dataset
        data (np.array): data to write to dataset
        compression (str): compression type
        compression_opts (int): compression level
    """
    if compression is None:
        dset = group.create_dataset(name, data=data)
    elif compression == 'gzip':
        dset = group.create_dataset(name, data=data, chunks=data.shape, compression=compression,
                                    compression_opts=compression_opts, shuffle=True, scaleoffset=0)
    else:
        dset = group.create_dataset(name, data=data, chunks=data.shape, compression=compression, shuffle=True)
    return dset


def resize_tensor(t, dims, interpolation_mode):
    h, w = dims
    if len(t.shape) == 3:
        t = t[:, None]
    t = t.permute(0, 3, 1, 2)
    if t.shape[-2] == h and t.shape[-1] == w:
        out = t
    else:
        # uses Bilinear interpolation by default, use antialiasing
        if interpolation_mode == InterpolationMode.BILINEAR:
            antialias = True
        else:
            antialias = False
        t = Resize(dims, interpolation=interpolation_mode, antialias=antialias)(t)
        out = t
    return out.permute(0, 2, 3, 1)


from torchvision.transforms import Resize, InterpolationMode
def resize_trajectory(traj, target_size):
    # Resize all observations which are images to have the given size using PIL Image resize (antialiased)
    for k in traj["obs"]:
        if len(traj["obs"][k].shape) >= 3:
            traj["obs"][k] = resize_tensor(torch.tensor(traj["obs"][k]), target_size, interpolation_mode=InterpolationMode.BILINEAR).numpy()
    return traj


def dataset_states_to_obs(args):
    # create environment to use for data processing
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=args.dataset)
    env_meta['env_kwargs']['control_freq'] = 5
    # env_meta['env_kwargs']['textures'] = 'test_white' if args.unseen_textures and not args.transparent else 'train'
    env_meta['env_kwargs']['textures'] = 'train'
    # env_meta['env_kwargs']['textured_table'] = args.textured_table
    # env_meta['env_kwargs']['transparent'] = args.transparent
    env = EnvUtils.create_env_for_data_processing(
        env_meta=env_meta,
        camera_names=args.camera_names, 
        camera_depths=args.camera_depths,
        camera_segmentations=args.camera_segmentations,
        camera_normals=args.camera_normals,
        camera_height=args.render_height,
        camera_width=args.render_width,
        reward_shaping=args.shaped,
        randomize_lighting=args.randomize_lighting,
        randomize_color=args.randomize_color,
        randomize_freq=args.randomize_freq,
        renderer=args.renderer,
    )
    # some operations for playback are robosuite-specific, so determine if this environment is a robosuite env
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    # add domain randomization if specified 
    print("==== Using environment with the following metadata ====")
    print(json.dumps(env.serialize(), indent=4))
    print("")

    # list of all demonstration episodes (sorted in increasing number order)
    f = h5py.File(args.dataset, "r")
    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if args.n is not None:
        demos = demos[:args.n]

    # output file in same directory as input file
    output_path = os.path.join(os.path.dirname(args.dataset), args.output_name)
    f_out = h5py.File(output_path, "w")
    data_grp = f_out.create_group("data")
    print("input file: {}".format(args.dataset))
    print("output file: {}".format(output_path))

    total_samples = 0
    for ind in range(len(demos)):
        ep = demos[ind]

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        try:
            task_start_index = f["data/{}/start_index".format(ep)][()]
        except:
            task_start_index = 0
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]

        # extract obs, rewards, dones
        actions = f["data/{}/actions".format(ep)][()]
        if args.from_actions:
            traj = extract_trajectory_actions(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                keep_textures=not args.unseen_textures,
                task_start_index=task_start_index,
            )
        elif args.transparent:
            traj_transparent = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                keep_textures=not args.unseen_textures,
                task_start_index=task_start_index
            )
            traj_opaque_depths = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                keep_textures=True,
                task_start_index=task_start_index
            )
            for i, camera_name in enumerate(args.camera_names):
                if args.camera_depths[i]:
                    traj_transparent['obs'][f'{camera_name}_depth'] = traj_opaque_depths['obs'][f'{camera_name}_depth']
                    traj_transparent['next_obs'][f'{camera_name}_depth'] = traj_opaque_depths['next_obs'][f'{camera_name}_depth']
            traj = traj_transparent

        else:
            traj = extract_trajectory(
                env=env,
                initial_state=initial_state,
                states=states,
                actions=actions,
                done_mode=args.done_mode,
                keep_textures=not args.unseen_textures,
                task_start_index=task_start_index
            )
        if args.verbose:
            def create_visualization_gif(obs_list, obs_list_2, name, fps=5):
                from moviepy.editor import ImageSequenceClip
                obs_list = [np.concatenate((a, b), axis=0) for a, b in zip(obs_list, obs_list_2)]
                clip = ImageSequenceClip(obs_list, fps=fps)
                clip.write_gif(f'{name}.gif', fps=fps)
            from perceptual_metrics.mpc.utils import save_np_img
            # save_np_img(traj['obs']['agentview_image'].astype(np.uint8)[0], 'test_render64')
            # from PIL import Image
            # resize_image = Image.fromarray(traj['obs']['agentview_image'].astype(np.uint8)[0])
            # resize_image = resize_image.resize((64, 64), resample=Image.LANCZOS)
            # save_np_img(np.array(resize_image), 'test_resize_256_64_nearest')
            # quit()
            create_visualization_gif(list(traj["obs"]["agentview_shift_2_image"]), list(traj_actions["obs"]["agentview_shift_2_image"]), f"vis_traj{ind}")
            #create_visualization_gif(list(traj["obs"]["agentview_image"]), list(traj["obs"]["agentview_image"]), f"vis_traj{ind}")
            #create_visualization_gif(list(traj["obs"]["agentview_image"]), list(traj_actions["obs"]["agentview_image"] - traj["obs"]["agentview_image"]), f"vis_traj{ind}")

        # maybe copy reward or done signal from source file
        if args.copy_rewards:
            traj["rewards"] = f["data/{}/rewards".format(ep)][()]
        if args.copy_dones:
            traj["dones"] = f["data/{}/dones".format(ep)][()]

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
            # ep_data_grp.create_dataset("obs/{}".format(k), data=np.array(traj["obs"][k]))
            create_dataset_with_compression(ep_data_grp, "obs/{}".format(k), data=np.array(traj["obs"][k]))
            if "goal_obs" in traj:
                # ep_data_grp.create_dataset("goal_obs/{}".format(k), data=np.array(traj["goal_obs"][k]))
                create_dataset_with_compression(ep_data_grp, "goal_obs/{}".format(k), data=np.array(traj["goal_obs"][k]))
            if args.store_next_obs:
                # ep_data_grp.create_dataset("next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))
                create_dataset_with_compression(ep_data_grp, "next_obs/{}".format(k), data=np.array(traj["next_obs"][k]))
        # episode metadata
        if is_robosuite_env:
            ep_data_grp.attrs["model_file"] = traj["initial_state_dict"]["model"] # model xml for this episode
        ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0] # number of transitions in this episode
        total_samples += traj["actions"].shape[0]
        print("ep {}: Wrote {} transitions to group {}".format(ind, ep_data_grp.attrs["num_samples"], ep))

    # copy over all filter keys that exist in the original hdf5
    if "mask" in f:
        f.copy("mask", f_out)

    # global metadata
    data_grp.attrs["total"] = total_samples
    data_grp.attrs["env_args"] = json.dumps(env.serialize(), indent=4) # environment info
    print("Wrote {} trajectories to {}".format(len(demos), output_path))

    f.close()
    f_out.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="path to input hdf5 dataset",
    )
    # name of hdf5 to write - it will be in the same directory as @dataset
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="name of output hdf5 dataset",
    )

    # specify number of demos to process - useful for debugging conversion with a handful
    # of trajectories
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="(optional) stop after n trajectories are processed",
    )

    # flag for reward shaping
    parser.add_argument(
        "--shaped", 
        action='store_true',
        help="(optional) use shaped rewards",
    )

    parser.add_argument(
        "--renderer",
        type=str,
        default="mujoco",
        help="renderer to use",
    )

    # flag for domain randomization
    parser.add_argument(
        "--randomize_lighting", 
        action='store_true',
        help="(optional) Use lighting domain randomization",
    )

    parser.add_argument(
        "--randomize_color", 
        action='store_true',
        help="(optional) Use color domain randomization",
    )

    parser.add_argument(
        "--randomize_freq", 
        type=int,
        default=0,
        help="frequency of randomization",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_names",
        type=str,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for image observations. Leave out to not use image observations.",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_depths",
        type=int,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for depth observations. Leave out to not use depth observations.",
    )

    # camera names to use for observations
    parser.add_argument(
        "--camera_normals",
        type=int,
        nargs='+',
        default=[],
        help="(optional) camera name(s) to use for normal observations. Leave out to not use depth observations.",
    )

    parser.add_argument(
        "--camera_segmentations",
        type=str,
        nargs='+',
        default=[],
        help="(optional) types of camera segmentations to use",
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
        default=84,
        help="(optional) height of image observations",
    )

    parser.add_argument(
        "--camera_width",
        type=int,
        default=84,
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

    # flag for copying rewards from source file instead of re-writing them
    parser.add_argument(
        "--copy_rewards", 
        action='store_true',
        help="(optional) copy rewards from source file instead of inferring them",
    )

    # flag for copying dones from source file instead of re-writing them
    parser.add_argument(
        "--copy_dones", 
        action='store_true',
        help="(optional) copy dones from source file instead of inferring them",
    )

    # flag for verbose output 
    parser.add_argument(
        "--verbose", 
        action='store_true',
        help="(optional) output comparison GIFs",
    )

    # Whether or not to save next obs, setting this to False will almost halve file size
    parser.add_argument(
        "--store_next_obs",
        action='store_true',
        help="(optional) whether to store next step observations",
    )

    parser.add_argument(
        "--from_actions",
        action='store_true',
        help="(optional) use actions to generate trajectory",
    )

    parser.add_argument(
        "--unseen_textures",
        action='store_true',
        help="(optional) unseen textures",
    )

    parser.add_argument(
        "--transparent",
        action='store_true',
        help="(optional) object transparency",
    )

    parser.add_argument(
        "--textured_table",
        action='store_true',
        help="(optional) object transparency",
    )

    args = parser.parse_args()
    dataset_states_to_obs(args)
