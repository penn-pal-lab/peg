import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import dreamerv2.api as dv2
import common
from common import Config
import envs
import numpy as np
import collections
import matplotlib.pyplot as plt
from dreamerv2.common.replay import convert
import pathlib
import sys
import ruamel.yaml as yaml

def make_env(config,  use_goal_idx=False, log_per_goal=False, eval=False):
  """
  Create environments from LEXA benchmark or MEGA benchmark.
  use_goal_idx, log_per_goal are LEXA benchmark specific args.
  eval flag used for creating MEGA eval envs
  """
  def wrap_lexa_env(e):
    e = common.GymWrapper(e)
    if hasattr(e.act_space['action'], 'n'):
      e = common.OneHotAction(e)
    else:
      e = common.NormalizeAction(e)
    e = common.TimeLimit(e, config.time_limit)
    return e

  def wrap_mega_env(e, info_to_obs_fn=None):
    e = common.GymWrapper(e, info_to_obs_fn=info_to_obs_fn)
    if hasattr(e.act_space['action'], 'n'):
      e = common.OneHotAction(e)
    else:
      e = common.NormalizeAction(e)
    return e

  if 'dmc' in config.task:
    suite_task, obs = config.task.rsplit('_', 1)
    suite, task = suite_task.split('_', 1)
    if 'proprio' in config.task:
      env = envs.DmcStatesEnv(task, config.render_size, config.action_repeat, use_goal_idx, log_per_goal)
      if 'humanoid' in config.task:
        keys = ['qpos', 'goal']
        env = common.NormObsWrapper(env, env.obs_bounds[:, 0], env.obs_bounds[:, 1], keys)
    elif 'vision' in config.task:
      env = envs.DmcEnv(task, config.render_size, config.action_repeat, use_goal_idx, log_per_goal)
    env = wrap_lexa_env(env)
  elif 'mtmw' in config.task:
    env = envs.MetaWorld('mtmw_sawyer_SawyerReachEnv', config.action_repeat, use_goal_idx, log_per_goal)
    # only for metaworld env
    env._env.max_path_length = np.inf
    env = wrap_lexa_env(env)
  elif 'kitchen' in config.task:
    env = envs.KitchenStatesEnv(action_repeat=config.action_repeat, use_goal_idx=use_goal_idx, log_per_goal=log_per_goal)
    keys = ['state', 'goal']
    env = common.NormObsWrapper(env, env.obs_bounds[:, 0], env.obs_bounds[:, 1], keys)
    env = wrap_lexa_env(env)
  elif 'robobin' in config.task:
    if 'proprio' in config.task:
      env = envs.RoboBinStatesEnv(config.action_repeat, use_goal_idx, log_per_goal)
    elif 'vision' in config.task:
      env = envs.RoboBinEnv(config.action_repeat, use_goal_idx, log_per_goal) # image based version
    # only for metaworld env
    env._env.max_path_length = np.inf
    env = wrap_lexa_env(env)
  elif 'pointmaze' in config.task:
    from envs.sibrivalry.toy_maze import MultiGoalPointMaze2D
    env = MultiGoalPointMaze2D(test=eval)
    env.max_steps = config.time_limit
    # PointMaze2D is a GoalEnv, so rename obs dict keys.
    env = common.ConvertGoalEnvWrapper(env)
    # LEXA assumes information is in obs dict already, so move info dict into obs.
    info_to_obs = None
    if eval:
      def info_to_obs(info, obs):
        if info is None:
          info = env.get_metrics_dict()
        obs = obs.copy()
        for k,v in info.items():
          if "metric" in k:
            obs[k] = v
        return obs
    env = wrap_mega_env(env, info_to_obs)
    class GaussianActions:
      """Add gaussian noise to the actions.
      """
      def __init__(self, env, std):
        self._env = env
        self.std = std

      def __getattr__(self, name):
        return getattr(self._env, name)

      def step(self, action):
        new_action = action
        if self.std > 0:
          noise = np.random.normal(scale=self.std, size=2)
          if isinstance(action, dict):
            new_action = {'action': action['action'] + noise}
          else:
            new_action = action + noise

        return self._env.step(new_action)
    env = GaussianActions(env, std=0)

  elif 'umazefull' == config.task:
    from envs.sibrivalry.ant_maze import AntMazeEnvFull
    env = AntMazeEnvFull(eval=eval)
    env.max_steps = config.time_limit
    # Antmaze is a GoalEnv
    env = common.ConvertGoalEnvWrapper(env)
    env = wrap_mega_env(env)
  elif config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
    from envs.sibrivalry.ant_maze import AntMazeEnvFullDownscale, AntHardMazeEnvFullDownscale
    if 'hard' in config.task:
      env = AntHardMazeEnvFullDownscale(eval=eval)
    else:
      env = AntMazeEnvFullDownscale(eval=eval)
    env.max_steps = config.time_limit
    # Antmaze is a GoalEnv
    env = common.ConvertGoalEnvWrapper(env)
    info_to_obs = None
    if eval:
      def info_to_obs(info, obs):
        if info is None:
          info = env.get_metrics_dict()
        obs = obs.copy()
        for k,v in info.items():
          if "metric" in k:
            obs[k] = v
        return obs
    env = wrap_mega_env(env, info_to_obs)
  elif 'a1umazefulldownscale' == config.task:
    from envs.sibrivalry.ant_maze import A1MazeEnvFullDownscale
    env = A1MazeEnvFullDownscale(eval=eval)
    env.max_steps = config.time_limit
    # Antmaze is a GoalEnv
    env = common.ConvertGoalEnvWrapper(env)
    info_to_obs = None
    if eval:
      def info_to_obs(info, obs):
        if info is None:
          info = env.get_metrics_dict()
        obs = obs.copy()
        for k,v in info.items():
          if "metric" in k:
            obs[k] = v
        return obs
    env = wrap_mega_env(env, info_to_obs)
  elif config.task in {'fetchpnp', 'fetchpnpeasy', 'demofetchpnp'}:
    if config.task in {'fetchpnp', 'fetchpnpeasy'}:
      from envs.customfetch.custom_fetch import EasyPickPlaceEnv, PickPlaceEnv, GoalType
      n_blocks = 1 # THIS IS THE "IN_AIR_PERCENTAGE"
      range_min = 0.2 # THIS IS THE MINIMUM_AIR
      range_max = 0.45 # THIS IS THE MAXIMUM_AIR
      if config.task == 'fetchpnp':
        Env = PickPlaceEnv
      elif config.task == 'fetchpnpeasy':
        Env = EasyPickPlaceEnv

      internal = GoalType.OBJ
      # external = GoalType.ALL
      external = GoalType.OBJ
      env = Env(max_step=config.time_limit, internal_goal = internal, external_goal = external, mode=0,
                          per_dim_threshold=0, hard=True, distance_threshold=0, n = n_blocks,
                          range_min=range_min, range_max=range_max)
  elif config.task in {'discwallsdemofetchpnp', 'wallsdemofetchpnp2', 'wallsdemofetchpnp3','demofetchpnp'}:
    from envs.customfetch.custom_fetch import DemoStackEnv, WallsDemoStackEnv, DiscreteWallsDemoStackEnv
    if 'walls' in config.task:
      if 'disc' in config.task:
        env = DiscreteWallsDemoStackEnv(max_step=config.time_limit, eval=eval, increment=0.01)
      else:
        n = int(config.task[-1])
        env = WallsDemoStackEnv(max_step=config.time_limit, eval=eval, n=int(config.task[-1]))
    else:
      env = DemoStackEnv(max_step=config.time_limit, eval=eval)

    # Antmaze is a GoalEnv
    env = common.ConvertGoalEnvWrapper(env)
    # LEXA assumes information is in obs dict already, so move info dict into obs.
    info_to_obs = None
    if config.task in {'discwallsdemofetchpnp', 'wallsdemofetchpnp2','wallsdemofetchpnp3','demofetchpnp'}:
      def info_to_obs(info, obs):
        if info is None:
          info = env.get_metrics_dict()
        obs = obs.copy()
        for k,v in info.items():
          if eval:
            if "metric" in k:
              obs[k] = v
          else:
            if "above" in k:
              obs[k] = v
        return obs
    else:
      if eval:
        def info_to_obs(info, obs):
          if info is None:
            info = env.get_metrics_dict()
          obs = obs.copy()
          for k,v in info.items():
            if "is_success" in k:
              obs[k] = v
          return obs

    class ClipObsWrapper:
      def __init__(self, env, obs_min, obs_max):
        self._env = env
        self.obs_min = obs_min
        self.obs_max = obs_max

      def __getattr__(self, name):
        return getattr(self._env, name)

      def step(self, action):
        obs, rew, done, info = self._env.step(action)
        new_obs = np.clip(obs['observation'], self.obs_min, self.obs_max)
        obs['observation'] = new_obs
        return obs, rew, done, info
    obs_min = np.ones(env.observation_space['observation'].shape) * -1e6
    pos_min = [1.0, 0.3, 0.35]
    if 'demofetchpnp' in config.task:
      obs_min[:3] = obs_min[5:8] = obs_min[8:11] = pos_min
      if env.n == 3:
        obs_min[11:14] = pos_min

    obs_max = np.ones(env.observation_space['observation'].shape) * 1e6
    pos_max = [1.6, 1.2, 1.0]
    if 'demofetchpnp' in config.task:
      obs_max[:3] = obs_max[5:8] = obs_max[8:11] = pos_max
      if env.n == 3:
        obs_max[11:14] = pos_max

    env = ClipObsWrapper(env, obs_min, obs_max)

    # first 3 dim are grip pos, next 2 dim are gripper, next n * 3 are obj pos.
    if n == 2: # noisy dim
      obs_min_noise = np.ones(noise_dim) * noise_low
      obs_min = np.concatenate([env.workspace_min, [0., 0.],  *[env.workspace_min for _ in range(env.n)], obs_min_noise], 0)
      obs_max_noise = np.ones(noise_dim) * noise_high
      obs_max = np.concatenate([env.workspace_max, [0.05, 0.05],  *[env.workspace_max for _ in range(env.n)], obs_max_noise], 0)
    else:
      obs_min = np.concatenate([env.workspace_min, [0., 0.],  *[env.workspace_min for _ in range(env.n)]], 0)
      obs_max = np.concatenate([env.workspace_max, [0.05, 0.05],  *[env.workspace_max for _ in range(env.n)]], 0)
    env = common.NormObsWrapper(env, obs_min, obs_max)
    env = wrap_mega_env(env, info_to_obs)
  return env

def make_report_render_function(config):
  """
  video_from_state_fn used to render model predictions. see report function in gc_agent.py
  """
  video_from_state_fn = None
  if 'dmc' in config.task:
    # TODO: implement state render fn for dmc.
    if 'vision' in config.task:
      # image based env doesn't need this.
      video_from_state_fn = None
  elif 'mtmw' in config.task:
    def video_from_state_fn(recon, openl, truth, env):
      # now render the states with the environment
      inner_env = env._env._env._env._env
      flat_recon = recon.numpy().reshape(-1,9)
      flat_openl = openl.numpy().reshape(-1,9)
      flat_truth = truth.numpy().reshape(-1,9)
      def generate_img_from_state(states):
        all_img = []
        for qpos in states:
          hand_init_pos = inner_env.hand_init_pos
          obj_init_pos = inner_env.init_config['obj_init_pos']
          # Render state
          hand_pos, obj_pos, hand_to_goal = np.split(qpos, 3)
          inner_env.hand_init_pos = hand_pos
          inner_env.init_config['obj_init_pos'] = obj_pos
          inner_env.reset_model()
          img = (env.render_offscreen().astype(np.float32) / 255.0) - 0.5
          # Revert environment
          inner_env.hand_init_pos = hand_init_pos
          inner_env.init_config['obj_init_pos'] = obj_init_pos
          inner_env.reset()
          all_img.append(img)
        return all_img
      recon_imgs = np.stack(generate_img_from_state(flat_recon),0)
      openl_imgs = np.stack(generate_img_from_state(flat_openl),0)
      truth_imgs = np.stack(generate_img_from_state(flat_truth),0) + 0.5
      recon_imgs = recon_imgs.reshape([*recon.shape[:2], *recon_imgs.shape[-3:]])
      openl_imgs = openl_imgs.reshape([*openl.shape[:2], *openl_imgs.shape[-3:]])
      truth_imgs = truth_imgs.reshape([*truth.shape[:2], *truth_imgs.shape[-3:]])

      model = tf.concat([recon_imgs[:, :5] + 0.5, openl_imgs + 0.5], 1)
      error = (model - truth_imgs + 1) / 2
      video = tf.concat([truth_imgs, model, error], 2)
      B, T, H, W, C = video.shape
      return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
  elif 'kitchen' in config.task:
    return None
  elif 'robobin' in config.task:
    def video_from_state_fn(recon, openl, truth, env):
      # now render the states with the environment
      inner_env = env._env._env._env._env
      flat_recon = recon.numpy().reshape(-1,9)
      flat_openl = openl.numpy().reshape(-1,9)
      flat_truth = truth.numpy().reshape(-1,9)
      def generate_img_from_state(states):
        all_img = []
        for qpos in states:
          obj_init_pos_temp = inner_env.init_config['obj_init_pos'].copy()
          goal = qpos

          inner_env.init_config['obj_init_pos'] = goal[3:]
          inner_env.obj_init_pos = goal[3:]
          inner_env.hand_init_pos = goal[:3]
          inner_env.reset_model()
          action = np.zeros(inner_env.action_space.low.shape)
          state, reward, done, info = inner_env.step(action)

          img = (env.render_offscreen().astype(np.float32) / 255.0) - 0.5
          inner_env.hand_init_pos = inner_env.init_config['hand_init_pos']
          inner_env.init_config['obj_init_pos'] = obj_init_pos_temp
          inner_env.obj_init_pos = inner_env.init_config['obj_init_pos']
          inner_env.reset()
          all_img.append(img)
        return all_img
      recon_imgs = np.stack(generate_img_from_state(flat_recon),0)
      openl_imgs = np.stack(generate_img_from_state(flat_openl),0)
      truth_imgs = np.stack(generate_img_from_state(flat_truth),0) + 0.5
      recon_imgs = recon_imgs.reshape([*recon.shape[:2], *recon_imgs.shape[-3:]])
      openl_imgs = openl_imgs.reshape([*openl.shape[:2], *openl_imgs.shape[-3:]])
      truth_imgs = truth_imgs.reshape([*truth.shape[:2], *truth_imgs.shape[-3:]])

      model = tf.concat([recon_imgs[:, :5] + 0.5, openl_imgs + 0.5], 1)
      error = (model - truth_imgs + 1) / 2
      video = tf.concat([truth_imgs, model, error], 2)
      B, T, H, W, C = video.shape
      return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    if config.task == 'robobin_vision':
      # image based env doesn't need this.
      video_from_state_fn = None
  elif 'pointmaze' in config.task:
    def video_from_state_fn(recon, openl, truth, env):
      # now render the states with the environment
      inner_env = env._env._env._env._env
      flat_recon = recon.numpy().reshape(-1,2)
      flat_openl = openl.numpy().reshape(-1,2)
      flat_truth = truth.numpy().reshape(-1,2)
      def generate_img_from_state(states):
        all_img = []
        for xy in states:
          inner_env.s_xy = xy
          img = (env.render().astype(np.float32) / 255.0) - 0.5
          all_img.append(img)
        # Revert environment
        env.clear_plots()
        return all_img
      recon_imgs = np.stack(generate_img_from_state(flat_recon),0)
      openl_imgs = np.stack(generate_img_from_state(flat_openl),0)
      truth_imgs = np.stack(generate_img_from_state(flat_truth),0) + 0.5
      inner_env.reset()
      recon_imgs = recon_imgs.reshape([*recon.shape[:2], *recon_imgs.shape[-3:]])
      openl_imgs = openl_imgs.reshape([*openl.shape[:2], *openl_imgs.shape[-3:]])
      truth_imgs = truth_imgs.reshape([*truth.shape[:2], *truth_imgs.shape[-3:]])

      model = tf.concat([recon_imgs[:, :5] + 0.5, openl_imgs + 0.5], 1)
      error = (model - truth_imgs + 1) / 2
      video = tf.concat([truth_imgs, model, error], 2)
      B, T, H, W, C = video.shape
      return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
  elif 'umazefull' == config.task:
    return None
  elif config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
    return None
  elif 'a1umazefulldownscale' == config.task:
    return None
  elif 'demofetchpnp' in config.task:
    return None
  elif config.task in {'fetchpnp', 'fetchpnpeasy'}:
    from gym.envs.robotics import rotations
    def video_from_state_fn(recon, openl, truth, env):
      sim = env.sim
      inner_env = env._env._env._env._env
      flat_recon = recon.numpy().reshape(-1,25)
      flat_openl = openl.numpy().reshape(-1,25)
      flat_truth = truth.numpy().reshape(-1,25)
      def generate_img_from_state(states):
        all_img = []
        for i, obs in enumerate(states):
          obj_pos = obs[:3]
          grip_pos = obs[3:6]
          obj_rel_pos = obs[6:9]
          gripper_state = obs[9:11]
          object_rot = obs[11:14]
          # reset the robot.
          if i == 0:
            env.reset()
          # move the robot end effector to correct position.
          gripper_target = grip_pos
          gripper_rotation = np.array([1., 0., 1., 0.])
          sim.data.set_mocap_pos('robot0:mocap', gripper_target)
          sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
          # set the gripper to the correct position.
          sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", gripper_state[0])
          sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", gripper_state[1])
          # set the objects to the correct position.
          obj_quat = rotations.euler2quat(object_rot)
          sim.data.set_joint_qpos("object0:joint", [*obj_pos,*obj_quat])
          # step the sim
          for _ in range(1):
            env.sim.step()
          env.sim.forward()
          img = (env.render("rgb_array", 200, 200).astype(np.float32) / 255.0) - 0.5
          all_img.append(img)
        all_img = np.stack(all_img, 0)
        return all_img
      recon_imgs = np.stack(generate_img_from_state(flat_recon),0)
      openl_imgs = np.stack(generate_img_from_state(flat_openl),0)
      truth_imgs = np.stack(generate_img_from_state(flat_truth),0) + 0.5
      inner_env.reset()
      recon_imgs = recon_imgs.reshape([*recon.shape[:2], *recon_imgs.shape[-3:]])
      openl_imgs = openl_imgs.reshape([*openl.shape[:2], *openl_imgs.shape[-3:]])
      truth_imgs = truth_imgs.reshape([*truth.shape[:2], *truth_imgs.shape[-3:]])

      model = tf.concat([recon_imgs[:, :5] + 0.5, openl_imgs + 0.5], 1)
      error = (model - truth_imgs + 1) / 2
      video = tf.concat([truth_imgs, model, error], 2)
      B, T, H, W, C = video.shape
      return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))
    if config.no_render:
      video_from_state_fn = None
  else:
    raise NotImplementedError
  return video_from_state_fn

def make_eval_fn(config):
  """ Make the function that evaluates the environment.
  """
  if 'dmc' in config.task or 'mtmw' in config.task or 'robobin' in config.task or 'kitchen' in config.task:
    if 'dmc' in config.task:
      episode_render_fn = None
      if 'proprio' in config.task:
        def episode_render_fn(env, ep):
          all_img = []
          goals = []
          executions = []
          inner_env = env._env._env._env._env
          if 'humanoid' in config.task:
            inner_env = env._env._env._env._env._env
            def unnorm_ob(ob):
              return env.obs_min + ob * (env.obs_max -  env.obs_min)
          goal_and_ep_qpos = [ep['goal'][0], *ep['qpos']]
          for qpos in goal_and_ep_qpos:
            size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
            if 'humanoid' in config.task:
              qpos = unnorm_ob(qpos)
            inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
            inner_env.physics.step()
            img = env.render()
            all_img.append(img)

          goals.append(all_img[0][None]) # 1 x H x W x C
          ep_img = np.stack(all_img[1:], 0)
          executions.append(ep_img[None]) # 1 x T x H x W x C
          return goals, executions

    elif 'mtmw' in config.task :
      def episode_render_fn(env, ep):
        all_img = []
        goals = []
        executions = []
        inner_env = env._env._env._env._env
        goal_and_ep_qpos = [ep['goal'][0], *ep['qpos']]
        for qpos in goal_and_ep_qpos:
          hand_init_pos = inner_env.hand_init_pos
          obj_init_pos = inner_env.init_config['obj_init_pos']
          # Render state
          hand_pos, obj_pos, hand_to_goal = np.split(qpos, 3)
          inner_env.hand_init_pos = hand_pos
          inner_env.init_config['obj_init_pos'] = obj_pos
          inner_env.reset_model()
          img = env.render_offscreen()
          # Revert environment
          inner_env.hand_init_pos = hand_init_pos
          inner_env.init_config['obj_init_pos'] = obj_init_pos
          inner_env.reset()
          all_img.append(img)
        goals.append(all_img[0][None]) # 1 x H x W x C
        ep_img = np.stack(all_img[1:], 0)
        executions.append(ep_img[None]) # 1 x T x H x W x C
        return goals, executions
    elif 'kitchen' in config.task:
      def episode_render_fn(env, ep):
        all_img = []
        goals = []
        executions = []
        kitchen_env = env._env._env._env._env
        inner_env = kitchen_env._env
        def unnorm_ob(ob):
          return env.obs_min + ob * (env.obs_max -  env.obs_min)
        for state in [ep['goal'][0], *ep['state']]:
          init_qpos = np.copy(inner_env.init_qpos)
          state = unnorm_ob(state)
          for obs_idx, obs_val in zip(kitchen_env.obs_idxs, state):
            init_qpos[obs_idx] = obs_val
          inner_env.set_state(init_qpos, np.zeros_like(init_qpos[:-1]))
          img = inner_env.render('rgb_array', width=100, height=100)
          all_img.append(img)
        goals.append(all_img[0][None]) # 1 x H x W x C
        ep_img = np.stack(all_img[1:], 0)
        executions.append(ep_img[None]) # 1 x T x H x W x C
        return goals, executions
    elif 'robobin' in config.task:
      def episode_render_fn(env, ep):
        all_img = []
        goals = []
        executions = []
        inner_env = env._env._env._env._env
        goal_and_ep_qpos = [ep['goal'][0], *ep['qpos']]
        for qpos in goal_and_ep_qpos:
          obj_init_pos_temp = inner_env.init_config['obj_init_pos'].copy()
          goal = qpos

          inner_env.init_config['obj_init_pos'] = goal[3:]
          inner_env.obj_init_pos = goal[3:]
          inner_env.hand_init_pos = goal[:3]
          inner_env.reset_model()
          action = np.zeros(inner_env.action_space.low.shape)
          state, reward, done, info = inner_env.step(action)

          img = env.render_offscreen()
          inner_env.hand_init_pos = inner_env.init_config['hand_init_pos']
          inner_env.init_config['obj_init_pos'] = obj_init_pos_temp
          inner_env.obj_init_pos = inner_env.init_config['obj_init_pos']
          inner_env.reset()
          all_img.append(img)
        goals.append(all_img[0][None]) # 1 x H x W x C
        ep_img = np.stack(all_img[1:], 0)
        executions.append(ep_img[None]) # 1 x T x H x W x C
        return goals, executions
      if config.task == 'robobin_vision':
        # image based env doesn't need this.
        episode_render_fn = None

    def evaluate_all_goals(driver, eval_policy, logger):
      env = driver._envs[0]
      num_goals = len(env.get_goals())
      num_eval_eps = 10
      executions = []
      goals = []
      # key is metric name, value is list of size num_eval_eps
      all_metric_success = []
      ep_metrics = collections.defaultdict(list)
      for ep_idx in range(num_eval_eps):
        should_video = ep_idx == 0
        state_based_render = episode_render_fn is not None
        for idx in range(num_goals):
          env.set_goal_idx(idx)
          env.reset()
          driver(eval_policy, episodes=1)
          ep = driver._eps[0] # get episode data of 1st env.
          ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
          score = float(ep['reward'].astype(np.float64).sum())
          print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
          for k, v in ep.items():
            if 'metric_success' in k:
              all_metric_success.append(np.max(v))
              ep_metrics[k].append(np.max(v))
            elif 'metric_reward' in k:
              ep_metrics[k].append(np.sum(v))

          if not should_video:
            continue
          if state_based_render:
            """ rendering based on state."""
            # render the goal img and rollout
            _goals, _executions = episode_render_fn(env, ep)
            goals.extend(_goals)
            executions.extend(_executions)
          else: # image based env
            _goals = ep[config.goal_key]
            _executions = ep[config.state_key]
            goals.append(_goals)
            executions.append(_executions)


        if should_video:
          if state_based_render:
            executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
          else:
            executions = np.stack(executions, 0) # num_goals x T x H x W x C
          goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
          if state_based_render:
            goals = np.repeat(goals, executions.shape[1], 1)
          gc_video = np.concatenate([goals, executions], -3)
          logger.video(f'eval_gc_policy', gc_video)

      # collect all the goal success metrics and get average
      all_metric_success = np.mean(all_metric_success)
      logger.scalar('max_eval_metric_success/goal_all', all_metric_success)
      for key, value in ep_metrics.items():
        if 'metric_success' in key:
          logger.scalar(f'max_eval_{key}', np.mean(value))
        elif 'metric_reward' in key:
          logger.scalar(f'sum_eval_{key}', np.mean(value))
      logger.write()

  elif 'pointmaze' in config.task:
    # Pointmaze and MEGA envs define a goal distribution, so we sample from it for eval.
    def episode_render_fn(env, ep):
      all_img = []
      goals = []
      executions = []
      inner_env = env._env._env._env._env
      inner_env.g_xy = ep['goal'][0]
      inner_env.s_xy = ep['goal'][0]
      goal_img = env.render()
      for xy in ep['observation']:
        inner_env.s_xy = xy
        img = env.render()
        all_img.append(img)
      env.clear_plots()
      goals.append(goal_img[None]) # 1 x H x W x C
      ep_img = np.stack(all_img, 0)
      # pad if episode length is shorter than time limit.
      T = ep_img.shape[0]
      ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
      executions.append(ep_img[None]) # 1 x T x H x W x C
      return goals, executions
    def evaluate_all_goals(driver, eval_policy, logger):
      env = driver._envs[0]
      num_goals = len(env.get_goals()) # maze has 5 goals
      num_eval_eps = 1
      executions = []
      goals = []
      all_metric_success = []
      all_metric_success_cell = []
      for ep_idx in range(num_eval_eps):
        should_video = ep_idx == 0 and episode_render_fn is not None
        for idx in range(num_goals):
          env.set_goal_idx(idx)
          driver(eval_policy, episodes=1)
          if not should_video:
            continue
          """ rendering based on state."""
          ep = driver._eps[0] # get episode data of 1st env.
          ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
          # aggregate goal metrics across goals together.
          for k, v in ep.items():
            if 'metric' in k:
              if 'cell' in k.split('/')[0]:
                all_metric_success_cell.append(np.max(v))
              else:
                all_metric_success.append(np.max(v))
          # render the goal img and rollout
          _goals, _executions = episode_render_fn(env, ep)
          goals.extend(_goals)
          executions.extend(_executions)

        if should_video:
          executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
          goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
          goals = np.repeat(goals, executions.shape[1], 1)
          gc_video = np.concatenate([goals, executions], -3)
          logger.video(f'eval_gc_policy', gc_video)
      all_metric_success = np.mean(all_metric_success)
      logger.scalar('max_eval_metric_success/goal_all', all_metric_success)
      all_metric_success_cell = np.mean(all_metric_success_cell)
      logger.scalar('max_eval_metric_success_cell/goal_all', all_metric_success_cell)
      logger.write()
  elif config.task in {'umazefull','umazefulldownscale', 'hardumazefulldownscale'}:
    def episode_render_fn(env, ep):
      all_img = []
      goals = []
      executions = []
      ant_env = env.maze.wrapped_env
      ant_env.set_state(ep['goal'][0][:15], ep['goal'][0][:14])
      # other_dims = np.concatenate([[6.08193526e-01,  9.87496030e-01,
      # 1.82685311e-03, -6.82827458e-03,  1.57485326e-01,  5.14617396e-02,
      # 1.22386603e+00, -6.58701813e-02, -1.06980319e+00,  5.09069276e-01,
      # -1.15506861e+00,  5.25953435e-01,  7.11716520e-01], np.zeros(14)])
      inner_env = env._env._env._env
      # inner_env.g_xy = np.concatenate((inner_env.goal_list[inner_env.goal_idx], other_dims))
      inner_env.g_xy = inner_env.goal_list[inner_env.goal_idx]
      ant_env.sim.forward()
      goal_img = env.render(mode='rgb_array')
      for obs in ep['observation']:
        env.maze.wrapped_env.set_state(obs[:15], np.zeros_like(obs[:14]))
        env.maze.wrapped_env.sim.forward()
        img = env.render(mode='rgb_array')
        all_img.append(img)

      goals.append(goal_img[None]) # 1 x H x W x C
      ep_img = np.stack(all_img, 0)
      # pad if episode length is shorter than time limit.
      T = ep_img.shape[0]
      ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
      executions.append(ep_img[::2][None]) # 1 x T x H x W x C
      return goals, executions
    def evaluate_all_goals(driver, eval_policy, logger):
      env = driver._envs[0]
      num_goals = len(env.get_goals()) if len(env.get_goals()) > 0 else 5 # MEGA uses 30 episodes for eval.
      num_eval_eps = 10
      executions = []
      goals = []
      all_metric_success = []
      # key is metric name, value is list of size num_eval_eps
      ep_metrics = collections.defaultdict(list)
      for ep_idx in range(num_eval_eps):
        should_video = ep_idx == 0 and episode_render_fn is not None
        for idx in range(num_goals):
          env.set_goal_idx(idx)
          driver.reset()
          driver(eval_policy, episodes=1)
          """ rendering based on state."""
          ep = driver._eps[0] # get episode data of 1st env.
          ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
          score = float(ep['reward'].astype(np.float64).sum())
          print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
          # render the goal img and rollout
          for k, v in ep.items():
            if 'metric' in k:
              ep_metrics[k].append(np.max(v))
              all_metric_success.append(np.max(v))

          if not should_video:
            continue
          _goals, _executions = episode_render_fn(env, ep)
          goals.extend(_goals)
          executions.extend(_executions)
        if should_video:
          executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
          goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
          goals = np.repeat(goals, executions.shape[1], 1)
          gc_video = np.concatenate([goals, executions], -3)
          logger.video('eval_gc_policy', gc_video)
      all_metric_success = np.mean(all_metric_success)
      logger.scalar('max_eval_metric_success/goal_all', all_metric_success)
      for key, value in ep_metrics.items():
        logger.scalar(f'mean_eval_{key}', np.mean(value))
      logger.write()
  elif 'a1umazefulldownscale' == config.task:
    def episode_render_fn(env, ep):
      all_img = []
      goals = []
      executions = []
      a1_env = env.maze.wrapped_env
      a1_env.set_state(ep['goal'][0][:19], ep['goal'][0][:18])
      other_dims = np.concatenate([[0.24556014,  0.986648,    0.09023235, -0.09100603,
      0.10050705, -0.07250207, -0.01489305,  0.09989551, -0.05246516, -0.05311238,
      -0.01864055, -0.05934234,  0.03910208, -0.08356607,  0.05515265, -0.00453086,
      -0.01196933], np.zeros(18)])
      inner_env = env._env._env._env
      inner_env.g_xy = np.concatenate((inner_env.goal_list[inner_env.goal_idx], other_dims))
      a1_env.sim.forward()
      goal_img = env.render(mode='rgb_array')
      for obs in ep['observation']:
        env.maze.wrapped_env.set_state(obs[:19], np.zeros_like(obs[:18]))
        env.maze.wrapped_env.sim.forward()
        img = env.render(mode='rgb_array')
        all_img.append(img)

      goals.append(goal_img[None]) # 1 x H x W x C
      ep_img = np.stack(all_img, 0)
      # pad if episode length is shorter than time limit.
      T = ep_img.shape[0]
      ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'constant', constant_values=(0))
      executions.append(ep_img[::2][None]) # 1 x T x H x W x C
      return goals, executions
    def evaluate_all_goals(driver, eval_policy, logger):
      env = driver._envs[0]
      num_goals = len(env.get_goals()) if len(env.get_goals()) > 0 else 5 # MEGA uses 30 episodes for eval.
      num_eval_eps = 10
      executions = []
      goals = []
      all_metric_success = []
      # key is metric name, value is list of size num_eval_eps
      ep_metrics = collections.defaultdict(list)
      for ep_idx in range(num_eval_eps):
        should_video = ep_idx == 0 and episode_render_fn is not None
        for idx in range(num_goals):
          env.set_goal_idx(idx)
          driver(eval_policy, episodes=1)
          """ rendering based on state."""
          ep = driver._eps[0] # get episode data of 1st env.
          ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
          score = float(ep['reward'].astype(np.float64).sum())
          print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
          # render the goal img and rollout
          for k, v in ep.items():
            if 'metric' in k:
              ep_metrics[k].append(np.max(v))
              all_metric_success.append(np.max(v))

          if not should_video:
            continue
          _goals, _executions = episode_render_fn(env, ep)
          goals.extend(_goals)
          executions.extend(_executions)
        if should_video:
          executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
          goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
          goals = np.repeat(goals, executions.shape[1], 1)
          gc_video = np.concatenate([goals, executions], -3)
          logger.video('eval_gc_policy', gc_video)
      all_metric_success = np.mean(all_metric_success)
      logger.scalar('max_eval_metric_success/goal_all', all_metric_success)
      for key, value in ep_metrics.items():
        logger.scalar(f'mean_eval_{key}', np.mean(value))
      logger.write()
  elif 'fetchpnp' == config.task:
    from gym.envs.robotics import rotations
    def episode_render_fn(env, ep):
      sim = env.sim
      all_img = []
      goals = []
      executions = []
      env.reset()
      inner_env = env._env._env._env._env
      inner_env.goal = ep['goal'][0]
      # now render the states.
      for obs in ep['observation']:
        grip_pos = obs[3:6]
        obj_pos = obs[:3]
        obj_rel_pos = obs[6:9]
        gripper_state = obs[9:11]
        object_rot = obs[11:14]
        # move the robot end effector to correct position.
        gripper_target = grip_pos
        gripper_rotation = np.array([1., 0., 1., 0.])
        sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        # set the gripper to the correct position.
        gripper_vel = obs[-2:]
        sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", gripper_state[0])
        sim.data.set_joint_qvel("robot0:r_gripper_finger_joint", gripper_vel[0])
        sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", gripper_state[1])
        sim.data.set_joint_qvel("robot0:l_gripper_finger_joint", gripper_vel[1])
        for _ in range(1):
          env.sim.step()
        # set the objects to the correct position.
        obj_quat = rotations.euler2quat(object_rot)
        sim.data.set_joint_qpos("object0:joint", [*obj_pos,*obj_quat])
        # step the sim
        env.sim.forward()
        img = env.render(mode='rgb_array', width=100, height=100)
        all_img.append(img)
      goals.append(all_img[0][None]) # 1 x H x W x C
      ep_img = np.stack(all_img, 0)
      executions.append(ep_img[None]) # 1 x T x H x W x C
      return goals, executions
    if config.no_render:
      episode_render_fn = None

    def evaluate_all_goals(driver, eval_policy, logger):
      env = driver._envs[0]
      num_goals = 5 # MEGA uses 30 episodes for eval.
      num_eval_eps = 1
      executions = []
      goals = []
      for ep_idx in range(num_eval_eps):
        should_video = ep_idx == 0 and episode_render_fn is not None
        for idx in range(num_goals):
          driver(eval_policy, episodes=1)
          if not should_video:
            continue
          """ rendering based on state."""
          ep = driver._eps[0] # get episode data of 1st env.
          ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
          # render the goal img and rollout
          _goals, _executions = episode_render_fn(env, ep)
          goals.extend(_goals)
          executions.extend(_executions)
        if should_video:
          executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
          goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
          goals = np.repeat(goals, executions.shape[1], 1)
          gc_video = np.concatenate([goals, executions], -3)
          logger.video(f'eval_gc_policy', gc_video)
        logger.write()
  elif 'fetchpnpeasy' == config.task:
    from gym.envs.robotics import rotations
    def episode_render_fn(env, ep):
      sim = env.sim
      all_img = []
      goals = []
      executions = []
      env.reset()
      inner_env = env._env._env._env._env
      inner_env.goal = ep['goal'][0]
      # now render the states.
      for obs in ep['observation']:
        grip_pos = obs[3:6]
        obj_pos = obs[:3]
        obj_rel_pos = obs[6:9]
        gripper_state = obs[9:11]
        object_rot = obs[11:14]
        # move the robot end effector to correct position.
        gripper_target = grip_pos
        gripper_rotation = np.array([1., 0., 1., 0.])
        sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        # set the gripper to the correct position.
        gripper_vel = obs[-2:]
        sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", gripper_state[0])
        sim.data.set_joint_qvel("robot0:r_gripper_finger_joint", gripper_vel[0])
        sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", gripper_state[1])
        sim.data.set_joint_qvel("robot0:l_gripper_finger_joint", gripper_vel[1])
        for _ in range(1):
          env.sim.step()
        # set the objects to the correct position.
        obj_quat = rotations.euler2quat(object_rot)
        sim.data.set_joint_qpos("object0:joint", [*obj_pos,*obj_quat])
        # step the sim
        env.sim.forward()
        img = env.render(mode='rgb_array', width=100, height=100)
        all_img.append(img)
      goals.append(all_img[0][None]) # 1 x H x W x C
      ep_img = np.stack(all_img, 0)
      executions.append(ep_img[None]) # 1 x T x H x W x C
      return goals, executions
    if config.no_render:
      episode_render_fn = None

    def evaluate_all_goals(driver, eval_policy, logger):
      env = driver._envs[0]
      num_goals = len(env.get_goals()) if len(env.get_goals()) > 0 else 5 # MEGA uses 30 episodes for eval.
      num_eval_eps = 10
      executions = []
      goals = []
      all_metric_success = []
      # key is metric name, value is list of size num_eval_eps
      ep_metrics = collections.defaultdict(list)
      for ep_idx in range(num_eval_eps):
        should_video = ep_idx == 0 and episode_render_fn is not None
        for idx in range(num_goals):
          env.set_goal_idx(idx)
          driver(eval_policy, episodes=1)
          """ rendering based on state."""
          ep = driver._eps[0] # get episode data of 1st env.
          ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
          score = float(ep['reward'].astype(np.float64).sum())
          print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
          # render the goal img and rollout
          for k, v in ep.items():
            if 'metric' in k:
              ep_metrics[k].append(np.max(v))
              all_metric_success.append(np.max(v))

          if not should_video:
            continue
          _goals, _executions = episode_render_fn(env, ep)
          goals.extend(_goals)
          executions.extend(_executions)
        if should_video:
          executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
          goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
          goals = np.repeat(goals, executions.shape[1], 1)
          gc_video = np.concatenate([goals, executions], -3)
          logger.video('eval_gc_policy', gc_video)
      all_metric_success = np.mean(all_metric_success)
      logger.scalar('max_eval_metric_success/goal_all', all_metric_success)
      for key, value in ep_metrics.items():
        logger.scalar(f'mean_eval_{key}', np.mean(value))
      logger.write()
  elif 'demofetchpnp' in config.task:
    from gym.envs.robotics import rotations
    def episode_render_fn(env, ep):
      sim = env.sim
      all_img = []
      goals = []
      executions = []
      env.reset()
      # move the robot arm out of the way
      if env.n == 2:
        out_of_way_state = np.array([ 4.40000000e+00,  4.04998318e-01,  4.79998255e-01,  3.11127168e-06,
          1.92819215e-02, -1.26133677e+00,  9.24837728e-02, -1.74551950e+00,
        -6.79993234e-01, -1.62616316e+00,  4.89490853e-01,  1.25022086e+00,
          2.02171933e+00, -2.35683450e+00,  8.60046276e-03, -6.44277362e-08,
          1.29999928e+00,  5.99999425e-01,  4.24784489e-01,  1.00000000e+00,
        -2.13882881e-07,  2.67353601e-07, -1.03622169e-15,  1.29999961e+00,
          8.99999228e-01,  4.24784489e-01,  1.00000000e+00, -2.95494240e-07,
          1.47747120e-07, -2.41072272e-15, -5.44202926e-07, -5.43454906e-07,
          7.61923038e-07,  5.39374476e-03,  1.92362793e-12,  7.54386574e-05,
          2.07866306e-04,  7.29063886e-03, -6.50353144e-03,  2.87876616e-03,
          8.29802372e-03, -3.06640616e-03, -1.17278073e-03,  2.71063610e-03,
        -1.62474545e-06, -1.60648093e-07, -1.28518475e-07,  1.09679929e-14,
          5.16300606e-06, -6.45375757e-06,  4.68203006e-17, -8.87786549e-08,
        -1.77557310e-07,  1.09035019e-14,  7.13305591e-06, -3.56652796e-06,
          6.54969586e-17])
      elif env.n == 3:
        out_of_way_state = np.array([4.40000000e+00,  4.04999349e-01,  4.79999636e-01,  2.79652104e-06, 1.56722299e-02,-3.41500342e+00, 9.11469058e-02,-1.27681180e+00,
      -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
        2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
        1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
      -2.28652597e-07, 2.56090909e-07,-1.20181003e-15, 1.32999955e+00,
        8.49999274e-01, 4.24784489e-01, 1.00000000e+00,-2.77140579e-07,
        1.72443027e-07,-1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
        4.24784489e-01, 1.00000000e+00,-2.31485576e-07, 2.31485577e-07,
      -6.68816586e-16,-4.48284993e-08,-8.37398903e-09, 7.56100615e-07,
        5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
      -7.15601860e-02,-9.44665089e-02, 1.49646097e-02,-1.10990294e-01,
      -3.30174644e-03, 1.19462201e-01, 4.05130821e-04,-3.95036450e-04,
      -1.53880539e-07,-1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
      -6.18188284e-06, 1.31307184e-17,-1.03617993e-07,-1.66528917e-07,
        1.06089030e-14, 6.69000941e-06,-4.16267252e-06, 3.63225324e-17,
      -1.39095626e-07,-1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
      -5.58792469e-06,-2.07082526e-17])
      sim.set_state_from_flattened(out_of_way_state)
      sim.forward()

      # inner_env.goal = ep['goal'][0]
      # now render the states.
      sites_offset = (sim.data.site_xpos - sim.model.site_pos)
      site_id = sim.model.site_name2id('gripper_site')

      def unnorm_ob(ob):
        return env.obs_min + ob * (env.obs_max -  env.obs_min)
      for obs in [ep['goal'][0], *ep['observation']]:
        obs = unnorm_ob(obs)
        grip_pos = obs[:3]
        gripper_state = obs[3:5]
        all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
        # set the end effector site instead of the actual end effector.
        sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
        # set the objects
        for i, pos in enumerate(all_obj_pos):
          sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1,0,0,0]])

        sim.forward()
        img = sim.render(height=100, width=100, camera_name="external_camera_0")[::-1]
        all_img.append(img)
      goals.append(all_img[0][None]) # 1 x H x W x C
      ep_img = np.stack(all_img[1:], 0)
      # pad if episode length is shorter than time limit.
      T = ep_img.shape[0]
      ep_img = np.pad(ep_img, ((0, (config.time_limit+1) - T), (0,0), (0,0), (0,0)), 'edge')
      executions.append(ep_img[None]) # 1 x T x H x W x C
      return goals, executions
    if config.no_render:
      episode_render_fn = None

    def evaluate_all_goals(driver, eval_policy, logger):
      env = driver._envs[0]
      if env.n == 3:
        # eval_goal_idxs = range(24, 36)
        # TODO: revert back to 3 block goals
        eval_goal_idxs = range(len(env.get_goals()))
      elif env.n == 2:
        eval_goal_idxs = range(len(env.get_goals()))
      num_eval_eps = 10
      executions = []
      goals = []
      all_metric_success = []
      # key is metric name, value is list of size num_eval_eps
      ep_metrics = collections.defaultdict(list)
      for ep_idx in range(num_eval_eps):
        should_video = ep_idx == 0 and episode_render_fn is not None
        for idx in eval_goal_idxs:
          driver.reset()
          env.set_goal_idx(idx)
          driver(eval_policy, episodes=1)
          """ rendering based on state."""
          ep = driver._eps[0] # get episode data of 1st env.
          ep = {k: driver._convert([t[k] for t in ep]) for k in ep[0]}
          score = float(ep['reward'].astype(np.float64).sum())
          print(f'Eval goal {idx} has {len(ep["reward"] - 1)} steps and return {score:.1f}.')
          # render the goal img and rollout
          for k, v in ep.items():
            if 'metric_success/goal_' in k:
              ep_metrics[k].append(np.max(v))
              all_metric_success.append(np.max(v))

          if not should_video:
            continue
          # render the goal img and rollout
          _goals, _executions = episode_render_fn(env, ep)
          goals.extend(_goals)
          executions.extend(_executions)
        if should_video:
          executions = np.concatenate(executions, 0) # num_goals x T x H x W x C
          goals = np.stack(goals, 0) # num_goals x 1 x H x W x C
          goals = np.repeat(goals, executions.shape[1], 1)
          gc_video = np.concatenate([goals, executions], -3)
          logger.video(f'eval_gc_policy', gc_video)
      all_metric_success = np.mean(all_metric_success)
      logger.scalar('mean_eval_metric_success/goal_all', all_metric_success)
      for key, value in ep_metrics.items():
        logger.scalar(f'mean_eval_{key}', np.mean(value))
      logger.write()
  else:
    raise NotImplementedError
  return evaluate_all_goals

def make_ep_render_fn(config):
  episode_render_fn = None
  if 'dmc_walker_walk_proprio' == config.task:
    def episode_render_fn(env, ep):
      all_img = []
      inner_env = env._env._env._env._env
      for qpos in ep['qpos']:
        size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
        inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
        inner_env.physics.step()
        img = env.render()
        all_img.append(img)

      ep_img = np.stack(all_img[1:], 0)
      return ep_img
  elif 'dmc_humanoid_walk_proprio' == config.task:
    def episode_render_fn(env, ep):
      all_img = []
      inner_env = env._env._env._env._env._env
      def unnorm_ob(ob):
        return env.obs_min + ob * (env.obs_max -  env.obs_min)
      for qpos in ep['qpos']:
        size = inner_env.physics.get_state().shape[0] - qpos.shape[0]
        qpos = unnorm_ob(qpos)
        inner_env.physics.set_state(np.concatenate((qpos, np.zeros([size]))))
        inner_env.physics.step()
        img = env.render()
        all_img.append(img)

      ep_img = np.stack(all_img[1:], 0)
      return ep_img
  elif 'kitchen' in config.task:
    def episode_render_fn(env, ep):
      all_img = []
      kitchen_env = env._env._env._env._env
      inner_env = kitchen_env._env
      def unnorm_ob(ob):
        return env.obs_min + ob * (env.obs_max -  env.obs_min)
      for state in ep['state']:
        init_qpos = np.copy(inner_env.init_qpos)
        state = unnorm_ob(state)
        for obs_idx, obs_val in zip(kitchen_env.obs_idxs, state):
          init_qpos[obs_idx] = obs_val
        inner_env.set_state(init_qpos, np.zeros_like(init_qpos[:-1]))
        img = inner_env.render('rgb_array', width=100, height=100)
        all_img.append(img)
      all_img = np.stack(all_img, 0)
      return all_img

  elif 'pointmaze' in config.task:
    def episode_render_fn(env, ep):
      all_img = []
      inner_env = env._env._env._env._env
      for g_xy, xy in zip(ep['goal'], ep['observation']):
        inner_env.g_xy = g_xy
        inner_env.s_xy = xy
        img = env.render()
        all_img.append(img)
      env.clear_plots()
      all_img = np.stack(all_img, 0) # T x H x W x C
      return all_img
  elif config.task in {'umazefull','umazefulldownscale', 'hardumazefulldownscale'}:
    def episode_render_fn(env, ep):
      inner_env = env._env._env._env
      all_img = []
      for obs, goal in zip(ep['observation'], ep['goal']):
        inner_env.maze.wrapped_env.set_state(obs[:15], np.zeros_like(obs[:14]))
        inner_env.g_xy = goal[:2]
        inner_env.maze.wrapped_env.sim.forward()
        img = env.render(mode='rgb_array')
        all_img.append(img)
      all_img = np.stack(all_img, 0)
      return all_img
  elif 'a1umazefulldownscale' == config.task:
    def episode_render_fn(env, ep):
      inner_env = env._env._env._env
      all_img = []
      for obs, goal in zip(ep['observation'], ep['goal']):
        inner_env.maze.wrapped_env.set_state(obs[:19], np.zeros_like(obs[:18]))
        inner_env.g_xy = goal[:2]
        inner_env.maze.wrapped_env.sim.forward()
        img = env.render(mode='rgb_array')
        all_img.append(img)
      all_img = np.stack(all_img, 0)
      return all_img
  elif 'demofetchpnp' in config.task:
    from gym.envs.robotics import rotations
    import cv2
    def episode_render_fn(env, ep):
      sim = env.sim
      all_img = []
      # reset the robot.
      env.reset()
      inner_env = env._env._env._env._env._env
      # move the robot arm out of the way
      if env.n == 2:
        out_of_way_state = np.array([ 4.40000000e+00,  4.04998318e-01,  4.79998255e-01,  3.11127168e-06,
          1.92819215e-02, -1.26133677e+00,  9.24837728e-02, -1.74551950e+00,
        -6.79993234e-01, -1.62616316e+00,  4.89490853e-01,  1.25022086e+00,
          2.02171933e+00, -2.35683450e+00,  8.60046276e-03, -6.44277362e-08,
          1.29999928e+00,  5.99999425e-01,  4.24784489e-01,  1.00000000e+00,
        -2.13882881e-07,  2.67353601e-07, -1.03622169e-15,  1.29999961e+00,
          8.99999228e-01,  4.24784489e-01,  1.00000000e+00, -2.95494240e-07,
          1.47747120e-07, -2.41072272e-15, -5.44202926e-07, -5.43454906e-07,
          7.61923038e-07,  5.39374476e-03,  1.92362793e-12,  7.54386574e-05,
          2.07866306e-04,  7.29063886e-03, -6.50353144e-03,  2.87876616e-03,
          8.29802372e-03, -3.06640616e-03, -1.17278073e-03,  2.71063610e-03,
        -1.62474545e-06, -1.60648093e-07, -1.28518475e-07,  1.09679929e-14,
          5.16300606e-06, -6.45375757e-06,  4.68203006e-17, -8.87786549e-08,
        -1.77557310e-07,  1.09035019e-14,  7.13305591e-06, -3.56652796e-06,
          6.54969586e-17])
      elif env.n == 3:
        out_of_way_state = np.array([4.40000000e+00,  4.04999349e-01,  4.79999636e-01,  2.79652104e-06,
      1.56722299e-02,-3.41500342e+00, 9.11469058e-02,-1.27681180e+00,
    -1.39750475e+00, 4.43858450e+00, 7.47892234e-01, 2.53633962e-01,
      2.34366216e+00, 3.35102418e+00, 8.32919575e-04, 1.41610111e-03,
      1.32999932e+00, 6.49999392e-01, 4.24784489e-01, 1.00000000e+00,
    -2.28652597e-07, 2.56090909e-07,-1.20181003e-15, 1.32999955e+00,
      8.49999274e-01, 4.24784489e-01, 1.00000000e+00,-2.77140579e-07,
      1.72443027e-07,-1.77971404e-15, 1.39999939e+00, 7.49999392e-01,
      4.24784489e-01, 1.00000000e+00,-2.31485576e-07, 2.31485577e-07,
    -6.68816586e-16,-4.48284993e-08,-8.37398903e-09, 7.56100615e-07,
      5.33433335e-03, 2.91848485e-01, 7.45623586e-05, 2.99902784e-01,
    -7.15601860e-02,-9.44665089e-02, 1.49646097e-02,-1.10990294e-01,
    -3.30174644e-03, 1.19462201e-01, 4.05130821e-04,-3.95036450e-04,
    -1.53880539e-07,-1.37393338e-07, 1.07636483e-14, 5.51953825e-06,
    -6.18188284e-06, 1.31307184e-17,-1.03617993e-07,-1.66528917e-07,
      1.06089030e-14, 6.69000941e-06,-4.16267252e-06, 3.63225324e-17,
    -1.39095626e-07,-1.39095626e-07, 1.10587840e-14, 5.58792469e-06,
    -5.58792469e-06,-2.07082526e-17])
      sim.set_state_from_flattened(out_of_way_state)
      sim.forward()
      inner_env.goal = ep['goal'][0]
      subgoal_time = ep['log_subgoal_time']
      sites_offset = (sim.data.site_xpos - sim.model.site_pos)
      site_id = sim.model.site_name2id('gripper_site')
      def unnorm_ob(ob):
        return env.obs_min + ob * (env.obs_max -  env.obs_min)
      for i, obs in enumerate(ep['observation']):
        obs = unnorm_ob(obs)
        grip_pos = obs[:3]
        gripper_state = obs[3:5]
        all_obj_pos = np.split(obs[5:5+3*env.n], env.n)
        # set the end effector site instead of the actual end effector.
        sim.model.site_pos[site_id] = grip_pos - sites_offset[site_id]
        # set the objects
        for i, pos in enumerate(all_obj_pos):
          sim.data.set_joint_qpos(f"object{i}:joint", [*pos, *[1,0,0,0]])

        sim.forward()
        img = sim.render(height=200, width=200, camera_name="external_camera_0")[::-1]
        if subgoal_time > 0 and i >= subgoal_time:
          img = img.copy()
          cv2.putText(
            img,
            f"expl",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
          )
        all_img.append(img)
      all_img = np.stack(all_img, 0)
      return all_img
  elif config.task in {'fetchpnp', 'fetchpnpeasy'}:
    from gym.envs.robotics import rotations
    import cv2
    def episode_render_fn(env, ep):
      sim = env.sim
      all_img = []
      # reset the robot.
      env.reset()
      inner_env = env._env._env._env._env
      inner_env.goal = ep['goal'][0]
      subgoal_time = ep['log_subgoal_time']
      for i, obs in enumerate(ep['observation']):
        obj_pos = obs[:3]
        grip_pos = obs[3:6]
        obj_rel_pos = obs[6:9]
        gripper_state = obs[9:11]
        object_rot = obs[11:14]
        # move the robot end effector to correct position.
        gripper_target = grip_pos
        gripper_rotation = np.array([1., 0., 1., 0.])
        sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        # set the gripper to the correct position.
        gripper_vel = obs[-2:]
        sim.data.set_joint_qpos("robot0:r_gripper_finger_joint", gripper_state[0])
        sim.data.set_joint_qvel("robot0:r_gripper_finger_joint", gripper_vel[0])
        sim.data.set_joint_qpos("robot0:l_gripper_finger_joint", gripper_state[1])
        sim.data.set_joint_qvel("robot0:l_gripper_finger_joint", gripper_vel[1])
        # step the sim
        for _ in range(1):
          env.sim.step()
        # set the objects to the correct position.
        obj_quat = rotations.euler2quat(object_rot)
        sim.data.set_joint_qpos("object0:joint", [*obj_pos,*obj_quat])
        env.sim.forward()
        img = env.render("rgb_array", 200,200)
        if subgoal_time > 0 and i >= subgoal_time:
          img = img.copy()
          cv2.putText(
            img,
            f"expl",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
          )
        all_img.append(img)
      all_img = np.stack(all_img, 0)
      return all_img
  if config.no_render:
    episode_render_fn = None
  return episode_render_fn

def make_cem_vis_fn(config):
  vis_fn = None
  if 'pointmaze' in config.task:
    num_vis = 10
    def vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger):
      elite_seq = tf.nest.map_structure(lambda x: tf.gather(x, elite_inds[:num_vis], axis=1), seq)
      elite_obs = wm.heads['decoder'](wm.rssm.get_feat(elite_seq))['observation'].mode()

      goal_states = tf.repeat(elite_samples[None, :num_vis], elite_obs.shape[0], axis=0).numpy() # T x topk x 2
      goal_states = goal_states.reshape(-1,2)
      maze_states =  elite_obs.numpy().reshape(-1, 2)
      inner_env = eval_env._env._env._env._env
      all_img = []
      for xy, g_xy in zip(maze_states, goal_states):
        inner_env.s_xy = xy
        inner_env.g_xy = g_xy
        img = (eval_env.render().astype(np.float32) / 255.0) - 0.5
        all_img.append(img)
      # Revert environment
      eval_env.clear_plots()
      imgs = np.stack(all_img, 0)
      imgs = imgs.reshape([*elite_obs.shape[:2], 100,100,3]) # T x B x H x W x 3
      T,B,H,W,C = imgs.shape
      # want T,H,B,W,C
      imgs = imgs.transpose(0,2,1,3,4).reshape((T,H,B*W,C)) + 0.5
      metric = {f"top_{num_vis}_cem": imgs}
      logger.add(metric)
      logger.write()
  elif config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
    num_vis = 10
    def vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger):
      elite_seq = tf.nest.map_structure(lambda x: tf.gather(x, elite_inds[:num_vis], axis=1), seq)
      elite_obs = wm.heads['decoder'](wm.rssm.get_feat(elite_seq))['observation'].mode()
      goal_states = tf.repeat(elite_samples[None, :num_vis], elite_obs.shape[0], axis=0).numpy() # T x topk x 2
      goal_list = goal_states[...,:2]
      goal_list = tf.reshape(goal_list, [-1, 2])

      fig, p2evalue_ax = plt.subplots(1, 1, figsize=(1, 3))
      p2evalue_ax.scatter(
        x=goal_list[:,0],
        y=goal_list[:,1],
        s=1,
        c='r',
        zorder=5,
      )
      elite_obs = tf.transpose(elite_obs, (1,0,2))
      # (num_vis,horizon,29)
      first_half = elite_obs[:, :-10]
      first_half = first_half[:, ::10]
      second_half = elite_obs[:, -10:]
      traj = tf.concat([first_half, second_half], axis=1)
      p2evalue_ax.plot(
          traj[:,:,0],
          traj[:,:,1],
          c='b',
          zorder=4,
          marker='.'
      )

      # plt.colorbar(p2e_scatter, ax=p2evalue_ax)
      if 'hard' in config.task:
        p2evalue_ax.set(xlim=(-1, 5.25), ylim=(-1, 9.25))
      else:
        p2evalue_ax.set(xlim=(-1, 5.25), ylim=(-1, 5.25))
      p2evalue_ax.set_title('elite goals and states')
      fig = plt.gcf()
      fig.set_size_inches(7, 6)
      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image(f'top_{num_vis}_cem', image_from_plot)
      logger.write()

  return vis_fn

def make_plot_fn(config):
  make_plot = None
  if 'dmc_walker_walk_proprio' == config.task:
    def make_plot(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50, obs_key = 'qpos', goal_key='goal'):
    # 1. Load all episodes
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      num_goals = min(int(config.eval_every), len(episodes))
      recent_episodes = episodes[-num_goals:]
      if len(episodes) > num_goals:
        old_episodes = episodes[:-num_goals][::5]
        old_episodes.extend(recent_episodes)
        episodes = old_episodes
      else:
        episodes = recent_episodes

      obs = []
      value_list = []
      goals = []
      for ep_count, episode in enumerate(episodes):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
          end = ep_count
          chunk = {k: tf.stack(v) for k, v in chunk.items()}
          embed = wm.encoder(chunk)
          post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
          chunk['feat'] = wm.rssm.get_feat(post)
          value_fn = agnt._expl_behavior.ac._target_critic
          value = value_fn(chunk['feat']).mode()
          value_list.append(value)

          obs.append(tf.stack(chunk[obs_key]))
          goals.append(tf.stack(chunk[goal_key]))
    # 4. Plotting
      fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(1, 3, figsize=(1, 3))
      xlim = np.array([-20.0, 20.0])
      ylim = np.array([-1.3, 1.0])

      state_ax.set(xlim=xlim, ylim=ylim)
      p2evalue_ax.set(xlim=xlim, ylim=ylim)
      goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)
      obs_list = tf.concat(obs, axis = 0)
      before = obs_list[:,:goal_time_limit,:]
      before = before[:,:,:2]
      ep_order_before = tf.range(before.shape[0])[:, None]
      ep_order_before = tf.repeat(ep_order_before, before.shape[1], axis=1)
      before = tf.reshape(before, [before.shape[0]*before.shape[1], 2])
      after = obs_list[:,goal_time_limit:,:]
      after = after[:,:,:2]
      ep_order_after = tf.range(after.shape[0])[:, None]
      ep_order_after = tf.repeat(ep_order_after, after.shape[1], axis=1)
      after = tf.reshape(after, [after.shape[0]*after.shape[1], 2])
      # obs_list = tf.concat(obs, axis = 0)
      # obs_list = obs_list[:,:,:2]
      # ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
      # ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #  Num_ep x T
      # obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
      # ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
      ep_order_before = tf.reshape(ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
      ep_order_after = tf.reshape(ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
      goal_list = tf.concat(goals, axis = 0)[:, 0, :2]
      goal_list = tf.reshape(goal_list, [-1, 2])
      # plt.scatter(
      #     x=obs_list[:,0],
      #     y=obs_list[:,1],
      #     s=1,
      #     c=ep_order,
      #     cmap='Blues',
      #     zorder=3,
      #     )
      state_ax.scatter(
          y=before[:,0],
          x=before[:,1],
          s=1,
          c=ep_order_before,
          cmap='Blues',
          zorder=3,
          )
      state_ax.scatter(
          y=after[:,0],
          x=after[:,1],
          s=1,
          c=ep_order_after,
          cmap='Greens',
          zorder=3,
          )
      state_ax.scatter(
          y=goal_list[:,0],
          x=goal_list[:,1],
          s=1,
          c=np.arange(goal_list.shape[0]),
          cmap='Reds',
          zorder=3,
          )

      x_min, x_max = xlim[0], xlim[1]
      y_min, y_max = ylim[0], ylim[1]
      x_div = y_div = 100
      other_dims = np.array([ 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1])
      gx = 5.0
      gy = 0.0

      x = np.linspace(x_min, x_max, x_div)
      y = np.linspace(y_min, y_max, y_div)
      XY = X, Y = np.meshgrid(y, x)
      XY = np.stack([X, Y], axis=-1)
      XY = XY.reshape(x_div * y_div, 2)
      XY_plus = np.hstack((XY, np.tile(other_dims, (XY.shape[0], 1))))
      # swap first and second element
      # import ipdb; ipdb.set_trace()


      goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
      goal_vec[:,0] = goal_vec[:,0] + gy
      goal_vec[:,1] = goal_vec[:,1] + gx
      goal_vec[:,2:] = goal_vec[:,2:] + other_dims

      obs = {"qpos": XY_plus, "goal": goal_vec, "reward": np.zeros(XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
      temporal_dist = agnt.temporal_dist(obs)
      if config.gc_reward == 'dynamical_distance':
        td_plot = dd_ax.tricontourf(XY[:, 1], XY[:, 0], temporal_dist)
        dd_ax.scatter(y = obs['goal'][0][0], x = obs['goal'][0][1], c="r", marker="*", s=20, zorder=2)
        dd_ax.scatter(y = before[0][0], x = before[0][1], c="b", marker=".", s=20, zorder=2)
        plt.colorbar(td_plot, ax=dd_ax)
        dd_ax.set_title('temporal distance')

      obs_list = obs_list[:, :, :2]
      obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
      values = tf.concat(value_list, axis = 0)
      values = values.numpy().flatten()
      cm = plt.cm.get_cmap("viridis")
      p2e_scatter = p2evalue_ax.scatter(
        y=obs_list[:,0],
        x=obs_list[:,1],
        s=1,
        c=values,
        cmap=cm,
        zorder=3,
      )
      plt.colorbar(p2e_scatter, ax=p2evalue_ax)
      p2evalue_ax.set_title('p2e value')

      fig = plt.gcf()
      fig.set_size_inches(10, 3)
      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
      plt.cla()
      plt.clf()
  elif 'dmc_humanoid_walk_proprio' == config.task:
    def make_plot(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50, obs_key = 'qpos', goal_key='goal'):
    # 1. Load all episodes
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      num_goals = min(int(config.eval_every), len(episodes))
      recent_episodes = episodes[-num_goals:]
      if len(episodes) > num_goals:
        old_episodes = episodes[:-num_goals][::5]
        old_episodes.extend(recent_episodes)
        episodes = old_episodes
      else:
        episodes = recent_episodes

      obs = []
      value_list = []
      goals = []
      for ep_count, episode in enumerate(episodes):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
          end = ep_count
          chunk = {k: tf.stack(v) for k, v in chunk.items()}
          embed = wm.encoder(chunk)
          post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
          chunk['feat'] = wm.rssm.get_feat(post)
          value_fn = agnt._expl_behavior.ac._target_critic
          value = value_fn(chunk['feat']).mode()
          value_list.append(value)

          obs.append(tf.stack(chunk[obs_key]))
          goals.append(tf.stack(chunk[goal_key]))
    # 4. Plotting
      fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(1, 3, figsize=(1, 3))
      xlim = np.array([-0.2, 1.2])
      ylim = np.array([-0.2, 1.2])

      state_ax.set(xlim=xlim, ylim=ylim)
      p2evalue_ax.set(xlim=xlim, ylim=ylim)
      goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)
      obs_list = tf.concat(obs, axis = 0)
      before = obs_list[:,:goal_time_limit,:]
      before = before[:,:,:28]
      ep_order_before = tf.range(before.shape[0])[:, None]
      ep_order_before = tf.repeat(ep_order_before, before.shape[1], axis=1)
      before = tf.reshape(before, [before.shape[0]*before.shape[1], 28])
      after = obs_list[:,goal_time_limit:,:]
      after = after[:,:,:28]
      ep_order_after = tf.range(after.shape[0])[:, None]
      ep_order_after = tf.repeat(ep_order_after, after.shape[1], axis=1)
      after = tf.reshape(after, [after.shape[0]*after.shape[1], 28])
      # obs_list = tf.concat(obs, axis = 0)
      # obs_list = obs_list[:,:,:2]
      # ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
      # ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #  Num_ep x T
      # obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
      # ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
      ep_order_before = tf.reshape(ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
      ep_order_after = tf.reshape(ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
      goal_list = tf.concat(goals, axis = 0)[:, 0, :28]
      goal_list = tf.reshape(goal_list, [-1, 28])
      # plt.scatter(
      #     x=obs_list[:,0],
      #     y=obs_list[:,1],
      #     s=1,
      #     c=ep_order,
      #     cmap='Blues',
      #     zorder=3,
      #     )
      state_ax.scatter(
          y=before[:,0],
          x=before[:,1],
          s=1,
          c=ep_order_before,
          cmap='Blues',
          zorder=3,
          )
      state_ax.scatter(
          y=after[:,0],
          x=after[:,1],
          s=1,
          c=ep_order_after,
          cmap='Greens',
          zorder=3,
          )
      state_ax.scatter(
          y=goal_list[:,0],
          x=goal_list[:,1],
          s=1,
          c=np.arange(goal_list.shape[0]),
          cmap='Reds',
          zorder=3,
          )

      # x_min, x_max = xlim[0], xlim[1]
      # y_min, y_max = ylim[0], ylim[1]
      # x_div = y_div = 100
      # other_dims = np.array([ 0.34, 0.74, -1.34, -0., 1.1, -0.66, -0.1])
      # gx = 5.0
      # gy = 0.0

      # x = np.linspace(x_min, x_max, x_div)
      # y = np.linspace(y_min, y_max, y_div)
      # XY = X, Y = np.meshgrid(y, x)
      # XY = np.stack([X, Y], axis=-1)
      # XY = XY.reshape(x_div * y_div, 2)
      # XY_plus = np.hstack((XY, np.tile(other_dims, (XY.shape[0], 1))))
      # swap first and second element
      # import ipdb; ipdb.set_trace()


      # goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
      # goal_vec[:,0] = goal_vec[:,0] + gy
      # goal_vec[:,1] = goal_vec[:,1] + gx
      # goal_vec[:,2:] = goal_vec[:,2:] + other_dims

      # obs = {"qpos": XY_plus, "goal": goal_vec, "reward": np.zeros(XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
      # temporal_dist = agnt.temporal_dist(obs)
      # if config.gc_reward == 'dynamical_distance':
      #   td_plot = dd_ax.tricontourf(XY[:, 1], XY[:, 0], temporal_dist)
      #   dd_ax.scatter(y = obs['goal'][0][0], x = obs['goal'][0][1], c="r", marker="*", s=20, zorder=2)
      #   dd_ax.scatter(y = before[0][0], x = before[0][1], c="b", marker=".", s=20, zorder=2)
      #   plt.colorbar(td_plot, ax=dd_ax)
      #   dd_ax.set_title('temporal distance')

      obs_list = obs_list[:, :, :28]
      obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 28])
      values = tf.concat(value_list, axis = 0)
      values = values.numpy().flatten()
      cm = plt.cm.get_cmap("viridis")
      p2e_scatter = p2evalue_ax.scatter(
        y=obs_list[:,0],
        x=obs_list[:,1],
        s=1,
        c=values,
        cmap=cm,
        zorder=3,
      )
      plt.colorbar(p2e_scatter, ax=p2evalue_ax)
      p2evalue_ax.set_title('p2e value')

      fig = plt.gcf()
      fig.set_size_inches(10, 3)
      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
      plt.cla()
      plt.clf()
  elif 'pointmaze' in config.task:
    def make_plot(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 100, obs_key = 'observation', goal_key='goal'):
    # 1. Load all episodes
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      obs = []
      goals = []
      reward_list = []
      for ep_count, episode in enumerate(episodes[::ep_subsample]):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes[::ep_subsample]) - 1):
          end = ep_count
          # chunk = {k: tf.stack(v) for k, v in chunk.items()}
          obs.append(tf.stack(chunk[obs_key]))
          goals.append(tf.stack(chunk[goal_key]))
    # 4. Plotting
      fig, ax = plt.subplots(1, 1, figsize=(1, 1))
      ax.set(xlim=(-1, 11), ylim=(-1, 11))
      maze.maze.plot(ax) # plot the walls
      obs_list = tf.concat(obs, axis = 0)
      ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
      ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #  Num_ep x T
      obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
      ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
      goal_list = tf.concat(goals, axis = 0)[:, 0, :]
      goal_list = tf.reshape(goal_list, [-1, 2])
      plt.scatter(
          x=obs_list[:,0],
          y=obs_list[:,1],
          s=1,
          c=ep_order,
          cmap='Blues',
          zorder=3,
          )
      plt.scatter(
          x=goal_list[:,0],
          y=goal_list[:,1],
          s=1,
          c=np.arange(goal_list.shape[0]),
          cmap='Reds',
          zorder=3,
          )
      fig = plt.gcf()
      plt.title('states')
      fig.set_size_inches(8, 6)
      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
  elif config.task in {'fetchpnp', 'fetchpnpeasy'}:
    def make_plot(env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50, obs_key = 'observation', goal_key='goal'):
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      num_goals = min(int(config.eval_every), len(episodes))
      recent_episodes = episodes[-num_goals:]
      if len(episodes) > num_goals:
        old_episodes = episodes[:-num_goals][::5]
        old_episodes.extend(recent_episodes)
        episodes = old_episodes
      else:
        episodes = recent_episodes

      all_observations = []
      value_list = []
      all_goals = []
      for ep_count, episode in enumerate(episodes):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
          end = ep_count
          chunk = {k: tf.stack(v) for k, v in chunk.items()}
          embed = wm.encoder(chunk)
          post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
          chunk['feat'] = wm.rssm.get_feat(post)
          value_fn = agnt._expl_behavior.ac._target_critic
          value = value_fn(chunk['feat']).mode()
          value_list.append(value)

          all_observations.append(tf.stack(chunk[obs_key]))
          all_goals.append(tf.stack(chunk[goal_key]))

      all_observations = np.concatenate(all_observations)
      all_observations = all_observations.reshape(-1, all_observations.shape[-1])
      ob_obj_pos = all_observations[:,:3]
      # ob_grip_pos = all_observations[:, 3:6]

      all_goals = np.concatenate(all_goals)[:, 0]
      g_obj_pos = all_goals[:,:3]
      g_grip_pos = all_goals[:, 3:6]

      plot_dims = [[0, 2]]
      plot_dim_name = dict([(0,'x'), (1,'y'), (2,'z')])
      def plot_axes(axes, data, cmap, title, zorder):
        for ax, pd in zip(axes, plot_dims):
          ax.scatter(x=data[:, pd[0]],
            y=data[:, pd[1]],
            s=1,
            c=np.arange(len(data)),
            cmap=cmap,
            zorder=zorder,
          )
          ax.set_title(f"{title} {plot_dim_name[pd[0]]}{plot_dim_name[pd[1]]}", fontdict={'fontsize':10})

      fig, (g_ax, ob_ax, rew_ax, p2evalue_ax) = plt.subplots(1,4, figsize=(13,3))
      plot_axes([ob_ax], ob_obj_pos, 'Blues',f"State ", 3)
      # plot_axes([ob_ax], ob_grip_pos, 'Reds',f"State ", 2)
      plot_axes([g_ax], g_obj_pos, 'Blues',f"Goal ", 3)
      if g_grip_pos.shape[-1] != 0:
        plot_axes([g_ax], g_grip_pos, 'Reds',f"Goal ", 3)

      # Plot temporal distance reward.
      x_min, x_max = 1.0, 1.6
      y_min, y_max = 0.3, 1.0
      x_div = y_div = 100
      x = np.linspace(x_min, x_max, x_div)
      y = np.linspace(y_min, y_max, y_div)
      XZ = X, Z = np.meshgrid(x, y)
      XZ = np.stack([X, Z], axis=-1)
      XZ = XZ.reshape(x_div * y_div, 2)

      start_pos = np.array([1.3, 0.65, 0.41])
      goal_pos = start_pos + np.array([0, 0, 0.4])
      goal_vec = np.zeros((x_div*y_div, 3))
      goal_vec[:,0] = goal_pos[0]
      goal_vec[:,1] = goal_pos[1]
      goal_vec[:,2] = goal_pos[2]

      observation = np.zeros((x_div*y_div, 3))
      observation[:, 0] = XZ[:, 0]
      observation[:, 1] = goal_pos[1]
      observation[:, 2] = XZ[:, 1]

      obs = {"observation": observation, "goal": goal_vec, "reward": np.zeros(len(XZ)), "discount": np.ones(len(XZ)), "is_terminal": np.zeros(len(XZ))}
      temporal_dist = agnt.temporal_dist(obs)
      if config.gc_reward == 'dynamical_distance':
        im = rew_ax.tricontourf(XZ[:, 0], XZ[:, 1], temporal_dist, zorder=1)
        rew_ax.scatter(x=[goal_pos[0]], y=[goal_pos[2]], c="r", marker="*", s=20, zorder=2)
        rew_ax.scatter(x=[start_pos[0]], y=[start_pos[2]], c="b", marker=".", s=20, zorder=2)
        plt.colorbar(im, ax=rew_ax)

      g_ax.set_xlim([1, 1.6]) # obj x axis
      g_ax.set_ylim([0.3, 1.0]) # obj z axis
      ob_ax.set_xlim([1, 1.6]) # obj x axis
      ob_ax.set_ylim([0.3, 1.0]) # obj z axis
      rew_ax.set_xlim([1, 1.6]) # obj x axis
      rew_ax.set_ylim([0.3, 1.0]) # obj z axis
      p2evalue_ax.set_xlim([1, 1.6]) # obj x axis
      p2evalue_ax.set_ylim([0.3, 1.0]) # obj z axis


      # plot p2e value function
      values = tf.concat(value_list, axis = 0)
      values = values.numpy().flatten()
      cm = plt.cm.get_cmap("viridis")
      p2e_scatter = p2evalue_ax.scatter(
        x=ob_obj_pos[:,plot_dims[0][0]],
        y=ob_obj_pos[:,plot_dims[0][1]],
        s=1,
        c=values,
        cmap=cm,
        zorder=3,
      )
      plt.colorbar(p2e_scatter, ax=p2evalue_ax)
      p2evalue_ax.set_title('p2e value')


      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
      plt.cla()
      plt.clf()

  elif 'kitchen' in config.task:
    def make_plot(env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50, obs_key = 'state', goal_key='goal'):
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      num_goals = min(int(config.eval_every), len(episodes))
      recent_episodes = episodes[-num_goals:]
      if len(episodes) > num_goals:
        old_episodes = episodes[:-num_goals][::5]
        old_episodes.extend(recent_episodes)
        episodes = old_episodes
      else:
        episodes = recent_episodes

      all_observations = []
      value_list = []
      all_goals = []

      # def unnorm_ob(ob):
      #   return env.obs_min + ob * (env.obs_max -  env.obs_min)

      for ep_count, episode in enumerate(episodes):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        # sequence['goal'] = unnorm_ob(sequence['goal'])
        # sequence['state'] = unnorm_ob(sequence['state'])
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
          end = ep_count
          chunk = {k: tf.stack(v) for k, v in chunk.items()}
          embed = wm.encoder(chunk)
          post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
          chunk['feat'] = wm.rssm.get_feat(post)
          value_fn = agnt._expl_behavior.ac._target_critic
          value = value_fn(chunk['feat']).mode()
          value_list.append(value)

          all_observations.append(tf.stack(chunk[obs_key]))
          all_goals.append(tf.stack(chunk[goal_key]))

      all_observations = np.concatenate(all_observations)
      all_observations = all_observations.reshape(-1, all_observations.shape[-1])
      # all_observations = unnorm_ob(all_observations)
      all_goals = np.concatenate(all_goals)[:, 0]
      # all_goals = unnorm_ob(all_goals)
      fig, all_axes = plt.subplots(3, 5, figsize=(6,3))
      state_axes = all_axes[0]
      value_axes = all_axes[1]
      goal_axes = all_axes[2]

      obj_to_ax = {
        "bottom_burner": (state_axes[0], value_axes[0], goal_axes[0]),
        "light_switch": (state_axes[1], value_axes[1], goal_axes[1]),
        "slide_cabinet": (state_axes[2], value_axes[2], goal_axes[2]),
        "hinge_cabinet": (state_axes[3], value_axes[3], goal_axes[3]),
        "microwave": (state_axes[2], value_axes[2], goal_axes[2]),
        "kettle": (state_axes[4], value_axes[4], goal_axes[4]),
      }
      object_obs_idxs = {'bottom_burner' :  [9, 10],
                      'light_switch' :  [11, 12],
                      'slide_cabinet':  [13],
                      'hinge_cabinet':  [14, 15],
                      'microwave'    :  [16],
                      'kettle'       :  [17, 18, 19]}
      # plot state occupancy
      for obj, axs in obj_to_ax.items():
        ax = axs[0]
        obs_idxs = object_obs_idxs[obj]
        color = 'Reds'
        ax.set_title(obj, fontsize=6)
        if obj == "kettle": # only plot xy dims.
          data = all_observations[:, obs_idxs[:2]]
        elif obj in {"microwave", "slide_cabinet"}: # plot both 1D lines on same plot.
          ax.set_title("microwave, slide_cabinet", fontsize=6)
          y = 0.25 if obj == "microwave" else 0.75
          color = 'Blues' if obj == "microwave" else 'Reds'
          x = all_observations[:, obs_idxs]
          y = np.ones_like(x) * y
          data = np.hstack([x,y])
        else:
          data = all_observations[:, obs_idxs]
        ax.scatter(x=data[:, 0],
          y=data[:, 1],
          s=1,
          c=np.arange(len(data)),
          cmap=color,
        )
        ax.set_xlim([-2.5, 2.5]) # assume obs are normalized.
        ax.set_ylim([-2.5, 2.5])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

      # plot value
      values = tf.concat(value_list, axis = 0)
      values = values.numpy().flatten()
      cm = plt.cm.get_cmap("viridis")
      for obj, axs in obj_to_ax.items():
        ax = axs[1]
        obs_idxs = object_obs_idxs[obj]
        if obj == "kettle": # only plot xy dims.
          data = all_observations[:, obs_idxs[:2]]
        elif obj in {"microwave", "slide_cabinet"}: # plot both 1D lines on same plot.
          y = 0.25 if obj == "microwave" else 0.75
          x = all_observations[:, obs_idxs]
          y = np.ones_like(x) * y
          data = np.hstack([x,y])
        else:
          data = all_observations[:, obs_idxs]
        p2e_scatter = ax.scatter(x=data[:, 0],
          y=data[:, 1],
          s=1,
          c=values,
          cmap=cm,
        )
        ax.set_xlim([-2.5, 2.5]) # assume obs are normalized.
        ax.set_ylim([-2.5, 2.5])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
      plt.colorbar(p2e_scatter, ax=ax)

      # plot goal
      for obj, axs in obj_to_ax.items():
        ax = axs[2]
        obs_idxs = object_obs_idxs[obj]
        color = 'Reds'
        if obj == "kettle": # only plot xy dims.
          data = all_goals[:, obs_idxs[:2]]
        elif obj in {"microwave", "slide_cabinet"}: # plot both 1D lines on same plot.
          y = 0.25 if obj == "microwave" else 0.75
          color = 'Blues' if obj == "microwave" else 'Reds'
          x = all_goals[:, obs_idxs]
          y = np.ones_like(x) * y
          data = np.hstack([x,y])
        else:
          data = all_goals[:, obs_idxs]
        ax.scatter(x=data[:, 0],
          y=data[:, 1],
          s=1,
          c=np.arange(len(data)),
          cmap=color,
        )
        ax.set_xlim([-2.5, 2.5]) # assume obs are normalized.
        ax.set_ylim([-2.5, 2.5])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)


      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
      plt.cla()
      plt.clf()

  elif 'demofetchpnp' in config.task:
    def make_plot(env, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50, obs_key = 'observation', goal_key='goal'):
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      num_goals = min(int(config.eval_every), len(episodes))
      recent_episodes = episodes[-num_goals:]
      if len(episodes) > num_goals:
        old_episodes = episodes[:-num_goals][::5]
        old_episodes.extend(recent_episodes)
        episodes = old_episodes
      else:
        episodes = recent_episodes

      all_observations = []
      value_list = []
      all_goals = []

      def unnorm_ob(ob):
        return env.obs_min + ob * (env.obs_max -  env.obs_min)

      for ep_count, episode in enumerate(episodes):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
          end = ep_count
          chunk = {k: tf.stack(v) for k, v in chunk.items()}
          embed = wm.encoder(chunk)
          post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
          chunk['feat'] = wm.rssm.get_feat(post)
          value_fn = agnt._expl_behavior.ac._target_critic
          value = value_fn(chunk['feat']).mode()
          value_list.append(value)

          all_observations.append(tf.stack(chunk[obs_key]))
          all_goals.append(tf.stack(chunk[goal_key]))

      all_observations = np.concatenate(all_observations)
      all_observations = all_observations.reshape(-1, all_observations.shape[-1])
      all_observations = unnorm_ob(all_observations)
      all_obj_obs_pos = np.split(all_observations[:, 5:5+3*env.n], env.n, axis=1)

      all_goals = np.concatenate(all_goals)[:, 0]
      all_goals = unnorm_ob(all_goals)
      all_obj_g_pos = np.split(all_goals[:, 5:5+3*env.n], env.n, axis=1)

      plot_dims = [[1, 2]]
      plot_dim_name = dict([(0,'x'), (1,'y'), (2,'z')])
      def plot_axes(axes, data, cmap, title, zorder):
        for ax, pd in zip(axes, plot_dims):
          ax.scatter(x=data[:, pd[0]],
            y=data[:, pd[1]],
            s=1,
            c=np.arange(len(data)),
            cmap=cmap,
            zorder=zorder,
          )
          ax.set_title(f"{title} {plot_dim_name[pd[0]]}{plot_dim_name[pd[1]]}", fontdict={'fontsize':10})

      fig, all_axes = plt.subplots(1,2+env.n, figsize=(1+(2+env.n*3),2))

      g_ax = all_axes[0]
      p2evalue_ax = all_axes[-1]
      obj_axes = all_axes[1:-1]
      obj_colors = ['Reds', 'Blues', 'Greens']
      for obj_ax, obj_pos, obj_g_pos, obj_color in zip(obj_axes, all_obj_obs_pos, all_obj_g_pos, obj_colors):
        plot_axes([obj_ax], obj_pos, obj_color, f"State ", 3)
        plot_axes([g_ax], obj_g_pos, obj_color, f"Goal ", 3)

      # Plot temporal distance reward.
      # x_min, x_max = 1.2, 1.65
      # y_min, y_max = 0.3, 0.7
      # x_div = y_div = 100
      # x = np.linspace(x_min, x_max, x_div)
      # y = np.linspace(y_min, y_max, y_div)
      # XZ = X, Z = np.meshgrid(x, y)
      # XZ = np.stack([X, Z], axis=-1)
      # XZ = XZ.reshape(x_div * y_div, 2)

      # start_pos = np.array([1.3, 0.65, 0.41])
      # goal_pos = start_pos + np.array([0, 0, 0.4])
      # goal_vec = np.zeros((x_div*y_div, 3))
      # goal_vec[:,0] = goal_pos[0]
      # goal_vec[:,1] = goal_pos[1]
      # goal_vec[:,2] = goal_pos[2]

      # observation = np.zeros((x_div*y_div, 3))
      # observation[:, 0] = XZ[:, 0]
      # observation[:, 1] = goal_pos[1]
      # observation[:, 2] = XZ[:, 1]

      # obs = {"observation": observation, "goal": goal_vec, "reward": np.zeros(len(XZ)), "discount": np.ones(len(XZ)), "is_terminal": np.zeros(len(XZ))}
      # temporal_dist = agnt.temporal_dist(obs)
      # if config.gc_reward == 'dynamical_distance':
      #   im = rew_ax.tricontourf(XZ[:, 0], XZ[:, 1], temporal_dist, zorder=1)
      #   rew_ax.scatter(x=[goal_pos[0]], y=[goal_pos[2]], c="r", marker="*", s=20, zorder=2)
      #   rew_ax.scatter(x=[start_pos[0]], y=[start_pos[2]], c="b", marker=".", s=20, zorder=2)
      #   plt.colorbar(im, ax=rew_ax)
      limits = [[0.5, 1.0], [0.4, 0.6]] if 'walls' in config.task else [[1, 1.6], [0.3, 0.7]]
      for _ax in all_axes:
        _ax.set_xlim(limits[0])
        _ax.set_ylim(limits[1])
        _ax.axes.get_yaxis().set_visible(False)

      # plot p2e value function
      values = tf.concat(value_list, axis = 0)
      values = values.numpy().flatten()
      cm = plt.cm.get_cmap("viridis")
      for obj_ax, obj_pos, obj_color in zip(obj_axes, all_obj_obs_pos, obj_colors):
        p2e_scatter = p2evalue_ax.scatter(
          x=obj_pos[:,plot_dims[0][0]],
          y=obj_pos[:,plot_dims[0][1]],
          s=1,
          c=values,
          cmap=cm,
          zorder=3,
        )
      plt.colorbar(p2e_scatter, ax=p2evalue_ax)
      p2evalue_ax.set_title('p2e value')

      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
      plt.cla()
      plt.clf()


  elif 'umazefull' == config.task:
    def make_plot(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50, obs_key = 'observation', goal_key='goal'):
    # 1. Load all episodes
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      obs = []
      goals = []
      reward_list = []
      for ep_count, episode in enumerate(episodes[::ep_subsample]):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes[::ep_subsample]) - 1):
          end = ep_count
          # chunk = {k: tf.stack(v) for k, v in chunk.items()}
          obs.append(tf.stack(chunk[obs_key]))
          goals.append(tf.stack(chunk[goal_key]))
    # 4. Plotting
      fig, ax = plt.subplots(1, 1, figsize=(1, 1))
      ax.set(xlim=(-3, 21), ylim=(-3, 21))
      obs_list = tf.concat(obs, axis = 0)
      obs_list = obs_list[:,:,:2]
      ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
      ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #  Num_ep x T
      obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
      ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
      goal_list = tf.concat(goals, axis = 0)[:, 0, :2]
      goal_list = tf.reshape(goal_list, [-1, 2])
      plt.scatter(
          x=obs_list[:,0],
          y=obs_list[:,1],
          s=1,
          c=ep_order,
          cmap='Blues',
          zorder=3,
          )
      plt.scatter(
          x=goal_list[:,0],
          y=goal_list[:,1],
          s=1,
          c=np.arange(goal_list.shape[0]),
          cmap='Reds',
          zorder=3,
          )
      fig = plt.gcf()
      plt.title('states')
      fig.set_size_inches(8, 6)
      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
  elif config.task in {'umazefulldownscale','a1umazefulldownscale', 'hardumazefulldownscale'}:
    def make_plot(maze, agnt, complete_episodes, logger, ep_subsample=5, step_subsample=1, batch_size = 50, obs_key = 'observation', goal_key='goal'):
    # 1. Load all episodes
      wm = agnt.wm
      episodes = list(complete_episodes.values())
      num_goals = min(int(config.eval_every), len(episodes))
      recent_episodes = episodes[-num_goals:]
      if len(episodes) > num_goals:
        old_episodes = episodes[:-num_goals][::5]
        old_episodes.extend(recent_episodes)
        episodes = old_episodes
      else:
        episodes = recent_episodes

      obs = []
      value_list = []
      goals = []
      for ep_count, episode in enumerate(episodes):
        # 2. Adding episodes to the batch
        if (ep_count % batch_size) == 0:
          start = ep_count
          chunk = collections.defaultdict(list)
        sequence = {
          k: convert(v[::step_subsample])
          for k, v in episode.items() if not k.startswith('log_')}
        data = wm.preprocess(sequence)
        for key, value in data.items():
          chunk[key].append(value)
        # 3. Forward passing each batch after it reaches size batch_size or it reaches the end of the episode list
        if (ep_count % batch_size == (batch_size - 1) or ep_count == len(episodes) - 1):
          end = ep_count
          chunk = {k: tf.stack(v) for k, v in chunk.items()}
          embed = wm.encoder(chunk)
          post, _ = wm.rssm.observe(embed, chunk['action'], chunk['is_first'], None)
          chunk['feat'] = wm.rssm.get_feat(post)
          value_fn = agnt._expl_behavior.ac._target_critic
          value = value_fn(chunk['feat']).mode()
          value_list.append(value)

          obs.append(tf.stack(chunk[obs_key]))
          goals.append(tf.stack(chunk[goal_key]))
    # 4. Plotting
      fig, (state_ax, dd_ax, p2evalue_ax) = plt.subplots(1, 3, figsize=(1, 3))
      xlim = np.array([-1, 5.25])
      ylim = np.array([-1, 5.25])
      if config.task == 'a1umazefulldownscale':
        xlim /= 2.0
        ylim /= 2.0
      elif config.task == 'hardumazefulldownscale':
        xlim = np.array([-1, 5.25])
        ylim = np.array([-1, 9.25])

      state_ax.set(xlim=xlim, ylim=ylim)
      p2evalue_ax.set(xlim=xlim, ylim=ylim)
      goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)
      obs_list = tf.concat(obs, axis = 0)
      before = obs_list[:,:goal_time_limit,:]
      before = before[:,:,:2]
      ep_order_before = tf.range(before.shape[0])[:, None]
      ep_order_before = tf.repeat(ep_order_before, before.shape[1], axis=1)
      before = tf.reshape(before, [before.shape[0]*before.shape[1], 2])
      after = obs_list[:,goal_time_limit:,:]
      after = after[:,:,:2]
      ep_order_after = tf.range(after.shape[0])[:, None]
      ep_order_after = tf.repeat(ep_order_after, after.shape[1], axis=1)
      after = tf.reshape(after, [after.shape[0]*after.shape[1], 2])
      # obs_list = tf.concat(obs, axis = 0)
      # obs_list = obs_list[:,:,:2]
      # ep_order = tf.range(obs_list.shape[0])[:, None] # Num_ep x 1
      # ep_order = tf.repeat(ep_order, obs_list.shape[1], axis=1) #  Num_ep x T
      # obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
      # ep_order = tf.reshape(ep_order, [ep_order.shape[0]*ep_order.shape[1]])
      ep_order_before = tf.reshape(ep_order_before, [ep_order_before.shape[0]*ep_order_before.shape[1]])
      ep_order_after = tf.reshape(ep_order_after, [ep_order_after.shape[0]*ep_order_after.shape[1]])
      goal_list = tf.concat(goals, axis = 0)[:, 0, :2]
      goal_list = tf.reshape(goal_list, [-1, 2])
      # plt.scatter(
      #     x=obs_list[:,0],
      #     y=obs_list[:,1],
      #     s=1,
      #     c=ep_order,
      #     cmap='Blues',
      #     zorder=3,
      #     )
      state_ax.scatter(
          x=before[:,0],
          y=before[:,1],
          s=1,
          c=ep_order_before,
          cmap='Blues',
          zorder=3,
          )
      state_ax.scatter(
          x=after[:,0],
          y=after[:,1],
          s=1,
          c=ep_order_after,
          cmap='Greens',
          zorder=3,
          )
      state_ax.scatter(
          x=goal_list[:,0],
          y=goal_list[:,1],
          s=1,
          c=np.arange(goal_list.shape[0]),
          cmap='Reds',
          zorder=3,
          )
      x_min, x_max = xlim[0], xlim[1]
      y_min, y_max = ylim[0], ylim[1]
      x_div = y_div = 100
      if config.task in {'umazefulldownscale', 'hardumazefulldownscale'}:
        other_dims = np.concatenate([[6.08193526e-01,  9.87496030e-01,
        1.82685311e-03, -6.82827458e-03,  1.57485326e-01,  5.14617396e-02,
        1.22386603e+00, -6.58701813e-02, -1.06980319e+00,  5.09069276e-01,
        -1.15506861e+00,  5.25953435e-01,  7.11716520e-01], np.zeros(14)])
      elif config.task == 'a1umazefulldownscale':
        other_dims = np.concatenate([[0.24556014,  0.986648,    0.09023235, -0.09100603,
          0.10050705, -0.07250207, -0.01489305,  0.09989551, -0.05246516, -0.05311238,
          -0.01864055, -0.05934234,  0.03910208, -0.08356607,  0.05515265, -0.00453086,
          -0.01196933], np.zeros(18)])
      gx = 0.0
      gy = 4.2
      if config.task == 'a1umazefulldownscale':
        gx /= 2
        gy /= 2
      elif config.task == 'hardumazefulldownscale':
        gx = 4.2
        gy = 8.2

      x = np.linspace(x_min, x_max, x_div)
      y = np.linspace(y_min, y_max, y_div)
      XY = X, Y = np.meshgrid(x, y)
      XY = np.stack([X, Y], axis=-1)
      XY = XY.reshape(x_div * y_div, 2)
      XY_plus = np.hstack((XY, np.tile(other_dims, (XY.shape[0], 1))))
      goal_vec = np.zeros((x_div*y_div, XY_plus.shape[-1]))
      goal_vec[:,0] = goal_vec[:,0] + gx
      goal_vec[:,1] = goal_vec[:,1] + gy
      goal_vec[:,2:] = goal_vec[:,2:] + other_dims
      obs = {"observation": XY_plus, "goal": goal_vec, "reward": np.zeros(XY.shape[0]), "discount": np.ones(XY.shape[0]), "is_terminal": np.zeros(XY.shape[0])}
      temporal_dist = agnt.temporal_dist(obs)
      if config.gc_reward == 'dynamical_distance':
        td_plot = dd_ax.tricontourf(XY[:, 0], XY[:, 1], temporal_dist)
        dd_ax.scatter(x = obs['goal'][0][0], y = obs['goal'][0][1], c="r", marker="*", s=20, zorder=2)
        dd_ax.scatter(x = before[0][0], y = before[0][1], c="b", marker=".", s=20, zorder=2)
        plt.colorbar(td_plot, ax=dd_ax)
        dd_ax.set_title('temporal distance')

      obs_list = obs_list[:, :, :2]
      obs_list = tf.reshape(obs_list, [obs_list.shape[0]*obs_list.shape[1], 2])
      values = tf.concat(value_list, axis = 0)
      values = values.numpy().flatten()
      cm = plt.cm.get_cmap("viridis")
      p2e_scatter = p2evalue_ax.scatter(
        x=obs_list[:,0],
        y=obs_list[:,1],
        s=1,
        c=values,
        cmap=cm,
        zorder=3,
      )
      plt.colorbar(p2e_scatter, ax=p2evalue_ax)
      p2evalue_ax.set_title('p2e value')

      fig = plt.gcf()
      fig.set_size_inches(10, 3)
      fig.canvas.draw()
      image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      image_from_plot = np.expand_dims(image_from_plot, axis = 0)
      logger.image('state_occupancy', image_from_plot)
      plt.cla()
      plt.clf()
  return make_plot

def make_obs2goal_fn(config):
  obs2goal = None
  if config.task in {"fetchpnp"}:
    def obs2goal(obs):
      return obs[..., :3]
  if "demofetchpnp" in config.task:
    def obs2goal(obs):
      return obs
  return obs2goal

def make_sample_env_goals(config, env):
  sample_fn = None
  if config.task == 'hardumazefulldownscale' or  'demofetchpnp' in config.task:
    def sample_fn(num_samples):
      all_goals = tf.convert_to_tensor(env.get_goals(), dtype=tf.float32)
      N = len(all_goals)
      goal_ids = tf.random.categorical(tf.math.log([[1/N] * N]), num_samples)
      # tf.print("goal ids", goal_ids)
      return tf.gather(all_goals, goal_ids)[0]

  return sample_fn


def main():
  """
  Pass in the config setting(s) you want from the configs.yaml. If there are multiple
  configs, we will override previous configs with later ones, like if you want to add
  debug mode to your environment.

  To override specific config keys, pass them in with --key value.

  python examples/run_goal_cond.py --configs <setting 1> <setting 2> ... --foo bar

  Examples:
    Normal scenario
      python examples/run_goal_cond.py --configs mega_fetchpnp_proprio
    Debug scenario
      python examples/run_goal_cond.py --configs mega_fetchpnp_proprio debug
    Override scenario
      python examples/run_goal_cond.py --configs mega_fetchpnp_proprio --seed 123
  """
  """ ========= SETUP CONFIGURATION  ========"""
  configs = yaml.safe_load((
      pathlib.Path(sys.argv[0]).parent.parent / 'dreamerv2/configs.yaml').read_text())
  parsed, remaining = common.Flags(configs=['defaults']).parse(known_only=True)
  config = common.Config(configs['defaults'])
  for name in parsed.configs:
    config = config.update(configs[name])
  config = old_config =  common.Flags(config).parse(remaining)

  logdir = pathlib.Path(config.logdir).expanduser()
  if logdir.exists():
    print('Loading existing config')
    yaml_config = yaml.safe_load((logdir / 'config.yaml').read_text())
    new_keys = []
    for key in new_keys:
      if key not in yaml_config:
        print(f"{key} does not exist in saved config file, using default value from default config file")
        yaml_config[key] = old_config[key]
    config = common.Config(yaml_config)
    config = common.Flags(config).parse(remaining)
    config.save(logdir / 'config.yaml')
    # config = common.Config(yaml_config)
    # config = common.Flags(config).parse(remaining)
  else:
    print('Creating new config')
    logdir.mkdir(parents=True, exist_ok=True)
    config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  """ ========= SETUP ENVIRONMENTS  ========"""
  env = make_env(config, use_goal_idx=False, log_per_goal=True)
  eval_env = make_env(config, use_goal_idx=True, log_per_goal=False, eval=True)
  sample_env_goals = make_sample_env_goals(config, eval_env)
  report_render_fn = make_report_render_function(config)
  eval_fn = make_eval_fn(config)
  plot_fn = make_plot_fn(config)
  ep_render_fn = make_ep_render_fn(config)
  cem_vis_fn = make_cem_vis_fn(config)
  obs2goal_fn = make_obs2goal_fn(config)

  """ ========= SETUP TF2 and GPU ========"""
  tf.config.run_functions_eagerly(not config.jit)
  # tf.data.experimental.enable_debug_mode(not config.jit)
  message = 'No GPU found. To actually train on CPU remove this assert.'
  assert tf.config.experimental.list_physical_devices('GPU'), message
  for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    from tensorflow.keras.mixed_precision import experimental as prec
    prec.set_policy(prec.Policy('mixed_float16'))

  """ ========= BEGIN TRAIN ALGORITHM ========"""
  dv2.train(env, eval_env, eval_fn, report_render_fn, ep_render_fn, plot_fn, cem_vis_fn, obs2goal_fn, sample_env_goals, config)

if __name__ == "__main__":
    main()