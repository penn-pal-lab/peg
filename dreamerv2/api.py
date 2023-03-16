import collections
import logging
import os
import pathlib
import re
import sys
import warnings
import signal
import tensorflow as tf
from time import time
import matplotlib.pyplot as plt
from functools import partial
import pickle
from collections import defaultdict
from time import time
from tqdm import tqdm
import imageio

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger().setLevel('ERROR')
warnings.filterwarnings('ignore', '.*box bound precision lowered.*')

sys.path.append(str(pathlib.Path(__file__).parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import numpy as np
import ruamel.yaml as yaml

import gc_agent
import common
import goal_picker

from common import Config
from common import GymWrapper
from common import RenderImage
from common import TerminalOutput
from common import JSONLOutput
from common import TensorBoardOutput

configs = yaml.safe_load(
    (pathlib.Path(__file__).parent / 'configs.yaml').read_text())
defaults = common.Config(configs.pop('defaults'))


def train(env, eval_env, eval_fn, report_render_fn, ep_render_fn, plot_fn, cem_vis_fn, obs2goal_fn, sample_env_goals, config, outputs=None):
  """Trains the GC-LEXA agent.
  Args:
      env (_type_): training env
      eval_env (_type_): eval env
      eval_fn (_type_): function that collects evaluation episodes and records stats.
      report_render_fn (_type_): function that renders images based on state predictions from model.
      config (_type_): config dictionary.
      outputs (_type_, optional): outputs for logging
  """
  csm = None
  if config.slurm_preempt:
    minutes = 60 * 7 + 45 # every X minutes, requeue this job
    minutes_in_seconds = minutes * 60
    csm = ClusterStateManager(minutes_in_seconds)

  logdir = pathlib.Path(config.logdir).expanduser()
  logdir.mkdir(parents=True, exist_ok=True)
  config.save(logdir / 'config.yaml')
  print(config, '\n')
  print('Logdir', logdir)

  outputs = outputs or [
      common.TerminalOutput(),
      common.JSONLOutput(config.logdir),
      common.TensorBoardOutput(config.logdir),
  ]
  replay = common.Replay(logdir / 'train_episodes', **config.replay) # initialize replay buffer
  eval_replay = common.Replay(logdir / 'eval_episodes', **dict(
      capacity=config.replay.capacity // 10,
      minlen=config.dataset.length,
      maxlen=config.dataset.length))
  step = common.Counter(replay.stats['total_steps']) # initialize step counter
  num_eps = common.Counter(replay.stats['total_episodes']) # initialize episode counter
  num_algo_updates = common.Counter(0)
  logger = common.Logger(step, outputs, multiplier=config.action_repeat) # initialize logger
  metrics = collections.defaultdict(list) # minitialize metrics list

  should_train = common.Every(config.train_every) # train every 5 steps
  should_goal_update = common.Every(config.goal_update_every) # how often to refresh goal picker distribution
  should_log = common.Every(config.log_every) # log every 1e4 steps
  should_video_train = common.Every(config.eval_every)
  should_video_eval = common.Every(0) # never record eval rollout.
  should_expl = common.Until(config.expl_until) # expl_until 0 steps, so no exploration.
  should_eval = common.Every(config.eval_every) # eval every 10 rollouts.
  should_ckpt = common.Every(config.ckpt_every) # ckpt every X episodes.
  should_gcp_rollout = common.Every(config.gcp_rollout_every)
  should_exp_rollout = common.Every(config.exp_rollout_every)
  should_two_policy_rollout = common.Every(config.two_policy_rollout_every)
  should_plot = common.Every(config.eval_every) # show image every time it evaluates
  should_cem_plot = common.Every(config.eval_every) # show image every time it evaluates

  def per_episode(ep, mode):
    length = len(ep['reward']) - 1
    score = float(ep['reward'].astype(np.float64).sum())
    print(f'{mode.title()} episode has {length} steps and return {score:.1f}.')
    logger.scalar(f'{mode}_return', score)
    logger.scalar(f'{mode}_length', length)
    for key, value in ep.items():
      if re.match(config.log_keys_sum, key):
        logger.scalar(f'sum_{mode}_{key}', ep[key].sum())
      if re.match(config.log_keys_mean, key):
        logger.scalar(f'mean_{mode}_{key}', ep[key].mean())
      if re.match(config.log_keys_max, key):
        logger.scalar(f'max_{mode}_{key}', ep[key].max(0).mean())
    should = {'train': should_video_train, 'eval': should_video_eval}[mode]
    if should(num_eps):
      if ep_render_fn is None and 'none' not in config.log_keys_video:
        for key in config.log_keys_video:
          logger.video(f'{mode}_policy_{key}', ep[key])
      elif ep_render_fn is not None:
        video = ep_render_fn(eval_env, ep)
        if video is not None:
          logger.video(f'{mode}_policy_{config.state_key}', video)
    _replay = dict(train=replay, eval=eval_replay)[mode]
    logger.add(_replay.stats, prefix=mode)
    logger.write()



  driver = common.GCDriver([env], config.goal_key)
  driver.on_episode(lambda ep: per_episode(ep, mode='train'))
  driver.on_episode(lambda ep: num_eps.increment())
  driver.on_step(lambda tran, worker: step.increment())
  driver.on_step(replay.add_step)
  driver.on_reset(replay.add_step)

  eval_driver = common.GCDriver([eval_env], config.goal_key)
  eval_driver.on_episode(lambda ep: per_episode(ep, mode='eval'))
  eval_driver.on_episode(eval_replay.add_episode)

  prefill = max(0, config.prefill - replay.stats['total_episodes'])
  random_agent = common.RandomAgent(env.act_space)
  if prefill:
    print(f'Prefill dataset ({prefill} episodes).')
    driver(random_agent, episodes=prefill)
    driver.reset()

  print('Create agent.')
  dataset = iter(replay.dataset(**config.dataset))
  if config.gcp_train_factor > 1:
    gcp_dataset = iter(replay.dataset(**config.dataset))
  if config.replay.sample_recent:
    recent_dataset = iter(replay.recent_dataset(**config.dataset))
    if config.gcp_train_factor > 1:
      recent_gcp_dataset = iter(replay.recent_dataset(**config.dataset))
  report_dataset = iter(replay.dataset(**config.dataset)) # for train vid pred.
  agnt = gc_agent.GCAgent(config, env.obs_space, env.act_space, step, obs2goal_fn, sample_env_goals)

  # def plot_episode(): # define the plot function after agent has been defined.
  #   if should_plot(num_eps) and plot_fn != None:
  #     from time import time
  #     plt.cla()
  #     plt.clf()
  #     start = time()
  #     plot_fn(eval_env, agnt=agnt, complete_episodes=replay._complete_eps, logger=logger, ep_subsample=1, step_subsample=1)
  #     print("plotting took ", time() - start)
  #     logger.write()
  # driver.on_episode(lambda ep: plot_episode())

  train_agent = common.CarryOverState(agnt.train)
  train_gcp = common.CarryOverState(agnt.train_gcp)
  train_agent(next(dataset))

  if (logdir / 'variables.pkl').exists():
    print('Found existing checkpoint.')
    agnt.load(logdir / 'variables.pkl')
  else:
    print('Pretrain agent.')
    for _ in range(config.pretrain):
      for i in range(config.gcp_train_factor - 1):
        train_gcp(next(gcp_dataset))
      train_agent(next(dataset))

  def slurm_preempt_step(tran, worker):
    if csm is None:
      return

    if csm.should_exit():
      # checkpoint.
      agnt.save(logdir / 'variables.pkl')
      # gracefully stop program.
      print("Exiting with exit code", csm.get_exit_code())
      sys.exit(csm.get_exit_code())
  driver.on_step(slurm_preempt_step)

  def train_step(tran, worker):
    if should_train(step):
      # start_time = time()
      # data_duration = 0
      # train_duration = 0
      for _ in range(config.train_steps):
        # data_start = time()
        _data = next(dataset)
        # data_duration += time() - data_start
        # train_start = time()
        mets = train_agent(_data)
        # train_duration += time() - train_start
        [metrics[key].append(value) for key, value in mets.items()]
        for i in range(config.gcp_train_factor - 1):
          mets = train_gcp(next(gcp_dataset))
          [metrics[key].append(value) for key, value in mets.items()]
        if config.replay.sample_recent:
          _data = next(recent_dataset)
          mets = train_agent(_data)
          [metrics[key].append(value) for key, value in mets.items()]
          for i in range(config.gcp_train_factor - 1):
            mets = train_gcp(next(recent_gcp_dataset))
            [metrics[key].append(value) for key, value in mets.items()]
      # duration = time() - start_time
      # print(config.train_steps, "train steps took", duration)
      # print("data loading  took", data_duration)
      # print("train step took", train_duration)
    if should_log(step):
      for name, values in metrics.items():
        logger.scalar(name, np.array(values, np.float64).mean())
        metrics[name].clear()
      logger.add(agnt.report(next(report_dataset), eval_env, report_render_fn))
      logger.write(fps=True)
  driver.on_step(train_step)

  train_gcpolicy = partial(agnt.policy, mode='train')
  eval_gcpolicy = partial(agnt.policy, mode='eval')
  def expl_policy(obs, state, **kwargs):
    actions, state = agnt.expl_policy(obs, state, mode='train')
    if config.go_expl_rand_ac:
      actions, _ = random_agent(obs)
    return  actions, state

  """Define goal picking """
  goal_picker_cls = getattr(goal_picker, config.goal_strategy)
  p_cfg = config.planner
  if config.goal_strategy == "Greedy":
    goal_strategy = goal_picker_cls(replay, agnt.wm, agnt._expl_behavior._intr_reward, config.state_key, config.goal_key, 1000)
  elif config.goal_strategy == "SampleReplay":
    goal_strategy = goal_picker_cls(agnt.wm, dataset, config.state_key, config.goal_key)
  elif config.goal_strategy == "SubgoalPlanner":
    if p_cfg.init_candidates[0] == 123456789.0: # ugly hack for specifying no init cand.
      init_cand = None
    else:
      init_cand = np.array(p_cfg.init_candidates, dtype=np.float32)
      # unflatten list of init candidates
      goal_dim=np.prod(env.obs_space[config.state_key].shape) # assume goal dim = state dim
      assert len(init_cand) == goal_dim, f"{len(init_cand)}, {goal_dim}"
      init_cand = np.split(init_cand, len(init_cand)//goal_dim)
      init_cand = tf.convert_to_tensor(init_cand)

    def vis_fn(elite_inds, elite_samples, seq, wm):
      if should_cem_plot(num_eps) and cem_vis_fn is not None:
        cem_vis_fn(elite_inds, elite_samples, seq, wm, eval_env, logger)

    goal_dataset = None
    if p_cfg.sample_replay:
      goal_dataset = iter(replay.dataset(batch=10000//(config.time_limit+1), length=config.time_limit+1)) # take 10K states.

    mega_prior = None
    if p_cfg.mega_prior:
      mega_prior = goal_picker.MEGA(agnt, replay, env.act_space, config.state_key, config.time_limit+1, obs2goal_fn)

    sample_env_goals_fn = None
    env_goals_percentage = p_cfg.init_env_goal_percent
    if env_goals_percentage > 0:
      sample_env_goals_fn = sample_env_goals

    goal_strategy = goal_picker_cls(
      agnt.wm,
      agnt._task_behavior.actor,
      agnt._expl_behavior.planner_intr_reward,
      gc_input=config.gc_input,
      obs2goal=obs2goal_fn,
      goal_dim=np.prod(env.observation_space[config.goal_key].shape),
      goal_min=np.array(p_cfg.goal_min, dtype=np.float32),
      goal_max=np.array(p_cfg.goal_max, dtype=np.float32),
      act_space=env.act_space,
      state_key=config.state_key,
      planner=p_cfg.planner_type,
      horizon=p_cfg.horizon,
      batch=p_cfg.batch,
      cem_elite_ratio=p_cfg.cem_elite_ratio,
      optimization_steps=p_cfg.optimization_steps,
      std_scale=p_cfg.std_scale,
      mppi_gamma=p_cfg.mppi_gamma,
      init_candidates=init_cand,
      dataset=goal_dataset,
      evaluate_only=p_cfg.evaluate_only,
      repeat_samples=p_cfg.repeat_samples,
      mega_prior=mega_prior,
      sample_env_goals_fn=sample_env_goals_fn,
      env_goals_percentage=env_goals_percentage,
      vis_fn=vis_fn
    )
  elif config.goal_strategy in {"MEGA", "Skewfit"}:
    goal_strategy = goal_picker_cls(agnt, replay, env.act_space, config.state_key, config.time_limit, obs2goal_fn)
  else:
    raise NotImplementedError

  def get_goal(obs, state=None, mode='train'):
    obs = tf.nest.map_structure(lambda x: tf.expand_dims(tf.expand_dims(tf.tensor(x),0),0), obs)[0]
    obs = agnt.wm.preprocess(obs)
    if np.random.uniform() < config.planner.sample_env_goal_percent:
      goal = sample_env_goals(1)
      return tf.squeeze(goal)

    if config.goal_strategy == "Greedy":
      goal = goal_strategy.get_goal()
      goal_strategy.will_update_next_call = False
    elif config.goal_strategy == "SampleReplay":
      goal = goal_strategy.get_goal(obs)
    elif config.goal_strategy == "SubgoalPlanner":
      goal = goal_strategy.search_goal(obs, state)
      goal_strategy.will_update_next_call = False
    elif config.goal_strategy == "SubgoalPlannerKDE":
      goal = goal_strategy.search_goal(obs, state)
      goal_strategy.will_update_next_call = False
    elif config.goal_strategy in {"MEGA", "Skewfit"}:
      goal = goal_strategy.sample_goal(obs, state)
    else:
      raise NotImplementedError
    return tf.squeeze(goal)

  def update_goal_strategy(*args):
    if should_goal_update(num_eps):
      if config.goal_strategy == "Greedy":
        goal_strategy.update_buffer_priorities()
      elif "SubgoalPlanner" in config.goal_strategy:
        #  goal strategy will search for new distribution next time we sample.
        goal_strategy.will_update_next_call = True
        if config.planner.mega_prior:
          goal_strategy.mega.update_kde()
      elif config.goal_strategy in {"MEGA", "Skewfit"}:
        goal_strategy.update_kde()
  driver.on_episode(lambda ep: update_goal_strategy())


  goal_time_limit = round(config.goal_policy_rollout_percentage * config.time_limit)
  def temporal_dist(obs):
    # TODO: assumes obs list is only 1 element.
    obs = tf.nest.map_structure(lambda x: tf.expand_dims(tf.tensor(x), 0), obs)[0]
    dist = agnt.temporal_dist(obs).numpy().item()
    success = dist < config.subgoal_threshold
    metric = {"subgoal_dist": dist, "subgoal_success": float(success)}
    return success, metric

  def eval_agent():
    if should_eval(num_eps):
      print('Start evaluation.')
      eval_fn(eval_driver, eval_gcpolicy, logger)
      agnt.save(logdir / 'variables.pkl')
    if should_ckpt(num_eps):
      print('Checkpointing.')
      agnt.save(logdir / f'variables_{num_eps.value}.pkl')

  while step < config.steps:
    logger.write()
    # alternate between these 3 types of rollouts.
    """ 1. train: run goal cond. policy for entire rollout"""
    if should_gcp_rollout(num_algo_updates):
      driver(train_gcpolicy, get_goal=get_goal, episodes=1)
      eval_agent()
    """ 2. expl: run expl policy for entire rollout """
    if should_exp_rollout(num_algo_updates):
      driver(expl_policy, episodes=1)
      eval_agent()
    """ 3. 2pol: run goal cond. and then expl policy."""
    if should_two_policy_rollout(num_algo_updates):
      driver(train_gcpolicy, expl_policy, get_goal, episodes=1, goal_time_limit=goal_time_limit, goal_checker=temporal_dist)
      eval_agent()
    num_algo_updates.increment()

def eval(env, eval_env, eval_fn, obs2goal_fn, sample_env_goals, config, outputs=None):
  """Evaluates the agent.
  Args:
      env (_type_): training env
      eval_env (_type_): eval env
      eval_fn (_type_): function that collects evaluation episodes and records stats.
      report_render_fn (_type_): function that renders images based on state predictions from model.
      config (_type_): config dictionary.
      outputs (_type_, optional): outputs for logging
  """
  logdir = pathlib.Path(config.logdir).expanduser()
  outputs = outputs or [
      common.TerminalOutput(),
      common.GIFOutput(logdir),
  ]
  print('loading replay')
  replay = common.Replay(logdir / 'train_episodes', **config.replay) # initialize replay buffer
  print('done loading replay')
  step = common.Counter(replay.stats['total_steps']) # initialize step counter
  logger = common.Logger(step, outputs, multiplier=config.action_repeat) # initialize logger

  eval_driver = common.GCDriver([eval_env], config.goal_key)

  print('Create agent.')
  dataset = iter(replay.dataset(**config.dataset))
  agnt = gc_agent.GCAgent(config, env.obs_space, env.act_space, step, obs2goal_fn, sample_env_goals)

  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(dataset))

  assert (logdir / 'variables.pkl').exists()
  print('Found existing checkpoint.')
  agnt.load(logdir / 'variables.pkl')

  eval_gcpolicy = partial(agnt.policy, mode='eval')
  print('Start evaluation.')
  eval_fn(eval_driver, eval_gcpolicy, logger)

def visualize_3stack(eval_env, obs2goal_fn, sample_env_goals, config, outputs=None):
  """Visualizes the agent.
  Args:
      env (_type_): training env
      eval_env (_type_): eval env
      eval_fn (_type_): function that collects evaluation episodes and records stats.
      report_render_fn (_type_): function that renders images based on state predictions from model.
      config (_type_): config dictionary.
      outputs (_type_, optional): outputs for logging
  """
  import pickle
  from tqdm import tqdm
  import imageio

  logdir = pathlib.Path(config.logdir).expanduser()
  outputs = outputs or [
      common.TerminalOutput(),
      # common.GIFOutput(logdir),
  ]
  print('loading replay')
  replay = common.Replay(logdir / 'train_episodes', **config.replay) # initialize replay buffer
  print('done loading replay')
  step = common.Counter(replay.stats['total_steps']) # initialize step counter

  # if not os.path.exists("3block_video.pkl"):
  print('Create agent.')
  dataset = iter(replay.dataset(**config.dataset))
  agnt = gc_agent.GCAgent(config, eval_env.obs_space, eval_env.act_space, step, obs2goal_fn, sample_env_goals)

  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(dataset))

  assert (logdir / 'variables.pkl').exists()
  print('Found existing checkpoint.')
  agnt.load(logdir / 'variables.pkl')

  eval_gcpolicy = partial(agnt.policy, mode='eval')
  ep_counter = 0
  all_success_states = []
  eval_env._env._env._env._env._env.distance_threshold = 0.05

  # goal_ranges = [range(3), range(3,15), range(15,21)]
  goal_ranges = [range(3,9)]
  all_goal_range_successes = []
  for goal_range in goal_ranges:
    print("evaluating goal family", goal_range)
    goal_range_successes = {}
    for goal_idx in goal_range:
      success_per_goal = 0
      for ep_idx in range(10):
        eval_env.set_goal_idx(goal_idx)
        env_states = []
        ob = eval_env.reset()
        sim_state = eval_env._env._env._env._env._env.sim.get_state().flatten()
        # env_states.append(sim_state)
        ep_counter += 1
        ob = {k: np.expand_dims(v, 0) for k,v in ob.items()}
        state= None
        time = last_success_time = 0
        for _ in range(config.time_limit):
          action, state = eval_gcpolicy(ob, state)
          ob = eval_env.step({'action': action['action'][0]})
          # sim_state =  eval_env._env._env._env._env._env.sim.get_state().flatten()
          # env_states.append(sim_state)
          ob = {k: np.expand_dims(v, 0) for k,v in ob.items()}
          time += 1
          if np.max(ob[f'metric_success/goal_{goal_idx}']) > 0:
            print('success', ep_counter)
            last_success_time = time
            success_per_goal += 1

          if ob['is_last']:
            print('fail', ep_counter)
            # if last_success_time > 0:
            #   all_success_states.append(env_states[:])
            break
      success_per_goal /= 10
      goal_range_successes[goal_idx] = success_per_goal

    all_goal_range_successes.append(goal_range_successes)

  all_goals_success = []
  for goal_range, goal_range_successes in zip(goal_ranges, all_goal_range_successes):
    print("goal family", goal_range, "success:")
    total_success = []
    for goal_idx, mean_success in goal_range_successes.items():
      print(f"goal {goal_idx} success: {mean_success * 100}")
      total_success.append(mean_success)

    family_success = np.mean(total_success) * 100
    all_goals_success.append(family_success)
    print("goal family success:", family_success)
    print("=" * 80)
    print()
  print("all goals success:", np.mean(all_goals_success))
    # with open("3block_video.pkl", "wb") as f:
    #   pickle.dump(all_success_states, f)
  # else:
  #   with open("3block_video.pkl", "rb") as f:
  #     all_success_states = pickle.load(f)

  # for i, states in tqdm(enumerate(all_success_states)):
    # video = []
    # eval_env.set_goal_idx(2)
    # eval_env.reset()
    # img = eval_env.render('rgb_array', 500, 500)
    # video.append(img)
    # for state in states:
    #   eval_env._env._env._env._env._env.sim.set_state_from_flattened(state)
    #   img = eval_env.render('rgb_array', 500, 500)
    #   video.append(img)
    # # render the final state for a bit longer.
    # for _ in range(10):
    #   img = eval_env.render('rgb_array', 500, 500)
    #   video.append(img)
    # imageio.mimwrite(f"3stack_{i}.mp4", video)

def eval_antmaze(eval_env, obs2goal_fn, sample_env_goals, config, outputs=None):
  """Visualizes the agent.
  Args:
      env (_type_): training env
      eval_env (_type_): eval env
      eval_fn (_type_): function that collects evaluation episodes and records stats.
      report_render_fn (_type_): function that renders images based on state predictions from model.
      config (_type_): config dictionary.
      outputs (_type_, optional): outputs for logging
  """
  import pickle
  from tqdm import tqdm
  import imageio

  logdir = pathlib.Path(config.logdir).expanduser()
  outputs = outputs or [
      common.TerminalOutput(),
      # common.GIFOutput(logdir),
  ]
  print('loading replay')
  replay = common.Replay(logdir / 'train_episodes', **config.replay) # initialize replay buffer
  print('done loading replay')
  step = common.Counter(replay.stats['total_steps']) # initialize step counter

  # if not os.path.exists("3block_video.pkl"):
  print('Create agent.')
  dataset = iter(replay.dataset(**config.dataset))
  agnt = gc_agent.GCAgent(config, eval_env.obs_space, eval_env.act_space, step, obs2goal_fn, sample_env_goals)

  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(dataset))

  assert (logdir / 'variables.pkl').exists()
  print('Found existing checkpoint.')
  agnt.load(logdir / 'variables.pkl')

  eval_gcpolicy = partial(agnt.policy, mode='eval')
  ep_counter = 0
  all_success_states = []
  # import ipdb; ipdb.set_trace()
  # eval_env._env._env._env._env._env.distance_threshold = 0.05

  # goal_ranges = [range(3), range(3,15), range(15,21)]
  goal_ranges = [range(0,4), range(4, 8), range(8, 12), range(12, 16), range(16, 20), range(20, 24), range(24, 28), range(28, 32)]
  all_goal_range_successes = []
  for goal_range in goal_ranges:
    print("evaluating goal family", goal_range)
    goal_range_successes = {}
    for goal_idx in goal_range:
      success_per_goal = 0
      for ep_idx in range(10):
        eval_env.set_goal_idx(goal_idx)
        env_states = []
        ob = eval_env.reset()
        # sim_state = eval_env._env._env._env._env._env.sim.get_state().flatten()
        # env_states.append(sim_state)
        ep_counter += 1
        ob = {k: np.expand_dims(v, 0) for k,v in ob.items()}
        state= None
        time = last_success_time = 0
        for _ in range(config.time_limit):
          action, state = eval_gcpolicy(ob, state)
          ob = eval_env.step({'action': action['action'][0]})
          # sim_state =  eval_env._env._env._env._env._env.sim.get_state().flatten()
          # env_states.append(sim_state)
          ob = {k: np.expand_dims(v, 0) for k,v in ob.items()}
          time += 1
          if np.max(ob[f'metric_success/goal_{goal_idx}']) > 0:
            print('success', ep_counter)
            last_success_time = time
            success_per_goal += 1

          if ob['is_last']:
            print('fail', ep_counter)
            # if last_success_time > 0:
            #   all_success_states.append(env_states[:])
            break
      success_per_goal /= 10
      goal_range_successes[goal_idx] = success_per_goal

    all_goal_range_successes.append(goal_range_successes)

  all_goals_success = []
  for goal_range, goal_range_successes in zip(goal_ranges, all_goal_range_successes):
    print("Goal family", goal_range, "success:")
    total_success = []
    for goal_idx, mean_success in goal_range_successes.items():
      print(f"Goal {goal_idx} success: {mean_success * 100}")
      total_success.append(mean_success)

    family_success = np.mean(total_success) * 100
    all_goals_success.append(family_success)
    print("Goal Family success:", family_success)
    print("=" * 80)
    print()
  print("All Goals success:", np.mean(all_goals_success))
    # with open("3block_video.pkl", "wb") as f:
    #   pickle.dump(all_success_states, f)
  # else:
  #   with open("3block_video.pkl", "rb") as f:
  #     all_success_states = pickle.load(f)

  # for i, states in tqdm(enumerate(all_success_states)):
    # video = []
    # eval_env.set_goal_idx(2)
    # eval_env.reset()
    # img = eval_env.render('rgb_array', 500, 500)
    # video.append(img)
    # for state in states:
    #   eval_env._env._env._env._env._env.sim.set_state_from_flattened(state)
    #   img = eval_env.render('rgb_array', 500, 500)
    #   video.append(img)
    # # render the final state for a bit longer.
    # for _ in range(10):
    #   img = eval_env.render('rgb_array', 500, 500)
    #   video.append(img)
    # imageio.mimwrite(f"3stack_{i}.mp4", video)
def evaluate_3stack(eval_env, obs2goal_fn, sample_env_goals, config, outputs=None):
  logdir = pathlib.Path(config.logdir).expanduser()
  # first check for existing checkpoints.
  entries = sorted(os.scandir(logdir), key=lambda ent: ent.stat().st_mtime)
  checkpoints = []
  for e in entries:
    if e.name.endswith("pkl") and e.name != 'variables.pkl':
      checkpoints.append(e)
  all_results = []
  for checkpoint in checkpoints:
    result = evaluate_3stack_checkpoint(checkpoint.path, eval_env, obs2goal_fn, sample_env_goals, config)
    num_episodes = int(checkpoint.path.split("_")[-1].split('.')[0])
    num_steps = num_episodes * config.time_limit
    all_results.append((num_steps, result))

  # save results to pkl.
  with open(logdir / "3stack_eval_data.pkl", "wb") as f:
    pickle.dump(all_results, f)

def evaluate_3stack_checkpoint(checkpoint_path, eval_env, obs2goal_fn, sample_env_goals, config, outputs=None):
  logdir = pathlib.Path(config.logdir).expanduser()
  outputs = outputs or [
      common.TerminalOutput(),
      # common.GifOutput(logdir),
  ]
  print('loading replay')
  replay = common.Replay(logdir / 'train_episodes', **config.replay) # initialize replay buffer
  print('done loading replay')
  step = common.Counter(replay.stats['total_steps']) # initialize step counter

  print('create agent.')
  dataset = iter(replay.dataset(**config.dataset))
  agnt = gc_agent.GCAgent(config, eval_env.obs_space, eval_env.act_space, step, obs2goal_fn, sample_env_goals)

  train_agent = common.CarryOverState(agnt.train)
  train_agent(next(dataset))

  print('Evaluating', checkpoint_path)
  agnt.load(checkpoint_path)

  eval_gcpolicy = partial(agnt.policy, mode='eval')
  ep_counter = 0
  all_success_states = []
  eval_env._env._env._env._env._env.distance_threshold = 0.03

  # goal_ranges = [range(3), range(3,15), range(15,21)]
  goal_ranges = [range(9,15)]
  all_goal_range_successes = []
  for goal_range in goal_ranges:
    print("evaluating goal family", goal_range)
    goal_range_successes = {}
    for goal_idx in goal_range:
      success_per_goal = 0
      for ep_idx in range(10):
        eval_env.set_goal_idx(goal_idx)
        env_states = []
        ob = eval_env.reset()
        sim_state = eval_env._env._env._env._env._env.sim.get_state().flatten()
        # env_states.append(sim_state)
        ep_counter += 1
        ob = {k: np.expand_dims(v, 0) for k,v in ob.items()}
        state= None
        time = last_success_time = 0
        for _ in range(config.time_limit):
          action, state = eval_gcpolicy(ob, state)
          ob = eval_env.step({'action': action['action'][0]})
          # sim_state =  eval_env._env._env._env._env._env.sim.get_state().flatten()
          # env_states.append(sim_state)
          ob = {k: np.expand_dims(v, 0) for k,v in ob.items()}
          time += 1
          if np.max(ob[f'metric_success/goal_{goal_idx}']) > 0:
            print('success', ep_counter)
            last_success_time = time
            success_per_goal += 1

          if ob['is_last']:
            print('fail', ep_counter)
            # if last_success_time > 0:
            #   all_success_states.append(env_states[:])
            break
      success_per_goal /= 10
      goal_range_successes[goal_idx] = success_per_goal

    all_goal_range_successes.append(goal_range_successes)

  results = {}
  all_goals_success = []
  for goal_range, goal_range_successes in zip(goal_ranges, all_goal_range_successes):
    print("goal family", goal_range, "success:")
    total_success = []
    for goal_idx, mean_success in goal_range_successes.items():
      results[goal_idx] = mean_success * 100
      print(f"goal {goal_idx} success: {mean_success * 100}")
      total_success.append(mean_success)

    family_success = np.mean(total_success) * 100
    all_goals_success.append(family_success)
    print("goal family success:", family_success)
    print("=" * 80)
    print()
  print("all goals success:", np.mean(all_goals_success))
  return results





class ClusterStateManager:
    def __init__(self, time_to_run):
        self.external_exit = None
        self.timer_exit = False

        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGALRM, self.timer_handler)
        signal.alarm(time_to_run) # in seconds.

    def signal_handler(self, signal, frame):
        print("Received signal [", signal, "]")
        self.external_exit = signal

    def timer_handler(self, signal, frame):
        print("Received alarm [", signal, "]")
        self.timer_exit = True

    def should_exit(self):
        if self.timer_exit:
            return True

        if self.external_exit is not None:
            return True

        return False

    def get_exit_code(self):
        if self.timer_exit:
            return 3

        if self.external_exit is not None:
            return 0

        return 0