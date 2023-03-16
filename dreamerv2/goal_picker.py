from collections import defaultdict
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
from time import time

class Greedy:
  def __init__(self, replay, wm, reward_fn,  state_key, goal_key, batch_size, topk=10, exp_weight=1.0):
    self.replay = replay
    self.wm = wm
    self.reward_fn = reward_fn
    self.state_key = state_key
    self.goal_key = goal_key
    self.batch_size = batch_size
    self.topk = topk
    self.exp_weight = exp_weight
    self.all_topk_states = None

  def update_buffer_priorities(self):
    start = time()
    # go through the entire replay buffer and extract top K goals.
    @tf.function
    def process_batch(data, reward_fn):
      data = self.wm.preprocess(data)
      states = data[self.state_key]
      # need to pass states through encoder / rssm first to get 'feat'
      embed = self.wm.encoder(data)
      post, prior = self.wm.rssm.observe(
          embed, data['action'], data['is_first'], state=None)
      data['feat'] = self.wm.rssm.get_feat(post)
      # feed these states into the plan2expl loss.
      reward = reward_fn(data).reshape((-1,))
      values, indices = tf.math.top_k(reward, self.topk)
      states = data[self.state_key].reshape((-1, data[self.state_key].shape[-1]))
      topk_states = tf.gather(states, indices)
      # last_state = {k: v[:, -1] for k, v in post.items()}
      return values, topk_states
    self.all_topk_states = []
    # reward_fn = agent._expl_behavior._intr_reward
    # this dict contains keys (file paths) and values (episode dicts)
    num_episodes = len(self.replay._complete_eps)
    chunk = defaultdict(list)
    count = 0
    for idx, ep_dict in enumerate(self.replay._complete_eps.values()):
      for k,v in ep_dict.items():
        chunk[k].append(v)
      count += 1
      if count >= self.batch_size or idx == num_episodes-1: # done with collecting chunk.
        count = 0
        data = {k: np.stack(v) for k,v in chunk.items()}
        # for k, v in data.items():
        #   print(k, v.shape)
        chunk = defaultdict(list)
        # do processing of batch here.
        values, top_states = process_batch(data, self.reward_fn)
        values_states = [(v,s) for v,s in zip(values, top_states)]
        self.all_topk_states.extend(values_states)
        self.all_topk_states.sort(key=lambda x: x[0], reverse=True)
        self.all_topk_states = self.all_topk_states[:self.topk]
    end = time() - start
    print("update buffer took", end)


  def get_goal(self):
    if self.all_topk_states is None:
      self.update_buffer_priorities()

    priorities = np.asarray([x[0] for x in self.all_topk_states])
    priorities += 1e-6  # epsilon to prevent collapse
    np.exp(priorities * self.exp_weight)
    prob = np.squeeze(priorities) / priorities.sum()
    idx = np.random.choice(len(self.all_topk_states), 1, replace=True, p=prob)[0]
    value, state = self.all_topk_states[idx]
    return state.numpy()

class SampleReplay:
  def __init__(self, wm, dataset, state_key, goal_key):
    self.state_key = state_key
    self.goal_key = goal_key
    self._dataset = dataset
    self.wm = wm

  @tf.function
  def get_goal(self, obs):
    random_batch = next(self._dataset)
    random_batch = self.wm.preprocess(random_batch)
    random_goals = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
    return random_goals[:obs[self.state_key].shape[0]]

class SubgoalPlanner:
  def __init__(
      self,
      wm,
      actor,
      reward_fn,
      gc_input,
      obs2goal,
      goal_dim, # D dims
      goal_min, #  D dims for min / max
      goal_max, #  D dims for min / max
      act_space,
      state_key,
      planner="shooting_cem",
      horizon=15,
      mpc_steps=10,
      batch=5,
      cem_elite_ratio=0.2,
      optimization_steps=5,
      std_scale=1.0,
      mppi_gamma=10.0,
      init_candidates=None,
      dataset=None,
      evaluate_only=False, #  don't run CEM, just evaluate goals with model.
      repeat_samples=0,
      mega_prior=False, # an instance of MEGA
      sample_env_goals_fn=None,
      env_goals_percentage=None,
      vis_fn=None,
    ):
    self.wm = wm
    self.dtype = wm.dtype
    self.actor = actor
    self.reward_fn = reward_fn
    self.gc_input = gc_input
    self.obs2goal = obs2goal
    self.goal_dim = goal_dim
    self.act_space = act_space
    if isinstance(act_space, dict):
      self.act_space = act_space['action']
    self.state_key = state_key
    self.planner = planner
    self.horizon = horizon
    self.mpc_steps = mpc_steps
    self.batch = batch
    self.cem_elite_ratio = cem_elite_ratio
    self.optimization_steps = optimization_steps
    self.std_scale = std_scale
    self.mppi_gamma= mppi_gamma
    self.env_goals_percentage = env_goals_percentage
    self.sample_env_goals = env_goals_percentage > 0
    self.sample_env_goals_fn = sample_env_goals_fn

    self.min_action = goal_min
    self.max_action = goal_max

    self.mega = mega_prior
    self.init_distribution = None
    if init_candidates is not None:
      self.create_init_distribution(init_candidates)

    self.dataset = dataset
    self.evaluate_only = evaluate_only
    if self.evaluate_only:
      assert self.dataset is not None, "need to sample from replay buffer."

    self.repeat_samples = repeat_samples
    self.vis_fn = vis_fn
    self.will_update_next_call = True
    self.mega_sample = None


  def search_goal(self, obs, state=None, mode='train'):
    if self.will_update_next_call is False:
      return self.sample_goal()

    elite_size = int(self.batch * self.cem_elite_ratio)
    if state is None:
      latent = self.wm.rssm.initial(1)
      action = tf.zeros((1,1,) + self.act_space.shape)
      state = latent, action
      # print("make new state")
    else:
      latent, action = state
      action = tf.expand_dims(action, 0)
      # action should be (1,1, D)
      # print("using exisitng state")


    # create start state.
    embed = self.wm.encoder(obs)
    # posterior is q(s' | s,a,e)
    post, prior = self.wm.rssm.observe(
        embed, action, obs['is_first'], latent)
    init_start = {k: v[:, -1] for k, v in post.items()}
    # print(action.shape)
    # for k,v in latent.keys():
    #   print(k, v.shape)
    @tf.function
    def eval_fitness(goal):
      # should be (55,128).
      start = {k: v for k, v in init_start.items()}
      start['feat'] = self.wm.rssm.get_feat(start) # (1, 1800)
      start = tf.nest.map_structure(lambda x: tf.repeat(x, goal.shape[0],0), start)
      if self.gc_input == "embed":
        goal_obs = start.copy()
        goal_obs[self.state_key] = goal
        goal_input = self.wm.encoder(goal_obs)
      elif self.gc_input == "state":
        goal_input = tf.cast(goal, self.dtype)

      actor_inp = tf.concat([start['feat'], goal_input], -1)
      start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
      seq = {k: [v] for k, v in start.items()}
      for _ in range(self.horizon):
        actor_inp = tf.concat([seq['feat'][-1], goal_input], -1)
        action = self.actor(actor_inp).sample()
        state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
        feat = self.wm.rssm.get_feat(state)
        for key, value in {**state, 'action': action, 'feat': feat}.items():
          seq[key].append(value)
      seq = {k: tf.stack(v, 0) for k, v in seq.items()}
      # rewards should be (batch,1)
      rewards = self.reward_fn(seq)
      returns = tf.reduce_sum(rewards, 0)
      # rewards = tf.ones([goal.shape[0],])
      return returns, seq

    # CEM loop
    # rewards = []
    # act_losses = []
    if self.init_distribution is None:
      # print("getting init distribtion from obs")
      means, stds = self.get_distribution_from_obs(obs)
    else:
      # print("getting init distribtion from init candidates")
      means, stds = self.init_distribution
    # print(means, stds)
    opt_steps = 1 if self.evaluate_only else self.optimization_steps
    for i in range(opt_steps):
      # Sample action sequences and evaluate fitness
      if i == 0 and (self.dataset or self.mega or self.sample_env_goals):
        if self.dataset:
          # print("getting init distribution from dataset")
          random_batch = next(self.dataset)
          random_batch = self.wm.preprocess(random_batch)
          samples = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
          if self.obs2goal is not None:
            samples = self.obs2goal(samples)
        elif self.sample_env_goals:
          num_cem_samples = int(self.batch * self.env_goals_percentage)
          num_env_samples = self.batch - num_cem_samples
          cem_samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[num_cem_samples])
          env_samples = self.sample_env_goals_fn(num_env_samples)
          samples = tf.concat([cem_samples, env_samples], 0)

        elif self.mega:
          # print("getting init distribution from MEGA")
          samples = self.mega.sample_goal(obs)[None]
          self.mega_sample = samples
          # since mega only returns 1 goal, repeat it.
          samples = tf.repeat(samples, self.batch, 0)
        # initialize means states.
        means, vars = tf.nn.moments(samples, 0)
        # stds = tf.sqrt(vars + 1e-6)
        # stds = tf.concat([[0.5, 0.5], stds[2:]], axis=0)
        # assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
        samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
        # print(i, samples)
        samples = tf.clip_by_value(samples, self.min_action, self.max_action)
      else:
        samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
        samples = tf.clip_by_value(samples, self.min_action, self.max_action)

      if self.repeat_samples > 1:
        repeat_samples = tf.repeat(samples, self.repeat_samples, 0)
        repeat_fitness, seq = eval_fitness(repeat_samples)
        fitness = tf.reduce_mean(tf.stack(tf.split(repeat_fitness, self.repeat_samples)), 0)
      else:
        fitness, seq = eval_fitness(samples)
      # Refit distribution to elite samples
      if self.planner == 'shooting_mppi':
        # MPPI
        weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)
        means = tf.reduce_sum(weights * samples, axis=0)
        stds = tf.sqrt(tf.reduce_sum(weights * tf.square(samples - means), axis=0))
        # rewards.append(tf.reduce_sum(fitness * weights[:, 0]).numpy())
      elif self.planner == 'shooting_cem':
        # CEM
        elite_score, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
        elite_samples = tf.gather(samples, elite_inds)
        # print(elite_samples)
        means, vars = tf.nn.moments(elite_samples, 0)
        stds = tf.sqrt(vars + 1e-6)
        # rewards.append(tf.reduce_mean(tf.gather(fitness, elite_inds)).numpy())

    if self.planner == 'shooting_cem':
      self.vis_fn(elite_inds, elite_samples, seq, self.wm)
      self.elite_inds = elite_inds
      self.elite_samples = elite_samples
      self.final_seq = seq
    elif self.planner == 'shooting_mppi':
      # print("mppi mean", means)
      # print("mppi std", stds)
      # TODO: figure out what elite inds means for shooting mppi.
      # self.vis_fn(elite_inds, elite_samples, seq, self.wm)
      self.elite_inds = None
      self.elite_samples = None
      self.final_seq = seq
    # TODO: potentially store these as initialization for the next update.
    self.means = means
    self.stds = stds

    if self.evaluate_only:
      self.elite_samples = elite_samples
      self.elite_score = elite_score

    return self.sample_goal()

  def sample_goal(self, batch=1):
    if self.evaluate_only:
      # samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])
      # weights = tf.nn.softmax(self.elite_score)
      weights = self.elite_score / self.elite_score.sum()
      idxs = tf.squeeze(tf.random.categorical(tf.math.log([weights]), batch), 0)
      samples = tf.gather(self.elite_samples, idxs)
    else:
      samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])
    return samples

  def create_init_distribution(self, init_candidates):
    """Create the starting distribution for seeding the planner.
    """
    def _create_init_distribution(init_candidates):
      means = tf.reduce_mean(init_candidates, 0)
      stds = tf.math.reduce_std(init_candidates, 0)
      # if there's only 1 candidate, set std to default
      if init_candidates.shape[0] == 1:
        stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
      return means, stds
    self.init_distribution = _create_init_distribution(init_candidates)

  def get_distribution_from_obs(self, obs):
    ob = tf.squeeze(obs[self.state_key])
    if self.gc_input == "state":
      ob = self.obs2goal(ob)
    means = tf.cast(tf.identity(ob), tf.float32)
    assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
    stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
    init_distribution = tf.identity(means), tf.identity(stds)
    return init_distribution

  def get_init_distribution(self):
    if self.init_distribution is None:
      means = tf.zeros(self.goal_dim, dtype=tf.float32)
      stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
      self.init_distribution = tf.identity(means), tf.identity(stds)

    return self.init_distribution

class MEGA:
  def __init__(self, agent, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn=None):
    self.agent = agent
    self.replay = replay
    self.wm = agent.wm
    self.act_space = act_space
    self.goal_sample_fn = goal_sample_fn
    if isinstance(act_space, dict):
      self.act_space = act_space['action']
    # TODO: remove hardcoding
    self.dataset = iter(replay.dataset(batch=10, length=ep_length))
    ## KDE STUFF
    from sklearn.neighbors import KernelDensity
    self.alpha = -1.0
    self.kernel = 'gaussian'
    self.bandwidth = 0.1
    self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.

    self.state_key = state_key
    self.ready = False
    self.random = False
    self.ep_length = ep_length
    self.obs2goal = obs2goal_fn

  def update_kde(self):
    self.ready = True
    # Follow RawKernelDensity in density.py of MRL codebase.

    # ========== Sample Goals=============
    # we know ep length is 51
    num_episodes = self.replay.stats['loaded_episodes']
    # sample 10K goals from the buffer.
    num_samples = min(10000, self.replay.stats['loaded_steps'])
    # first uniformly sample from episodes.
    ep_idx = np.random.randint(0, num_episodes, num_samples)
    # uniformly sample from timesteps
    t_idx = np.random.randint(0, self.ep_length, num_samples)
    # store all these goals.
    all_episodes = list(self.replay._complete_eps.values())
    if self.obs2goal is None:
      kde_samples = [all_episodes[e][self.state_key][t] for e,t in zip(ep_idx, t_idx)]
    else:
      kde_samples = [self.obs2goal(all_episodes[e][self.state_key][t]) for e,t in zip(ep_idx, t_idx)]
    # normalize goals
    self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
    self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
    kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std
  #   # Now also log the entropy
  #   if self.log_entropy and hasattr(self, 'logger') and self.step % 250 == 0:
  #     # Scoring samples is a bit expensive, so just use 1000 points
  #     num_samples = 1000
  #     s = self.fitted_kde.sample(num_samples)
  #     entropy = -self.fitted_kde.score(s)/num_samples + np.log(self.kde_sample_std).sum()
  #     self.logger.add_scalar('Explore/{}_entropy'.format(self.module_name), entropy, log_every=500)


    # =========== Fit KDE ================
    self.fitted_kde = self.kde.fit(kde_samples)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def sample_goal(self, obs, state=None, mode='train'):
    if not self.ready:
      self.update_kde()
    if self.goal_sample_fn:
      num_samples = 10000
      sampled_ags = self.goal_sample_fn(num_samples)
    else:
      # ============ Sample goals from buffer ============
      # random_batch = next(self.dataset)
      # random_batch = self.wm.preprocess(random_batch)
      # sampled_ags = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
      num_episodes = self.replay.stats['loaded_episodes']
      # sample 10K goals from the buffer.
      num_samples = min(10000, self.replay.stats['loaded_steps'])
      # first uniformly sample from episodes.
      ep_idx = np.random.randint(0, num_episodes, num_samples)
      # uniformly sample from timesteps
      t_idx = np.random.randint(0, self.ep_length, num_samples)
      # store all these goals.
      all_episodes = list(self.replay._complete_eps.values())
      if self.obs2goal is None:
        sampled_ags = np.asarray([all_episodes[e][self.state_key][t] for e,t in zip(ep_idx, t_idx)])
      else:
        sampled_ags = np.asarray([self.obs2goal(all_episodes[e][self.state_key][t]) for e,t in zip(ep_idx, t_idx)])

      if self.obs2goal is not None:
        sampled_ags = self.obs2goal(sampled_ags)

    # ============Q cutoff ================
    # NOT IMPORTANT FOR MAZE, so ignore.
    # # 1. get feat of state.
    # if state is None:
    #   latent = self.wm.rssm.initial(1)
    #   action = tf.zeros((1,1,) + self.act_space.shape)
    #   state = latent, action
    # else:
    #   latent, action = state
    # embed = self.wm.encoder(obs)
    # post, prior = self.wm.rssm.observe( # q(s' | e,a,s)
    #     embed, action, obs['is_first'], latent)
    # start_state = {k: v[:, -1] for k, v in post.items()}
    # start_state['feat'] = self.wm.rssm.get_feat(start_state) # (1, 1800)

    # start = tf.nest.map_structure(lambda x: tf.repeat(x, sampled_ags.shape[0],0), start_state)
    # goal_obs = start.copy()
    # goal_obs[self.state_key] = sampled_ags
    # goal_embed = self.wm.encoder(goal_obs)
    # feat_goal = tf.concat([start['feat'], goal_embed], -1)
    # q_values = self.agent._task_behavior.critic(feat_goal).mode()
    # import ipdb; ipdb.set_trace()
    # bad_q_idxs = q_values < self.cutoff
    q_values = None
    bad_q_idxs = None

    # ============ Scoring ==================
    sampled_ag_scores = self.evaluate_log_density(sampled_ags)
    # Take softmax of the alpha * log density.
    # If alpha = -1, this gives us normalized inverse densities (higher is rarer)
    # If alpha < -1, this skews the density to give us low density samples
    normalized_inverse_densities = softmax(sampled_ag_scores * self.alpha)
    normalized_inverse_densities *= -1.  # make negative / reverse order so that lower is better.
    goal_values = normalized_inverse_densities
    # ============ Get Minimum Density Goals ===========
    if q_values is not None:
      goal_values[bad_q_idxs] = q_values[bad_q_idxs] * -1e-8

    if self.random:
      abs_goal_values = np.abs(goal_values)
      normalized_values = abs_goal_values / np.sum(abs_goal_values, axis=0, keepdims=True)
      # chosen_idx = (normalized_values.cumsum(0) > np.random.rand(normalized_values.shape[0])).argmax(0)
      chosen_idx = np.random.choice(len(abs_goal_values), 1, replace=True, p=normalized_values)[0]
    else:
      chosen_idx = np.argmin(goal_values)
    chosen_ags = sampled_ags[chosen_idx]

    # Store if we need the MEGA goal distribution
    self.sampled_ags = sampled_ags
    self.goal_values = goal_values
    return chosen_ags

class SubgoalPlannerKDE:
  def __init__(self, agent, replay, act_space, state_key, ep_length, obs2goal,
      gc_input,
      goal_dim, # D dims
      goal_min, #  D dims for min / max
      goal_max, #  D dims for min / max
      planner="shooting_cem",
      horizon=15,
      mpc_steps=10,
      batch=5,
      cem_elite_ratio=0.2,
      optimization_steps=5,
      std_scale=1.0,
      mppi_gamma=10.0,
      init_candidates=None,
      dataset=None,
      evaluate_only=False, #  don't run CEM, just evaluate goals with model.
      repeat_samples=0,
      mega_prior=False, # an instance of MEGA
      vis_fn=None,
  ):
    self.agent = agent
    self.replay = replay
    self.wm = agent.wm
    self.actor = agent._task_behavior.actor

    self.act_space = act_space
    if isinstance(act_space, dict):
      self.act_space = act_space['action']
    ## KDE STUFF
    from sklearn.neighbors import KernelDensity
    self.alpha = -1.0
    self.kernel = 'gaussian'
    self.bandwidth = 0.1
    self.kde = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
    self.kde_sample_mean = 0.
    self.kde_sample_std = 1.

    self.state_key = state_key
    self.ready = False
    self.random = False
    self.ep_length = ep_length

    self.gc_input = gc_input
    self.obs2goal = obs2goal
    self.goal_dim = goal_dim
    self.planner = planner
    self.horizon = horizon
    self.mpc_steps = mpc_steps
    self.batch = batch
    self.cem_elite_ratio = cem_elite_ratio
    self.optimization_steps = optimization_steps
    self.std_scale = std_scale
    self.mppi_gamma= mppi_gamma

    self.min_action = goal_min
    self.max_action = goal_max

    self.mega = mega_prior
    self.init_distribution = None
    if init_candidates is not None:
      self.create_init_distribution(init_candidates)

    self.dataset = dataset
    self.evaluate_only = evaluate_only
    if self.evaluate_only:
      assert (self.dataset is not None or self.mega is not None), "need to sample from something."

    self.repeat_samples = repeat_samples

    self.vis_fn = vis_fn
    self.will_update_next_call = True

  def update_kde(self):
    self.ready = True
    # Follow RawKernelDensity in density.py of MRL codebase.

    # ========== Sample Goals=============
    num_episodes = self.replay.stats['loaded_episodes']
    # sample 10K goals from the buffer.
    num_samples = min(10000, self.replay.stats['loaded_steps'])
    # first uniformly sample from episodes.
    ep_idx = np.random.randint(0, num_episodes, num_samples)
    # uniformly sample from timesteps
    t_idx = np.random.randint(0, self.ep_length, num_samples)
    # store all these goals.
    all_episodes = list(self.replay._complete_eps.values())
    kde_samples = [all_episodes[e][self.state_key][t] for e,t in zip(ep_idx, t_idx)]
    # normalize goals
    self.kde_sample_mean = np.mean(kde_samples, axis=0, keepdims=True)
    self.kde_sample_std  = np.std(kde_samples, axis=0, keepdims=True) + 1e-4
    kde_samples = (kde_samples - self.kde_sample_mean) / self.kde_sample_std
    # =========== Fit KDE ================
    self.fitted_kde = self.kde.fit(kde_samples)

  def evaluate_log_density(self, samples):
    assert self.ready, "ENSURE READY BEFORE EVALUATING LOG DENSITY"
    return self.fitted_kde.score_samples( (samples  - self.kde_sample_mean) / self.kde_sample_std )

  def search_goal(self, obs, state=None, mode='train'):
    if self.will_update_next_call is False:
      return self.sample_goal()

    # update kde
    self.update_kde()

    elite_size = int(self.batch * self.cem_elite_ratio)
    if state is None:
      latent = self.wm.rssm.initial(1)
      action = tf.zeros((1,1,) + self.act_space.shape)
      state = latent, action
      # print("make new state")
    else:
      latent, action = state
      action = tf.expand_dims(action, 0)
      # action should be (1,1, D)
      # print("using exisitng state")

    # create start state.
    embed = self.wm.encoder(obs)
    # posterior is q(s' | s,a,e)
    post, prior = self.wm.rssm.observe(
        embed, action, obs['is_first'], latent)
    init_start = {k: v[:, -1] for k, v in post.items()}
    # print(action.shape)
    # for k,v in latent.keys():
    #   print(k, v.shape)
    @tf.function
    def eval_fitness(goal):
      # should be (55,128).
      start = {k: v for k, v in init_start.items()}
      start['feat'] = self.wm.rssm.get_feat(start) # (1, 1800)
      start = tf.nest.map_structure(lambda x: tf.repeat(x, goal.shape[0],0), start)
      goal_obs = start.copy()
      goal_obs[self.state_key] = goal
      goal_embed = self.wm.encoder(goal_obs)

      actor_inp = tf.concat([start['feat'], goal_embed], -1)
      start['action'] = tf.zeros_like(self.actor(actor_inp).mode())
      seq = {k: [v] for k, v in start.items()}
      for _ in range(self.horizon):
        actor_inp = tf.concat([seq['feat'][-1], goal_embed], -1)
        action = self.actor(actor_inp).sample()
        state = self.wm.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
        feat = self.wm.rssm.get_feat(state)
        for key, value in {**state, 'action': action, 'feat': feat}.items():
          seq[key].append(value)
      seq = {k: tf.stack(v, 0) for k, v in seq.items()}
      # rewards should be (batch,1)
      states = self.wm.heads['decoder'](seq['feat'])[self.state_key].mode()
      return states, seq

    # CEM loop
    # rewards = []
    # act_losses = []
    if self.init_distribution is None:
      means, stds = self.get_distribution_from_obs(obs)
    else:
      means, stds = self.init_distribution

    opt_steps = 1 if self.evaluate_only else self.optimization_steps
    for i in range(opt_steps):
      # Sample action sequences and evaluate fitness
      if i == 0 and (self.dataset or self.mega):
        if self.dataset:
          random_batch = next(self.dataset)
          random_batch = self.wm.preprocess(random_batch)
          samples = tf.reshape(random_batch[self.state_key], (-1,) + tuple(random_batch[self.state_key].shape[2:]))
          if self.obs2goal is not None:
            samples = self.obs2goal(samples)
          # initialize means states.
          means, vars = tf.nn.moments(samples, 0)
          stds = tf.sqrt(vars + 1e-6)
          # assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
          samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
          # print(i, samples)
          samples = tf.clip_by_value(samples, self.min_action, self.max_action)
        elif self.mega:
          samples = self.mega.sample_goal(obs)[None]
          # create initial goal samples
          if self.evaluate_only:
            # use MEGA distribution
            values = -self.mega.goal_values
            # use top 100
            mega_score, mega_inds = tf.nn.top_k(values, min(len(values), 100), sorted=False)
            samples = tf.gather(self.mega.sampled_ags, mega_inds)
          else:
            # since mega's best goal and repeat it.
            samples = tf.repeat(samples, self.batch, 0)
            # initialize means states.
            means, vars = tf.nn.moments(samples, 0)
            stds = tf.sqrt(vars + 1e-6)
            # TODO: remove this std hack
            stds = tf.concat([[0.5, 0.5], stds[2:]], axis=0)
            # assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
            samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
            # print(i, samples)
            samples = tf.clip_by_value(samples, self.min_action, self.max_action)
      else:
        samples = tfd.MultivariateNormalDiag(means, stds).sample(sample_shape=[self.batch])
        # print(i, samples)
        samples = tf.clip_by_value(samples, self.min_action, self.max_action)

      if self.repeat_samples > 1:
        samples = tf.repeat(samples, self.repeat_samples, 0)
      # fitness, seq = eval_fitness(samples)
      states, seq = eval_fitness(samples)
      # score these states with kde.
      _states = states
      if self.agent.config.planner.final_step_cost:
        # Take the last 10 steps instead of the entire trajectory.
        _states = states[-10:]

      densities = self.evaluate_log_density(tf.reshape(_states, (-1, _states.shape[-1])))
      densities = tf.reshape(densities, (*_states.shape[:2],))
      fitness = tf.reduce_sum(-densities, 0)
      # Refit distribution to elite samples
      if self.planner == 'shooting_mppi':
        # MPPI
        weights = tf.expand_dims(tf.nn.softmax(self.mppi_gamma * fitness), axis=1)
        means = tf.reduce_sum(weights * samples, axis=0)
        stds = tf.sqrt(tf.reduce_sum(weights * tf.square(samples - means), axis=0))
        # rewards.append(tf.reduce_sum(fitness * weights[:, 0]).numpy())
      elif self.planner == 'shooting_cem':
        # CEM
        elite_score, elite_inds = tf.nn.top_k(fitness, elite_size, sorted=False)
        elite_samples = tf.gather(samples, elite_inds)
        # print(elite_samples)
        means, vars = tf.nn.moments(elite_samples, 0)
        stds = tf.sqrt(vars + 1e-6)
        # rewards.append(tf.reduce_mean(tf.gather(fitness, elite_inds)).numpy())

    self.vis_fn(elite_inds, elite_samples, seq, self.wm)
    # TODO: potentially store these as initialization for the next update.
    self.means = means
    self.stds = stds

    if self.evaluate_only:
      self.elite_samples = elite_samples
      self.elite_score = elite_score

    return self.sample_goal()

  def sample_goal(self, batch=1):
    if self.evaluate_only:
      # samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])
      weights = self.elite_score / self.elite_score.sum()
      # weights = tf.nn.softmax(self.elite_score)
      idxs = tf.squeeze(tf.random.categorical(tf.math.log([weights]), batch), 0)
      samples = tf.gather(self.elite_samples, idxs)
    else:
      samples = tfd.MultivariateNormalDiag(self.means, self.stds).sample(sample_shape=[batch])
    return samples

  def create_init_distribution(self, init_candidates):
    """Create the starting distribution for seeding the planner.
    """
    def _create_init_distribution(init_candidates):
      means = tf.reduce_mean(init_candidates, 0)
      stds = tf.math.reduce_std(init_candidates, 0)
      # if there's only 1 candidate, set std to default
      if init_candidates.shape[0] == 1:
        stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
      return means, stds
    self.init_distribution = _create_init_distribution(init_candidates)

  def get_distribution_from_obs(self, obs):
    ob = tf.squeeze(obs[self.state_key])
    if self.gc_input == "state":
      ob = self.obs2goal(ob)
    means = tf.cast(tf.identity(ob), tf.float32)
    assert np.prod(means.shape) == self.goal_dim, f"{np.prod(means.shape)}, {self.goal_dim}"
    stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
    init_distribution = tf.identity(means), tf.identity(stds)
    return init_distribution

  def get_init_distribution(self):
    if self.init_distribution is None:
      means = tf.zeros(self.goal_dim, dtype=tf.float32)
      stds = tf.ones(self.goal_dim, dtype=tf.float32) * self.std_scale
      self.init_distribution = tf.identity(means), tf.identity(stds)

    return self.init_distribution


class Skewfit(MEGA):
  def __init__(self, agent, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn):
    super().__init__(agent, replay, act_space, state_key, ep_length, obs2goal_fn, goal_sample_fn)
    self.random = True

def softmax(X, theta=1.0, axis=None):
  """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

  # make X at least 2d
  y = np.atleast_2d(X)

  # find axis
  if axis is None:
    axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

  # multiply y against the theta parameter,
  y = y * float(theta)

  # subtract the max for numerical stability
  y = y - np.max(y, axis=axis, keepdims=True)

  # exponentiate y
  y = np.exp(y)

  # take the sum along the specified axis
  ax_sum = np.sum(y, axis=axis, keepdims=True)

  # finally: divide elementwise
  p = y / ax_sum

  # flatten if X was 1D
  if len(X.shape) == 1: p = p.flatten()

  return p
