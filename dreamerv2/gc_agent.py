import tensorflow as tf
from tensorflow.keras import mixed_precision as prec
import numpy as np

import common
import expl
import agent


class GCAgent(common.Module):

  def __init__(self, config, obs_space, act_space, step, obs2goal, sample_env_goals):
    self.config = config
    # TODO: assumes we're doing state based envs.
    self.state_key = config.state_key
    self.goal_key = config.goal_key
    self.obs_space = obs_space
    goal_dim = np.prod(self.obs_space[self.goal_key].shape)
    self.obs_space.pop(self.goal_key)

    self.act_space = act_space['action']
    self.step = step
    self.tfstep = tf.Variable(int(self.step), tf.int64)
    self.wm = GCWorldModel(config, obs_space, self.tfstep, obs2goal, sample_env_goals)
    self._task_behavior = GCActorCritic(config, self.act_space, self.tfstep, obs2goal, goal_dim)
    if config.expl_behavior == 'greedy':
      self._expl_behavior = self._task_behavior
    else:
      self._expl_behavior = getattr(expl, config.expl_behavior)(
          self.config, self.act_space, self.wm, self.tfstep,
          lambda seq: self.wm.heads['reward'](seq['feat']).mode())

  @tf.function
  def expl_policy(self, obs, state=None, mode='train'):
    if self.config.expl_behavior == 'greedy':
      return self.policy(obs, state, mode)

    # run the plan2expl policy (not goal cond.)
    obs = tf.nest.map_structure(tf.tensor, obs)
    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(self.wm.preprocess(obs))
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample)
    feat = self.wm.rssm.get_feat(latent)
    actor = self._expl_behavior.actor(feat)
    action = actor.sample()
    noise = self.config.expl_noise
    action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def policy(self, obs, state=None, mode='train'):
    obs = tf.nest.map_structure(tf.tensor, obs)
    obs = self.wm.preprocess(obs)
    assert mode in {'train', 'eval'}
     # use given goal
    goal = self.wm.get_goal(obs, training=False) # just use current goal from obs

    tf.py_function(lambda: self.tfstep.assign(
        int(self.step), read_value=False), [], [])
    if state is None:
      latent = self.wm.rssm.initial(len(obs['reward']))
      action = tf.zeros((len(obs['reward']),) + self.act_space.shape)
      state = latent, action
    latent, action = state
    embed = self.wm.encoder(obs)
    sample = (mode == 'train') or not self.config.eval_state_mean
    latent, _ = self.wm.rssm.obs_step(
        latent, action, embed, obs['is_first'], sample)
    feat = self.wm.rssm.get_feat(latent)
    actor_inp = tf.concat([feat, goal], -1)
    if mode == 'eval':
      actor = self._task_behavior.actor(actor_inp)
      action = actor.mode()
      noise = self.config.eval_noise
    elif mode == 'explore':
      actor = self._expl_behavior.actor(actor_inp)
      action = actor.sample()
      noise = self.config.expl_noise
    elif mode == 'train':
      actor = self._task_behavior.actor(actor_inp)
      action = actor.sample()
      noise = self.config.expl_noise
    if self.config.epsilon_expl_noise > 0 and mode != 'eval':
      action = common.epsilon_action_noise(action, self.config.epsilon_expl_noise, self.act_space)
    else:
      action = common.action_noise(action, noise, self.act_space)
    outputs = {'action': action}
    state = (latent, action)
    return outputs, state

  @tf.function
  def train_gcp(self, data, state=None):
    metrics = {}
    pdata = self.wm.preprocess(data)
    embed = self.wm.encoder(pdata)
    start, _ = self.wm.rssm.observe(
        embed, pdata['action'], pdata['is_first'], state)
    state = {k: v[:, -1] for k, v in start.items()}
    metrics.update(self._task_behavior.train(self.wm, start, data['is_terminal'], obs=data))
    return state, metrics

  @tf.function
  def train(self, data, state=None):
    metrics = {}
    state, outputs, mets = self.wm.train(data, state)
    metrics.update(mets)
    start = outputs['post']
    metrics.update(self._task_behavior.train(
        self.wm, start, data['is_terminal'], obs=data))
    if self.config.expl_behavior != 'greedy':
      mets = self._expl_behavior.train(start, outputs, data)[-1]
      metrics.update({'expl_' + key: value for key, value in mets.items()})
    return state, metrics

  # @tf.function
  def report(self, data, env, video_from_state_fn=None):
    report = {}
    data = self.wm.preprocess(data)
    if video_from_state_fn is not None:
      recon, openl, truth = self.wm.state_pred(data)
      report[f'openl_{self.state_key}'] = video_from_state_fn(recon, openl, truth, env)
    else: # image based env
      for key in self.wm.heads['decoder'].cnn_keys:
        name = key.replace('/', '_')
        report[f'openl_{name}'] = self.wm.video_pred(data, key)
    return report

  @tf.function
  def temporal_dist(self, obs):
    dist = self._task_behavior.subgoal_dist(self.wm, obs)
    if self.config.gc_reward == 'dynamical_distance' and self.config.dd_norm_reg_label:
        dist *= self._task_behavior.dd_seq_len
    return dist


class GCWorldModel(agent.WorldModel):

  def __init__(self, config, obs_space, tfstep, obs2goal, sample_env_goals):
    shapes = {k: tuple(v.shape) for k, v in obs_space.items()}
    self.config = config
    self.state_key = config.state_key
    self.goal_key = config.goal_key
    self.tfstep = tfstep
    self.obs2goal = obs2goal
    self.sample_env_goals = sample_env_goals
    self.rssm = common.EnsembleRSSM(**config.rssm)
    self.encoder = common.Encoder(shapes, **config.encoder)
    self.embed_size = self.encoder.embed_size
    self.heads = {}
    self.heads['decoder'] = common.Decoder(shapes, **config.decoder)
    if config.pred_reward:
      self.heads['reward'] = common.MLP([], **config.reward_head)
    if config.pred_discount:
      self.heads['discount'] = common.MLP([], **config.discount_head)
    if config.pred_embed:
      self.heads['embed'] = common.MLP([self.embed_size], **config.embed_head)
    # for name in config.grad_heads:
    #   assert name in self.heads, name
    self.model_opt = common.Optimizer('model', **config.model_opt)
    self.dtype = prec.global_policy().compute_dtype

  def train(self, data, state=None):
    data = self.preprocess(data)
    with tf.GradientTape() as model_tape:
      model_loss, state, outputs, metrics = self.loss(data, state)
    modules = [self.encoder, self.rssm, *self.heads.values()]
    metrics.update(self.model_opt(model_tape, model_loss, modules))
    return state, outputs, metrics

  def loss(self, data, state=None):
    # wm_start = time()
    embed = self.encoder(data)
    data['embed'] = tf.cast(tf.stop_gradient(embed), tf.float32) # Needed for the embed head
    post, prior = self.rssm.observe(
        embed, data['action'], data['is_first'], state)
    # wm_duration = time() - wm_start
    # print("wm loss1/2 duration", wm_duration)
    kl_loss, kl_value = self.rssm.kl_loss(post, prior, **self.config.kl)
    # scale = 1.0
    # free = self.config.kl['free']
    # balance = self.config.kl['balance']
    # kl_loss, kl_value = self.rssm.lexa_kl_loss(post, prior, balance, free, scale)
    assert len(kl_loss.shape) == 0
    likes = {}
    losses = {'kl': kl_loss}
    feat = self.rssm.get_feat(post)
    for name, head in self.heads.items():
      grad_head = (name in self.config.grad_heads)
      inp = feat if grad_head else tf.stop_gradient(feat)
      out = head(inp)
      dists = out if isinstance(out, dict) else {name: out}
      for key, dist in dists.items():
        like = tf.cast(dist.log_prob(data[key]), tf.float32)
        likes[key] = like
        losses[key] = -like.mean()
    model_loss = sum(
        self.config.loss_scales.get(k, 1.0) * v for k, v in losses.items())
    outs = dict(
        embed=embed, feat=feat, post=post,
        prior=prior, likes=likes, kl=kl_value)
    metrics = {f'{name}_loss': value for name, value in losses.items()}
    metrics['model_kl'] = kl_value.mean()
    metrics['prior_ent'] = self.rssm.get_dist(prior).entropy().mean()
    metrics['post_ent'] = self.rssm.get_dist(post).entropy().mean()
    last_state = {k: v[:, -1] for k, v in post.items()}
    return model_loss, last_state, outs, metrics

  def imagine(self, policy, start, is_terminal, horizon, goal=None):
    if goal is None: # happens when plan2expl actor trains in imag.
      return super().imagine(policy, start, is_terminal, horizon)

    flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
    goal = flatten(goal)
    start = {k: flatten(v) for k, v in start.items()}
    start['feat'] = self.rssm.get_feat(start)
    actor_inp = tf.concat([start['feat'], goal], -1)
    start['action'] = tf.zeros_like(policy(actor_inp).mode())
    seq = {k: [v] for k, v in start.items()}
    for _ in range(horizon):
      # print(seq['feat'][-1].shape, goal.shape)
      actor_inp = tf.concat([seq['feat'][-1], goal], -1)
      action = policy(tf.stop_gradient(actor_inp)).sample()
      state = self.rssm.img_step({k: v[-1] for k, v in seq.items()}, action)
      feat = self.rssm.get_feat(state)
      for key, value in {**state, 'action': action, 'feat': feat}.items():
        seq[key].append(value)
    seq = {k: tf.stack(v, 0) for k, v in seq.items()}
    if 'discount' in self.heads:
      disc = self.heads['discount'](seq['feat']).mean()
      if is_terminal is not None:
        # Override discount prediction for the first step with the true
        # discount factor from the replay buffer.
        true_first = 1.0 - flatten(is_terminal).astype(disc.dtype)
        true_first *= self.config.discount
        disc = tf.concat([true_first[None], disc[1:]], 0)
    else:
      disc = self.config.discount * tf.ones(seq['feat'].shape[:-1])
    seq['discount'] = disc
    # Shift discount factors because they imply whether the following state
    # will be valid, not whether the current state is valid.
    seq['weight'] = tf.math.cumprod(
        tf.concat([tf.ones_like(disc[:1]), disc[:-1]], 0), 0)
    return seq

  @tf.function
  def preprocess(self, obs):
    dtype = prec.global_policy().compute_dtype
    obs = obs.copy()
    for key, value in obs.items():
      if key.startswith('log_'):
        continue
      if value.dtype == tf.int32:
        value = value.astype(dtype)
      if value.dtype == tf.uint8:
        value = value.astype(dtype) / 255.0 - 0.5
      obs[key] = value
    obs['reward'] = {
        'identity': tf.identity,
        'sign': tf.sign,
        'tanh': tf.tanh,
    }[self.config.clip_rewards](obs['reward'])
    obs['discount'] = 1.0 - obs['is_terminal'].astype(dtype)
    obs['discount'] *= self.config.discount
    return obs

  @tf.function
  def video_pred(self, data, key):
    decoder = self.heads['decoder']
    truth = data[key][:6] + 0.5
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode()
    model = tf.concat([recon[:, :5] + 0.5, openl + 0.5], 1)
    error = (model - truth + 1) / 2
    video = tf.concat([truth, model, error], 2)
    B, T, H, W, C = video.shape
    return video.transpose((1, 2, 0, 3, 4)).reshape((T, H, B * W, C))

  @tf.function
  def state_pred(self, data):
    key = self.state_key
    decoder = self.heads['decoder']
    truth = data[key][:6]
    embed = self.encoder(data)
    states, _ = self.rssm.observe(
        embed[:6, :5], data['action'][:6, :5], data['is_first'][:6, :5])
    recon = decoder(self.rssm.get_feat(states))[key].mode()[:6]
    init = {k: v[:, -1] for k, v in states.items()}
    prior = self.rssm.imagine(data['action'][:6, 5:], init)
    openl = decoder(self.rssm.get_feat(prior))[key].mode()
    return recon, openl, truth

  def get_goal(self, obs, training=False):
    if self.config.gc_input == 'state':
      if (not training) or self.config.training_goals == 'env':
        goal = tf.cast(obs[self.goal_key], self.dtype)
        return goal
      elif self.config.training_goals == 'batch':
        # Use random goals from the same batch
        # This is only run during imagination training
        goal_embed = tf.cast(self.obs2goal(obs[self.state_key]), self.dtype)
        sh = goal_embed.shape
        goal_embed = tf.reshape(goal_embed, (-1, sh[-1]))

        ids = tf.random.shuffle(tf.range(tf.shape(goal_embed)[0]))
        goal_embed = tf.gather(goal_embed, ids)
        goal_embed = tf.reshape(goal_embed, sh)
        return goal_embed
    else:
      if (not training) or self.config.training_goals == 'env':
        # Never alter the goal when evaluating
        goal_obs = obs.copy()
        goal_obs[self.state_key] = obs[self.goal_key]
        _embed = self.encoder(goal_obs)
        if self.config.gc_input == 'embed':
          return _embed
        elif 'feat' in self.config.gc_input:
          return self.get_init_feat_embed(_embed) if len(_embed.shape) == 2 else tf.vectorized_map(self.get_init_feat_embed, _embed)
      elif self.config.training_goals == 'batch':
        if self.config.train_env_goal_percent > 0:
          orig_ag_sh = obs[self.state_key].shape
          num_goals = tf.math.reduce_prod(orig_ag_sh[:-1])
          num_dgs = tf.cast(tf.cast(num_goals, tf.float32) * self.config.train_env_goal_percent, tf.int32)
          num_ags = num_goals - num_dgs
          flat_ags = tf.reshape(obs[self.state_key], (-1, obs[self.state_key].shape[-1]))
          # flat_dgs = tf.reshape(obs[self.goal_key], (-1, obs[self.goal_key].shape[-1]))
          ag_ids = tf.random.shuffle(tf.range(tf.shape(flat_ags)[0]))[:num_ags]
          # dg_ids = tf.random.shuffle(tf.range(tf.shape(flat_dgs)[0]))[:num_dgs]
          sel_ags = tf.gather(flat_ags, ag_ids)
          assert self.sample_env_goals is not None, "need to support sample_env_goals"
          sel_dgs = self.sample_env_goals(num_dgs)
          all_goals = tf.concat([sel_ags, sel_dgs], 0)
          goal_embed = self.encoder({self.state_key: all_goals})
          # shuffle one more time to mix dgs and ags
          ids = tf.random.shuffle(tf.range(tf.shape(goal_embed)[0]))
          goal_embed = tf.gather(goal_embed, ids)
          goal_embed = tf.reshape(goal_embed, (*orig_ag_sh[:-1], goal_embed.shape[-1]))
        else:
          # Use random goals from the same batch
          # This is only run during imagination training
          goal_embed = self.encoder(obs)
          sh = goal_embed.shape
          goal_embed = tf.reshape(goal_embed, (-1, sh[-1]))
          # goal_embed = tf.random.shuffle(goal_embed)  # shuffle doesn't have gradients so need this workaround...
          ids = tf.random.shuffle(tf.range(tf.shape(goal_embed)[0]))
          goal_embed = tf.gather(goal_embed, ids)
          goal_embed = tf.reshape(goal_embed, sh)

        if 'feat' in self.config.gc_input:
          return tf.vectorized_map(self.get_init_feat_embed, goal_embed)
        else:
          return goal_embed


class GCActorCritic(common.Module):

  def __init__(self, config, act_space, tfstep, obs2goal, goal_dim):
    self.config = config
    self.state_key = config.state_key
    self.dtype = prec.global_policy().compute_dtype
    self.act_space = act_space
    self.tfstep = tfstep
    self.obs2goal = obs2goal
    self.goal_dim = goal_dim
    discrete = hasattr(act_space, 'n')
    if self.config.actor.dist == 'auto':
      self.config = self.config.update({
          'actor.dist': 'onehot' if discrete else 'trunc_normal'})
    if self.config.actor_grad == 'auto':
      self.config = self.config.update({
          'actor_grad': 'reinforce' if discrete else 'dynamics'})
    self.actor = common.MLP(act_space.shape[0], **self.config.actor)
    self.critic = common.MLP([], **self.config.critic)
    if self.config.slow_target:
      self._target_critic = common.MLP([], **self.config.critic)
      self._updates = tf.Variable(0, tf.int64)
    else:
      self._target_critic = self.critic
    self.actor_opt = common.Optimizer('actor', **self.config.actor_opt)
    self.critic_opt = common.Optimizer('critic', **self.config.critic_opt)
    self.rewnorm = common.StreamNorm(**self.config.reward_norm)
    if config.gc_reward == "dynamical_distance":
      dd_out_dim = 1
      self.dd_loss_fn = tf.keras.losses.MSE
      self.dd_seq_len = self.config.imag_horizon
      self.dd_out_dim = dd_out_dim
      self.dynamical_distance = common.GC_Distance(out_dim = dd_out_dim, input_type= self.config.dd_inp, units=400, normalize_input = self.config.dd_norm_inp)
      self.dd_cur_idxs, self.dd_goal_idxs = get_future_goal_idxs(seq_len = self.config.imag_horizon, bs = self.config.dataset.batch*self.config.dataset.length)
      self._dd_opt = common.Optimizer(
            'dyn_dist', **config.dd_opt)

  def train(self, world_model, start, is_terminal,  obs=None):
    metrics = {}
    hor = self.config.imag_horizon
    # The weights are is_terminal flags for the imagination start states.
    # Technically, they should multiply the losses from the second trajectory
    # step onwards, which is the first imagined step. However, we are not
    # training the action that led into the first step anyway, so we can use
    # them to scale the whole sequence.
    with tf.GradientTape() as actor_tape:
      obs = world_model.preprocess(obs)
      goal = world_model.get_goal(obs, training=True) # get goal embeddings from same batch.
      # start is Batch x Length x D.
      seq = world_model.imagine(self.actor, start, is_terminal, hor, goal) # Seq is Horizon x (Batch x Length) x D.
      # reward = reward_fn(seq)
      imag_feat =  seq['feat']
      imag_state = seq
      imag_action = seq['action']
      actor_inp = get_actor_inp(imag_feat, goal) # add goal embed to input embedding
      seq['feat_goal'] = actor_inp
      reward = self._gc_reward(world_model, actor_inp, imag_state, imag_action, obs)
      seq['reward'], mets1 = self.rewnorm(reward)
      mets1 = {f'reward_{k}': v for k, v in mets1.items()}
      target, mets2 = self.target(seq)
      actor_loss, mets3 = self.actor_loss(seq, target)
    with tf.GradientTape() as critic_tape:
      critic_loss, mets4 = self.critic_loss(seq, target)

    mets5 = {}
    if self.config.gc_reward == "dynamical_distance":
      with tf.GradientTape() as df_tape:
        if self.config.gc_input == 'embed':
          _inp = world_model.heads['embed'](imag_feat).mode()
        elif self.config.gc_input == 'state':
          _inp = world_model.heads['decoder'](imag_feat)[self.state_key].mode()
          _inp = tf.cast(self.obs2goal(_inp), self.dtype)
        dd_loss, mets5 = self.get_dynamical_distance_loss(_inp)
      metrics.update(self._dd_opt(df_tape, dd_loss, self.dynamical_distance))

    metrics.update(self.actor_opt(actor_tape, actor_loss, self.actor))
    metrics.update(self.critic_opt(critic_tape, critic_loss, self.critic))
    metrics.update(**mets1, **mets2, **mets3, **mets4, **mets5)
    self.update_slow_target()  # Variables exist after first forward pass.
    return metrics

  def actor_loss(self, seq, target):
    # Actions:      0   [a1]  [a2]   a3
    #                  ^  |  ^  |  ^  |
    #                 /   v /   v /   v
    # States:     [z0]->[z1]-> z2 -> z3
    # Targets:     t0   [t1]  [t2]
    # Baselines:  [v0]  [v1]   v2    v3
    # Entropies:        [e1]  [e2]
    # Weights:    [ 1]  [w1]   w2    w3
    # Loss:              l1    l2
    metrics = {}
    # Two states are lost at the end of the trajectory, one for the boostrap
    # value prediction and one because the corresponding action does not lead
    # anywhere anymore. One target is lost at the start of the trajectory
    # because the initial state comes from the replay buffer.
    policy = self.actor(tf.stop_gradient(seq['feat_goal'][:-2]))
    if self.config.actor_grad == 'dynamics':
      objective = target[1:]
    elif self.config.actor_grad == 'reinforce':
      baseline = self._target_critic(seq['feat_goal'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      action = tf.stop_gradient(seq['action'][1:-1])
      objective = policy.log_prob(action) * advantage
    elif self.config.actor_grad == 'both':
      baseline = self._target_critic(seq['feat_goal'][:-2]).mode()
      advantage = tf.stop_gradient(target[1:] - baseline)
      objective = policy.log_prob(seq['action'][1:-1]) * advantage
      mix = common.schedule(self.config.actor_grad_mix, self.tfstep)
      objective = mix * target[1:] + (1 - mix) * objective
      metrics['actor_grad_mix'] = mix
    else:
      raise NotImplementedError(self.config.actor_grad)
    ent = policy.entropy()
    ent_scale = common.schedule(self.config.actor_ent, self.tfstep)
    objective += ent_scale * ent
    weight = tf.stop_gradient(seq['weight'])
    actor_loss = -(weight[:-2] * objective).mean()
    metrics['actor_ent'] = ent.mean()
    metrics['actor_ent_scale'] = ent_scale
    return actor_loss, metrics

  def critic_loss(self, seq, target):
    # States:     [z0]  [z1]  [z2]   z3
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]   v3
    # Weights:    [ 1]  [w1]  [w2]   w3
    # Targets:    [t0]  [t1]  [t2]
    # Loss:        l0    l1    l2
    dist = self.critic(seq['feat_goal'][:-1])
    target = tf.stop_gradient(target)
    weight = tf.stop_gradient(seq['weight'])
    critic_loss = -(dist.log_prob(target) * weight[:-1]).mean()
    metrics = {'critic': dist.mode().mean()}
    return critic_loss, metrics

  def target(self, seq):
    # States:     [z0]  [z1]  [z2]  [z3]
    # Rewards:    [r0]  [r1]  [r2]   r3
    # Values:     [v0]  [v1]  [v2]  [v3]
    # Discount:   [d0]  [d1]  [d2]   d3
    # Targets:     t0    t1    t2
    reward = tf.cast(seq['reward'], tf.float32)
    disc = tf.cast(seq['discount'], tf.float32)
    value = self._target_critic(seq['feat_goal']).mode()
    # Skipping last time step because it is used for bootstrapping.
    target = common.lambda_return(
        reward[:-1], value[:-1], disc[:-1],
        bootstrap=value[-1],
        lambda_=self.config.discount_lambda,
        axis=0)
    metrics = {}
    metrics['critic_slow'] = value.mean()
    metrics['critic_target'] = target.mean()
    return target, metrics

  def update_slow_target(self):
    if self.config.slow_target:
      if self._updates % self.config.slow_target_update == 0:
        mix = 1.0 if self._updates == 0 else float(
            self.config.slow_target_fraction)
        for s, d in zip(self.critic.variables, self._target_critic.variables):
          d.assign(mix * s + (1 - mix) * d)
      self._updates.assign_add(1)

  def get_dynamical_distance_loss(self, _data, corr_factor = None):
    metrics = {}
    seq_len, bs = _data.shape[:2]
    # pred = tf.cast(self.dynamical_distance(tf.concat([_data, _data], axis=-1)), tf.float32)
    # _label = 1.0
    # loss = tf.reduce_mean((_label-pred)**2)
    # return loss, metrics

    def _helper(cur_idxs, goal_idxs, distance):
      loss = 0
      cur_states = tf.expand_dims(tf.gather_nd(_data, cur_idxs),0)
      goal_states = tf.expand_dims(tf.gather_nd(_data, goal_idxs),0)
      pred = tf.cast(self.dynamical_distance(tf.concat([cur_states, goal_states], axis=-1)), tf.float32)

      if self.config.dd_loss == 'regression':
        _label = distance
        if self.config.dd_norm_reg_label and self.config.dd_distance == 'steps_to_go':
          _label = _label/self.dd_seq_len
        loss += tf.reduce_mean((_label-pred)**2)
      else:
        _label = tf.one_hot(tf.cast(distance, tf.int32), self.dd_out_dim)
        loss += self.dd_loss_fn(_label, pred)
      return loss

    #positives
    idxs = np.random.choice(np.arange(len(self.dd_cur_idxs)), self.config.dd_num_positives)
    loss = _helper(self.dd_cur_idxs[idxs], self.dd_goal_idxs[idxs], self.dd_goal_idxs[idxs][:,0] - self.dd_cur_idxs[idxs][:,0])
    # metrics['dd_pos_loss'] = loss

    #negatives
    corr_factor = corr_factor if corr_factor != None else self.config.dataset.length
    if self.config.dd_neg_sampling_factor>0:
      num_negs = int(self.config.dd_neg_sampling_factor*self.config.dd_num_positives)
      neg_cur_idxs, neg_goal_idxs = get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, corr_factor)
      neg_loss = _helper(neg_cur_idxs, neg_goal_idxs, tf.ones(num_negs)*seq_len)
      loss += neg_loss
      # metrics['dd_neg_loss'] = neg_loss

    return loss, metrics


  def _gc_reward(self, world_model, feat, inp_state=None, action=None, obs=None):
    # feat is a tensor containing [inp_feat, goal_emb]
    #image embedding as goal
    if self.config.gc_input == 'embed':
      inp_feat, goal_embed = tf.split(feat, [-1, world_model.encoder.embed_size], -1)
      if self.config.gc_reward == 'l2':
        # goal_feat = tf.vectorized_map(self.world_model.get_init_feat_embed, goal_embed)
        # return -tf.reduce_mean((goal_feat - inp_feat) ** 2, -1)
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        goal_state = tf.nest.map_structure(lambda x: flatten(tf.zeros_like(x)), inp_state)
        goal_action = flatten(tf.zeros_like(action))
        is_first = flatten(tf.ones(action.shape[:2], dtype=tf.bool))
        goal_embed = flatten(goal_embed)
        goal_latent, _ = world_model.rssm.obs_step(goal_state, goal_action, goal_embed, is_first, sample=False)
        goal_feat = world_model.rssm.get_feat(goal_latent)
        goal_feat = goal_feat.reshape(inp_feat.shape)
        return -tf.reduce_mean((goal_feat - inp_feat) ** 2, -1)

      elif self.config.gc_reward == 'cosine':
        goal_feat = tf.vectorized_map(self.world_model.get_init_feat_embed, goal_embed)
        norm = tf.norm(goal_feat, axis =-1)*tf.norm(inp_feat, axis = -1)
        dot_prod = tf.expand_dims(goal_feat,2)@tf.expand_dims(inp_feat,3)
        return tf.squeeze(dot_prod)/(norm+1e-8)

      elif self.config.gc_reward == 'dynamical_distance':
        inp_embed = tf.cast(world_model.heads['embed'](inp_feat).mode(), goal_embed.dtype)
        dd_out = self.dynamical_distance(tf.concat([inp_embed, goal_embed], axis =-1))
        reward = -dd_out
        if self.config.gc_reward_shape == 'sum_diff':
          # s1 a1 s2 a2 s3
          # r1 = d(s2) - d(s1)
          # r2 = d(s3) - d(s2)
          # r3 = 0, terminal.
          diff_reward = reward[1:] - reward[:1]
          reward = tf.concat([diff_reward, tf.zeros_like(diff_reward)[None,0]], 0)
        return reward
    elif self.config.gc_input == 'state':
      inp_feat, goal = tf.split(feat, [-1, self.goal_dim], -1)
      if self.config.gc_reward == 'dynamical_distance':
        # need to decode inp_feat to states and then convert to goal space
        # to compare against goal_embed.
        current = tf.cast(world_model.heads['decoder'](inp_feat)[self.state_key].mode(), goal.dtype)
        current = tf.cast(self.obs2goal(current), self.dtype)
        dd_out = self.dynamical_distance(tf.concat([current, goal], axis =-1))
        reward = -dd_out
        if self.config.gc_reward_shape == 'sum_diff':
          # s1 a1 s2 a2 s3
          # r1 = d(s2) - d(s1)
          # r2 = d(s3) - d(s2)
          # r3 = 0, terminal.
          diff_reward = reward[1:] - reward[:1]
          reward = tf.concat([diff_reward, tf.zeros_like(diff_reward)[None,0]], 0)
        return reward
      elif self.config.gc_reward == 'l2':
        current = tf.cast(world_model.heads['decoder'](inp_feat)[self.state_key].mode(), goal.dtype)
        current = tf.cast(self.obs2goal(current), self.dtype)
        # TODO: this is block stack specific, abstract out.
        threshold = 0.05
        num_blocks = (current.shape[-1] - 5)// 3
        current_per_obj = tf.concat([current[None, ..., :3], tf.stack(tf.split(current[..., 5:], num_blocks, axis=2))], axis=0)
        goal_per_obj = tf.concat([goal[None, ..., :3], tf.stack(tf.split(goal[..., 5:], num_blocks, axis=2))], axis=0)
        dist_per_obj = tf.sqrt(tf.reduce_sum((current_per_obj - goal_per_obj)**2, axis=-1))
        success_per_obj = tf.cast(dist_per_obj < threshold, self.dtype)
        grip_success =  success_per_obj[0]
        obj_success = tf.reduce_prod(success_per_obj[1:], axis=0)
        reward = 0.1 * grip_success +  obj_success
        return reward
        # return -tf.reduce_mean((goal - current) ** 2, -1)
      else:
        raise NotImplementedError


  def subgoal_dist(self, world_model, obs):
    """Directly converts to embedding with encoder.
    """
    obs = world_model.preprocess(obs)
    if self.config.gc_input == 'embed':
      ob_inp = world_model.encoder(obs)
    elif self.config.gc_input == 'state':
      ob_inp = tf.cast(self.obs2goal(obs[self.state_key]), self.dtype)

    goal_inp = world_model.get_goal(obs, training=False)
    if self.config.gc_reward == 'dynamical_distance':
      dist = self.dynamical_distance(tf.concat([ob_inp, goal_inp], axis =-1))
    elif self.config.gc_reward == 'l2':
      dist = tf.sqrt(tf.reduce_mean((goal_inp - ob_inp) ** 2))
    else:
      raise NotImplementedError
    return dist


def get_future_goal_idxs(seq_len, bs):

    cur_idx_list = []
    goal_idx_list = []
    #generate indices grid
    for cur_idx in range(seq_len):
      for goal_idx in range(cur_idx, seq_len):
        cur_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*cur_idx, np.arange(bs).reshape(-1,1)], axis = -1))
        goal_idx_list.append(np.concatenate([np.ones((bs,1), dtype=np.int32)*goal_idx, np.arange(bs).reshape(-1,1)], axis = -1))

    return np.concatenate(cur_idx_list,0), np.concatenate(goal_idx_list,0)

def get_future_goal_idxs_neg_sampling(num_negs, seq_len, bs, batch_len):
    cur_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    goal_idxs = np.random.randint((0,0), (seq_len, bs), size=(num_negs,2))
    for i in range(num_negs):
      goal_idxs[i,1] = np.random.choice([j for j in range(bs) if j//batch_len != cur_idxs[i,1]//batch_len])
    return cur_idxs, goal_idxs

def get_actor_inp(feat, goal, repeats=None):
  # Image and goal together - input to the actor
  goal = tf.reshape(goal, [1, feat.shape[1], -1])
  goal = tf.repeat(goal, feat.shape[0], 0)
  if repeats:
    goal = tf.repeat(tf.expand_dims(goal, 2), repeats,2)

  return tf.concat([feat, goal], -1)