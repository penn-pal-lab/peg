import numpy as np


class Driver:

  def __init__(self, envs, **kwargs):
    self._envs = envs
    self._kwargs = kwargs
    self._on_steps = []
    self._on_resets = []
    self._on_episodes = []
    self._act_spaces = [env.act_space for env in envs]
    self.reset()

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_reset(self, callback):
    self._on_resets.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def reset(self):
    self._obs = [None] * len(self._envs)
    self._eps = [None] * len(self._envs)
    self._state = None

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      obs = {
          i: self._envs[i].reset()
          for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
        self._eps[i] = [tran]
      obs = {k: np.stack([o[k] for o in self._obs]) for k in self._obs[0]}
      actions, self._state = policy(obs, self._state, **self._kwargs)
      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      obs = [e.step(a) for e, a in zip(self._envs, actions)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      for i, (act, ob) in enumerate(zip(actions, obs)):
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(tran)
        step += 1
        if ob['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          [fn(ep, **self._kwargs) for fn in self._on_episodes]
          episode += 1
      self._obs = obs

  def _convert(self, value):
    value = np.array(value)
    if np.issubdtype(value.dtype, np.floating):
      return value.astype(np.float32)
    elif np.issubdtype(value.dtype, np.signedinteger):
      return value.astype(np.int32)
    elif np.issubdtype(value.dtype, np.uint8):
      return value.astype(np.uint8)
    return value

class GCDriver(Driver):
  def __init__(self, envs, goal_key, **kwargs):
    super().__init__(envs, **kwargs)
    self.goal_key = goal_key

  def reset(self):
    super().reset()
    self._subgoals = [None] * len(self._envs)
    self._use_policy_2 = [False] * len(self._envs)
    self._goal_time = [0] * len(self._envs)
    self._goal_dist = [0] * len(self._envs) # store subgoal dist per episode.
    self._goal_success = [0] * len(self._envs) # store subgoal success per episode.

  def __call__(self, policy_1, policy_2=None, get_goal=None, steps=0, episodes=0, goal_time_limit=None, goal_checker=None):
    """
    1. train: run gcp for entire rollout using goals from buffer/search.
    2. expl: run plan2expl for entire rollout
    3. 2pol: run gcp with goals from buffer/search and then expl policy
    
    LEXA is (1,2) and choosing goals from buffer.
    Ours can be (1,2,3), or (1,3) and choosing goals from search
    
    Args:
        policy_1 (_type_): 1st policy to run in episode
        policy_2 (_type_, optional): 2nd policy that runs after first policy is done. If None, then only run 1st policy.
        goal_strategy (_type_, optional): How to sample a goal
        steps (int, optional): _description_. Defaults to 0.
        episodes (int, optional): _description_. Defaults to 0.
        goal_time_limit (_type_, optional): _description_. Defaults to None.
        goal_checker (_type_, optional): _description_. Defaults to None.
    """

    step, episode = 0, 0
    while step < steps or episode < episodes:
      # obs contains initial ob for all envs that need resetting.
      obs = {
          i: self._envs[i].reset()
          for i, ob in enumerate(self._obs) if ob is None or ob['is_last']}
      for i, ob in obs.items():
        self._obs[i] = ob() if callable(ob) else ob
        act = {k: np.zeros(v.shape) for k, v in self._act_spaces[i].items()}
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        self._use_policy_2[i] = False
        self._goal_time[i] = 0
        # subgoals search
        if get_goal:
          self._subgoals[i] = subgoal = get_goal(obs, self._state, **self._kwargs)
          tran[self.goal_key] = subgoal.numpy()

        [fn(tran, worker=i, **self._kwargs) for fn in self._on_resets]
        if goal_checker is not None:
          # _, goal_info = goal_checker(obs) # update goal distance metric
          self._goal_dist[i] = 0
          self._goal_success[i] = 0.0
        self._eps[i] = [tran]

      obs = {}
      for k in self._obs[0]:
        if k == self.goal_key: # use subgoal if generated else use original goal.
          goals = [g if g is not None else self._obs[i][k] for (i,g) in enumerate(self._subgoals)]
          obs[k] = np.stack(goals)
        else:
          obs[k] = np.stack([o[k] for o in self._obs])
      # TODO: hack, we know there is only 1 env and policy.
      policy = policy_2 if self._use_policy_2[0] else policy_1
      actions, self._state = policy(obs, self._state, **self._kwargs)

      actions = [
          {k: np.array(actions[k][i]) for k in actions}
          for i in range(len(self._envs))]
      assert len(actions) == len(self._envs)
      obs = [e.step(a) for e, a in zip(self._envs, actions)]
      obs = [ob() if callable(ob) else ob for ob in obs]
      if get_goal: # overwrite goal since obs just came from env.
        for o in obs:
          o[self.goal_key] = subgoal.numpy()
      # now check if obs achieved subgoal or not.
      for i, ob in enumerate(obs):
        if policy_2 is None or self._use_policy_2[i]:
          continue
        self._goal_time[i] += 1
        subgoal = self._subgoals[i]
        out_of_time = goal_time_limit and self._goal_time[i] > goal_time_limit
        close_to_goal, goal_info = goal_checker(obs)
        self._goal_dist[i] += goal_info["subgoal_dist"]
        self._goal_success[i] += goal_info["subgoal_success"]
        if out_of_time or close_to_goal:
          self._use_policy_2[i] = True

      for i, (act, ob) in enumerate(zip(actions, obs)):
        tran = {k: self._convert(v) for k, v in {**ob, **act}.items()}
        [fn(tran, worker=i, **self._kwargs) for fn in self._on_steps]
        self._eps[i].append(tran)
        step += 1
        if ob['is_last']:
          ep = self._eps[i]
          ep = {k: self._convert([t[k] for t in ep]) for k in ep[0]}
          ep["log_subgoal_dist"] = np.array([self._goal_dist[i]]) # add subgoal metrics
          ep["log_subgoal_success"] = np.array([float(self._goal_success[i] > 0)]) # add subgoal metrics
          ep["log_subgoal_time"] = np.array([self._goal_time[i]]) # time to reach subgoal.
          [fn(ep, **self._kwargs) for fn in self._on_episodes]
          episode += 1
      self._obs = obs