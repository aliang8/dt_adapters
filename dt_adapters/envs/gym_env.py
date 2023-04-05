import gym


class GymEnv(object):
    """
    Lightweight wrapper for gym environments
    """

    def __init__(self, env, env_kwargs=None, action_repeat=1, *args, **kwargs):
        # get the correct env behavior
        if type(env) == str:
            env = gym.make(env)
        elif isinstance(env, gym.Env):
            env = env
        elif callable(env):
            env = env(**env_kwargs)
        else:
            print("Unsupported environment format")
            raise AttributeError

        self.env = env
        self.action_repeat = action_repeat

        try:
            self._action_dim = self.env.action_space.shape[0]
        except AttributeError:
            self._action_dim = self.env.unwrapped.action_dim

        try:
            self._observation_dim = self.env.observation_space.shape[0]
        except AttributeError:
            self._observation_dim = self.env.unwrapped.obs_dim

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def reset(self, seed=None):
        try:
            self.env._elapsed_steps = 0
            return self.env.unwrapped.reset_model(seed=seed)
        except:
            if seed is not None:
                self.set_seed(seed)
            return self.env.reset()

    def step(self, action):
        action = action.clip(self.action_space.low, self.action_space.high)
        if self.action_repeat == 1:
            obs, cum_reward, done, ifo = self.env.step(action)
        else:
            cum_reward = 0.0
            for i in range(self.action_repeat):
                obs, reward, done, ifo = self.env.step(action)
                cum_reward += reward
                if done:
                    break
        return obs, cum_reward, done, ifo

    def render(self, image_dim=None, camera_name=None):
        try:
            self.env.unwrapped.mujoco_render_frames = True
            self.env.unwrapped.mj_render()
        except:
            return self.env.sim.render(
                height=image_dim, width=image_dim, camera_name=camera_name
            )
