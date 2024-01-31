from gymnasium.envs.registration import register
register(
    id='pushBox-v0',
    entry_point='push_box.envs:PushBoxEnv',)