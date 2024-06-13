from gymnasium.envs.registration import register

register(
    id="Blackjack/Blackjack-v0",
    entry_point="Blackjack.envs:BlackjackEnv",
    max_episode_steps=300,
)
