import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

class FraudEnv(gym.Env):
    """Custom RL environment — agent decides to FLAG or APPROVE each node"""
    metadata = {"render_modes": []}

    def __init__(self, risk_scores, true_labels):
        super().__init__()
        self.risk_scores = risk_scores.astype(np.float32)
        self.true_labels = true_labels
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(2)  # 0=Approve, 1=Flag
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.array([self.risk_scores[0]], dtype=np.float32), {}

    def step(self, action):
        true = self.true_labels[self.current_step]
        score = self.risk_scores[self.current_step]

        # Cost-sensitive reward: missing fraud is worst
        if   action == 1 and true == 1: reward =  2.0  # ✅ Caught fraud
        elif action == 0 and true == 0: reward =  1.0  # ✅ Correct approval
        elif action == 1 and true == 0: reward = -2.0  # ⚠️ False alarm
        else:                            reward = -4.0  # ❌ Missed fraud!

        self.current_step += 1
        done = self.current_step >= len(self.risk_scores) - 1
        next_score = self.risk_scores[
            min(self.current_step, len(self.risk_scores) - 1)
        ]
        return (np.array([next_score], dtype=np.float32),
                reward, done, False, {})

if __name__ == "__main__":
    # Quick test
    dummy_scores = np.random.rand(100).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, 100)
    env = FraudEnv(dummy_scores, dummy_labels)
    obs, _ = env.reset()
    print(f"✅ FraudEnv ready | Obs shape: {obs.shape}")
    obs, reward, done, _, _ = env.step(1)
    print(f"✅ Step test passed | Reward: {reward}")
