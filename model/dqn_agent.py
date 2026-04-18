import gymnasium as gym
import numpy as np

class FraudEnv(gym.Env):
    """
    Custom RL Environment for Fraud Detection

    Agent decides:
    0 → Approve transaction
    1 → Flag transaction

    State = risk score from GNN
    """

    metadata = {"render_modes": []}

    def __init__(self, risk_scores, true_labels):
        super(FraudEnv, self).__init__()

        # Convert to numpy (safe)
        self.risk_scores = np.array(risk_scores, dtype=np.float32)
        self.true_labels = np.array(true_labels, dtype=np.int32)

        # Step pointer
        self.current_step = 0

        # Action space: 0 (approve), 1 (flag)
        self.action_space = gym.spaces.Discrete(2)

        # Observation space: single risk score
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

    # ─────────────────────────
    # RESET
    # ─────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        return np.array([self.risk_scores[self.current_step]], dtype=np.float32), {}

    # ─────────────────────────
    # STEP
    # ─────────────────────────
    def step(self, action):

        true_label = self.true_labels[self.current_step]
        risk_score = self.risk_scores[self.current_step]

        # 🎯 Reward design (VERY IMPORTANT)
        if action == 1 and true_label == 1:
            reward = 2.0   # ✅ Correct fraud detection
        elif action == 0 and true_label == 0:
            reward = 1.0   # ✅ Correct approval
        elif action == 1 and true_label == 0:
            reward = -2.0  # ❌ False positive
        else:
            reward = -4.0  # ❌ Missed fraud (worst case)

        # Move to next step
        self.current_step += 1

        done = self.current_step >= len(self.risk_scores) - 1

        next_score = self.risk_scores[
            min(self.current_step, len(self.risk_scores) - 1)
        ]

        next_state = np.array([next_score], dtype=np.float32)

        return next_state, reward, done, False, {}

    # ─────────────────────────
    # OPTIONAL: RENDER
    # ─────────────────────────
    def render(self):
        print(f"Step: {self.current_step} | Score: {self.risk_scores[self.current_step]}")

    # ─────────────────────────
    # OPTIONAL: CLOSE
    # ─────────────────────────
    def close(self):
        pass


# ─────────────────────────
# TEST ENV (optional)
# ─────────────────────────
if __name__ == "__main__":

    print("🔧 Testing FraudEnv...")

    dummy_scores = np.random.rand(100).astype(np.float32)
    dummy_labels = np.random.randint(0, 2, 100)

    env = FraudEnv(dummy_scores, dummy_labels)

    state, _ = env.reset()
    print("Initial state:", state)

    for _ in range(5):
        action = env.action_space.sample()
        next_state, reward, done, _, _ = env.step(action)

        print(f"Action: {action} | Reward: {reward}")

        if done:
            break

    print("✅ Environment test completed!")