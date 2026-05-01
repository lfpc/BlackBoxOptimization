import random
from collections import deque
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Normal

from environment import MuonShieldContinuousEnvironment


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianPolicy(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mean = torch.nn.Linear(hidden_dim, action_dim)
        self.log_std = torch.nn.Linear(hidden_dim, action_dim)

        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", torch.as_tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor(action_bias, dtype=torch.float32))

    def forward(self, state):
        hidden = self.net(state)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action_tanh = torch.tanh(z)

        action = action_tanh * self.action_scale + self.action_bias

        # Correct log-probability for tanh squashing and affine action rescaling.
        log_prob = normal.log_prob(z) - torch.log(self.action_scale * (1.0 - action_tanh.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action


class CriticNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(state_dim + action_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat((state, action), dim=-1))


def soft_update_target_network(source_net, destination_net, tau):
    with torch.no_grad():
        for destination_param, source_param in zip(destination_net.parameters(), source_net.parameters()):
            destination_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


# Training config
FIGS_PATH = "figs"
latent_dim = 32
n_samples = 1_000_000
num_episodes = 4000
gamma = 1.0
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 3e-4
batch_size = 128
buffer_size = 100_000
warmup_steps = 2_000
random_steps = 2_000
target_update_tau = 0.01
hidden_dim = 256
eval_every_episodes = 50
eval_episodes = 3
seed = 42


random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(FIGS_PATH, exist_ok=True)


base_env = MuonShieldContinuousEnvironment(latent_dim=latent_dim, n_samples=n_samples)
state_dim = base_env.observation_space.shape[0]
action_dim = base_env.action_space.shape[0]
action_low = base_env.action_space.low
action_high = base_env.action_space.high

actor = GaussianPolicy(
    state_dim=state_dim,
    action_dim=action_dim,
    action_low=action_low,
    action_high=action_high,
    hidden_dim=hidden_dim,
).to(device)
critic_1 = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
critic_2 = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
target_critic_1 = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
target_critic_2 = CriticNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(device)
target_critic_1.load_state_dict(critic_1.state_dict())
target_critic_2.load_state_dict(critic_2.state_dict())

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = torch.optim.Adam(
    list(critic_1.parameters()) + list(critic_2.parameters()),
    lr=critic_lr,
)

target_entropy = -float(action_dim)
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optimizer = torch.optim.Adam([log_alpha], lr=alpha_lr)


def alpha_value():
    return log_alpha.exp()

replay_buffer = deque(maxlen=buffer_size)
episode_returns = []
best_return = float("-inf")
best_magnet = None
best_actions = []
best_eval_return = float("-inf")
best_eval_actions = []
best_actor_state_dict = None
total_steps = 0
reward_scale = 1.0 / float(n_samples)


def optimize_model():
    if len(replay_buffer) < max(batch_size, warmup_steps):
        return None

    batch = random.sample(replay_buffer, batch_size)
    states = torch.stack([item[0] for item in batch]).to(device)
    actions = torch.stack([item[1] for item in batch]).to(device)
    rewards = torch.as_tensor([item[2] for item in batch], dtype=torch.float32, device=device).unsqueeze(1)
    next_states = torch.stack([item[3] for item in batch]).to(device)
    dones = torch.as_tensor([item[4] for item in batch], dtype=torch.float32, device=device).unsqueeze(1)

    with torch.no_grad():
        next_actions, next_log_prob, _ = actor.sample(next_states)
        next_q_1 = target_critic_1(next_states, next_actions)
        next_q_2 = target_critic_2(next_states, next_actions)
        next_q = torch.min(next_q_1, next_q_2) - alpha_value() * next_log_prob
        q_target = rewards + gamma * (1.0 - dones) * next_q

    q_1 = critic_1(states, actions)
    q_2 = critic_2(states, actions)

    critic_loss_1 = torch.nn.functional.mse_loss(q_1, q_target)
    critic_loss_2 = torch.nn.functional.mse_loss(q_2, q_target)
    critic_loss = critic_loss_1 + critic_loss_2

    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(list(critic_1.parameters()) + list(critic_2.parameters()), max_norm=1.0)
    critic_optimizer.step()

    sampled_actions, log_prob, _ = actor.sample(states)
    q_pi = torch.min(critic_1(states, sampled_actions), critic_2(states, sampled_actions))
    actor_loss = (alpha_value().detach() * log_prob - q_pi).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
    actor_optimizer.step()

    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
    alpha_optimizer.zero_grad()
    alpha_loss.backward()
    alpha_optimizer.step()

    soft_update_target_network(critic_1, target_critic_1, target_update_tau)
    soft_update_target_network(critic_2, target_critic_2, target_update_tau)

    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "alpha": float(alpha_value().item()),
    }


def select_action(observation, deterministic=False):
    state = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if deterministic:
            _, _, action = actor.sample(state)
        else:
            action, _, _ = actor.sample(state)
    return action.squeeze(0).cpu().numpy().astype(np.float32)


def run_policy_once(deterministic=True):
    was_training = actor.training
    actor.eval()

    obs, _ = base_env.reset()
    done = False
    actions = []
    total_reward = 0.0

    while not done:
        action = select_action(obs, deterministic=deterministic)
        actions.append(action.tolist())

        next_obs, reward, terminated, truncated, _ = base_env.step(action)
        total_reward += float(reward)
        done = bool(terminated or truncated)
        obs = next_obs

    if was_training:
        actor.train()

    return actions, total_reward


for episode in range(num_episodes):
    obs, _ = base_env.reset()
    done = False
    episode_return = 0.0
    episode_actions = []
    last_opt_metrics = None

    while not done:
        if total_steps < random_steps:
            action = base_env.action_space.sample().astype(np.float32)
        else:
            action = select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = base_env.step(action)
        done_flag = float(terminated or truncated)
        done = bool(terminated or truncated)

        scaled_reward = reward * reward_scale

        replay_buffer.append(
            (
                torch.as_tensor(obs, dtype=torch.float32),
                torch.as_tensor(action, dtype=torch.float32),
                float(scaled_reward),
                torch.as_tensor(next_obs, dtype=torch.float32),
                done_flag,
            )
        )

        last_opt_metrics = optimize_model()
        total_steps += 1

        episode_actions.append(action.tolist())
        episode_return += float(reward)
        obs = next_obs

    episode_returns.append(episode_return)
    final_muons = base_env.muons.shape[0]

    if episode_return > best_return:
        best_return = episode_return
        best_magnet = base_env.current_magnet.detach().cpu().clone()
        best_actions = episode_actions.copy()

    if last_opt_metrics is None:
        opt_info = "optimizer=warmup"
    else:
        opt_info = (
            f"actor_loss={last_opt_metrics['actor_loss']:.4f} | "
            f"critic_loss={last_opt_metrics['critic_loss']:.4f} | "
            f"alpha={last_opt_metrics['alpha']:.4f}"
        )

    print(
        f"Episode {episode + 1}/{num_episodes} | "
        f"return={episode_return:.2f} | final_muons={final_muons} | {opt_info}"
    )

    if (episode + 1) % eval_every_episodes == 0 or (episode + 1) == num_episodes:
        eval_rewards = []
        eval_actions_snapshot = None
        for _ in range(eval_episodes):
            actions_eval, reward_eval = run_policy_once(deterministic=True)
            eval_rewards.append(reward_eval)
            if eval_actions_snapshot is None:
                eval_actions_snapshot = actions_eval

        eval_mean_return = sum(eval_rewards) / float(len(eval_rewards))

        if eval_mean_return > best_eval_return:
            best_eval_return = eval_mean_return
            best_eval_actions = eval_actions_snapshot.copy()
            best_actor_state_dict = {
                key: value.detach().cpu().clone() for key, value in actor.state_dict().items()
            }

        print(
            f"  Eval ({eval_episodes} greedy rollouts) | "
            f"mean_return={eval_mean_return:.2f} | best_eval_return={best_eval_return:.2f}"
        )


plot_path = os.path.join(FIGS_PATH, "reward_evolution.png")
plt.figure(figsize=(8, 4))
plt.plot(episode_returns)
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("SAC Reward Evolution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_path, dpi=180)
print("Figure saved to:", plot_path)

if best_actor_state_dict is not None:
    actor.load_state_dict(best_actor_state_dict)


optimized_actions, optimized_reward = run_policy_once(deterministic=True)

print("\n=== Final Training Summary ===")
print(f"Best training configuration: actions={best_actions}, reward={best_return:.2f}")
print(f"Best evaluated policy during training: actions={best_eval_actions}, mean_reward={best_eval_return:.2f}")
print(f"Optimized policy rollout: actions={optimized_actions}, reward={optimized_reward:.2f}")


