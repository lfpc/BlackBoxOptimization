import random
from collections import deque
import os

import matplotlib.pyplot as plt
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential

from environment import MuonShieldDiscreteEnvironment
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs.utils import step_mdp
from torchrl.modules import QValueModule


class DeepQNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def linear_epsilon(step, eps_start, eps_end, decay_steps):
    if step >= decay_steps:
        return eps_end
    alpha = step / float(decay_steps)
    return eps_start + alpha * (eps_end - eps_start)


# Training config
FIGS_PATH = "figs"
latent_dim = 32
n_samples = 100_000_000
num_episodes = 4000
gamma = 1.0
lr = 3e-4
batch_size = 128
buffer_size = 100_000
warmup_steps = 2_000
target_update_tau = 0.01
eps_start = 1.0
eps_end = 0.01
eps_decay_steps = 20_000
eval_every_episodes = 50
eval_episodes = 3
seed = 42


random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

os.makedirs(FIGS_PATH, exist_ok=True)


base_env = MuonShieldDiscreteEnvironment(latent_dim=latent_dim, n_samples=n_samples)
env = GymWrapper(base_env)
n_actions = env.action_space.n

q_net = DeepQNetwork(input_dim=latent_dim, output_dim=n_actions)
target_net = DeepQNetwork(input_dim=latent_dim, output_dim=n_actions)
target_net.load_state_dict(q_net.state_dict())
optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)

value_net = TensorDictModule(q_net, in_keys=["observation"], out_keys=["action_value"])
policy = TensorDictSequential(value_net, QValueModule(spec=env.action_spec))

replay_buffer = deque(maxlen=buffer_size)
episode_returns = []
best_return = float("-inf")
best_magnet = None
best_actions = []
best_eval_return = float("-inf")
best_eval_actions = []
best_q_state_dict = None
total_steps = 0
reward_scale = 1.0 / float(n_samples)


def optimize_model():
    if len(replay_buffer) < max(batch_size, warmup_steps):
        return None

    batch = random.sample(replay_buffer, batch_size)
    states = torch.stack([item[0] for item in batch])
    actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
    rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32)
    next_states = torch.stack([item[3] for item in batch])
    dones = torch.tensor([item[4] for item in batch], dtype=torch.float32)

    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        # Double DQN target to reduce Q overestimation.
        next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
        next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
        target = rewards + gamma * (1.0 - dones) * next_q_values

    loss = torch.nn.functional.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=1.0)
    optimizer.step()
    return float(loss.item())


def soft_update_target_network(source_net, destination_net, tau):
    with torch.no_grad():
        for destination_param, source_param in zip(destination_net.parameters(), source_net.parameters()):
            destination_param.data.mul_(1.0 - tau).add_(tau * source_param.data)


def run_policy_once(epsilon=0.0):
    was_training = q_net.training
    q_net.eval()

    td_eval = env.reset()
    done_eval = False
    actions = []
    total_reward = 0.0

    with torch.no_grad():
        while not done_eval:
            td_eval_policy = policy(td_eval.clone())
            if random.random() < epsilon:
                random_action_eval = torch.randint(
                    low=0,
                    high=n_actions,
                    size=td_eval_policy.get("action").shape,
                    dtype=td_eval_policy.get("action").dtype,
                )
                td_eval_policy.set("action", random_action_eval)

            action_eval = int(td_eval_policy.get("action").reshape(-1)[0].item())
            actions.append(action_eval)

            td_eval_step = env.step(td_eval_policy)
            reward_eval = float(td_eval_step.get(("next", "reward")).reshape(-1)[0].item())
            total_reward += reward_eval

            td_eval = step_mdp(td_eval_step)
            done_eval = bool(td_eval.get("done").item())

    if was_training:
        q_net.train()

    return actions, total_reward


for episode in range(num_episodes):
    td = env.reset()
    done = False
    episode_return = 0.0
    episode_actions = []

    while not done:
        epsilon = linear_epsilon(total_steps, eps_start, eps_end, eps_decay_steps)

        td_policy = policy(td.clone())
        if random.random() < epsilon:
            random_action = torch.randint(
                low=0,
                high=n_actions,
                size=td_policy.get("action").shape,
                dtype=td_policy.get("action").dtype,
            )
            td_policy.set("action", random_action)

        action = int(td_policy.get("action").reshape(-1)[0].item())
        episode_actions.append(action)
        state = td.get("observation").detach().clone().reshape(-1).float()

        td_step = env.step(td_policy)

        reward_tensor = td_step.get(("next", "reward"))
        next_obs = td_step.get(("next", "observation"))
        done_tensor = td_step.get(("next", "done"))

        reward = float(reward_tensor.reshape(-1)[0].item())
        scaled_reward = reward * reward_scale
        done_flag = float(done_tensor.reshape(-1)[0].item())
        next_state = next_obs.detach().clone().reshape(-1).float()

        replay_buffer.append((state, action, scaled_reward, next_state, done_flag))

        optimize_model()
        soft_update_target_network(q_net, target_net, target_update_tau)
        total_steps += 1

        episode_return += reward
        td = step_mdp(td_step)
        done = bool(td.get("done").item())

    episode_returns.append(episode_return)
    final_muons = base_env.muons.shape[0]

    if episode_return > best_return:
        best_return = episode_return
        best_magnet = base_env.current_magnet.detach().cpu().clone()
        best_actions = episode_actions.copy()

    print(
        f"Episode {episode + 1}/{num_episodes} | "
        f"return={episode_return:.2f} | epsilon={epsilon:.3f} | final_muons={final_muons}"
    )

    if (episode + 1) % eval_every_episodes == 0 or (episode + 1) == num_episodes:
        eval_rewards = []
        eval_actions_snapshot = None
        for _ in range(eval_episodes):
            actions_eval, reward_eval = run_policy_once(epsilon=0.0)
            eval_rewards.append(reward_eval)
            if eval_actions_snapshot is None:
                eval_actions_snapshot = actions_eval

        eval_mean_return = sum(eval_rewards) / float(len(eval_rewards))

        if eval_mean_return > best_eval_return:
            best_eval_return = eval_mean_return
            best_eval_actions = eval_actions_snapshot.copy()
            best_q_state_dict = {
                key: value.detach().cpu().clone() for key, value in q_net.state_dict().items()
            }

        print(
            f"  Eval ({eval_episodes} greedy rollouts) | "
            f"mean_return={eval_mean_return:.2f} | best_eval_return={best_eval_return:.2f}"
        )


plot_path = os.path.join(FIGS_PATH,"reward_evolution.png")
plt.figure(figsize=(8, 4))
plt.plot(episode_returns)
plt.xlabel("Episode")
plt.ylabel("Episode Return")
plt.title("DQN Reward Evolution")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plot_path, dpi=180)
print("Figure saved to:", plot_path)

if best_q_state_dict is not None:
    q_net.load_state_dict(best_q_state_dict)
    target_net.load_state_dict(best_q_state_dict)


optimized_actions, optimized_reward = run_policy_once(epsilon=0.0)

print("\n=== Final Training Summary ===")
print(f"Best training configuration: actions={best_actions}, reward={best_return:.2f}")
print(f"Best evaluated policy during training: actions={best_eval_actions}, mean_reward={best_eval_return:.2f}")
print(f"Optimized policy rollout: actions={optimized_actions}, reward={optimized_reward:.2f}")


