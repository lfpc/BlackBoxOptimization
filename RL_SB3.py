import math
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
import numpy as np
import gymnasium as gym
import d3rlpy
from d3rlpy.models import IQNQFunctionFactory
from d3rlpy.metrics import EnvironmentEvaluator
import shutil
import random
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

if torch.cuda.is_available(): 
    dev=torch.device(f'cuda:0')
    torch.cuda.set_device(dev)

def f(phi,noise=True):#Minimum is for x=y=5.4829 and f=-210.4823/200
    x,y=phi.squeeze().unbind()
    sum_x=0
    for n in range(1,6):
        sum_x+=n*math.cos(n+x*(n+1))
    sum_y=0
    for n in range(1,6):
        sum_y+=n*math.cos(n+y*(n+1))
    if noise:
        return 100*(-sum_x*sum_y/200)#+np.random.normal(0, 0.03))#TO_DO: Add a noise gaussian with amplitude 0.05
    else:
        return 100*(-sum_x*sum_y/200)
    
class RL_muons_env(gym.Env):
    def __init__(self, problem_fn, phi_bounds, initial_phi, max_steps, tolerance, step_scale):
        super().__init__()
        self.problem_fn=problem_fn
        self.phi_bounds=phi_bounds
        self.initial_phi=initial_phi
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.step_scale=step_scale#0.05 would be a suitable value

        self.steps = 0
        self.x = None
        self.prev_f = None
        self.best_x = None
        self.best_f = None

        self.dim=len(self.phi_bounds.T)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.dim,), dtype=np.float32
        )
        low_bounds, high_bounds = self.phi_bounds.numpy()
        self.low_bounds = low_bounds.astype(np.float32)
        self.high_bounds = high_bounds.astype(np.float32)
        obs_low = np.concatenate([self.low_bounds, np.array([-np.inf], dtype=np.float32)])
        obs_high = np.concatenate([self.high_bounds, np.array([np.inf], dtype=np.float32)])
        self.observation_space = gym.spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )
        self.step_scale=[self.step_scale*(high.item()-low.item()) for low,high in self.phi_bounds.T]

        self.reset()#TO_DO: Check if I need to reset here

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.x=self.initial_phi.copy()

        phi=torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        y = self.problem_fn(phi)

        self.prev_f = y
        self.best_x = self.x.copy()
        self.best_f = self.prev_f
        self.steps = 0
        obs = np.concatenate([self.x, np.array([self.prev_f], dtype=np.float32)])
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).squeeze()
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.step_scale
        self.x = np.clip(self.x + delta, self.low_bounds, self.high_bounds)

        phi=torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        y = self.problem_fn(phi)
        f_val=y
        reward = self.prev_f - f_val  #Improvement reward: Note that since we are minimizing the loss, the reward is the previous loss minus the current loss
        self.prev_f = f_val
        self.steps += 1

        if f_val < self.best_f:# track best
            self.best_f = f_val
            self.best_x = self.x.copy()

        done = bool(f_val <= self.tolerance)
        truncated = bool(self.steps >= self.max_steps)
        obs = np.concatenate([self.x, np.array([f_val], dtype=np.float32)])
        info = {"best_f": self.best_f, "best_x": self.best_x.copy()}
        if done or truncated:
            print(f"Done: {done}, truncated: {truncated}")
            self.render()
            self.last_x_before_reset = self.x.copy()
            self.last_f_before_reset = f_val
        return obs, float(reward), done, truncated, info

    def render(self, mode="human"):
        print(f"step={self.steps}, x={self.x}, f(x)={self.prev_f:.4f}, best_x={self.best_x}, best_f={self.best_f:.4f}")

    def seed(self, seed=None):
        np.random.seed(seed)

class TrainingStatsCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env #eval_env should be a plain (non-vec) env instance
        self.eval_freq = eval_freq
        self.last_eval_step = 0
        self.episode_final_f = []
        self.best_f = float('inf')
        self.best_x = None
        self.eval_scores = []

        self.total_steps=0
        self.best_f_history=[]
        self.total_steps_history=[]

    def _on_step(self) -> bool:
        self.total_steps+=1
        #Detect episode end from Monitor wrapper:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:#If True it means this is the end of an episode
                final_f=info["terminal_observation"][-1]
                final_x=info["terminal_observation"][:-1].copy()
                self.episode_final_f.append(final_f)
                #Check if this is the best episode:
                if final_f < self.best_f:
                    self.best_f = final_f
                    self.best_x = final_x
                    self.best_f_history.append(self.best_f)
                    self.total_steps_history.append(self.total_steps)
        #Periodic deterministic evaluation:
        if self.num_timesteps - self.last_eval_step >= self.eval_freq:
            obs, _ = self.eval_env.reset()
            done, truncated = False, False
            while not (done or truncated):
                action, _ = self.model.predict(obs.reshape(1, -1), deterministic=True)
                obs, reward, done, truncated, info = self.eval_env.step(action)
            final_f = float(obs[-1])
            self.eval_scores.append(final_f)
            self.last_eval_step = self.num_timesteps
            if self.verbose:
                print(f"[Eval] step={self.num_timesteps} final_f={final_f:.4f}")
        return True

class RL():
    def __init__(self,problem_fn,phi_bounds,initial_phi,max_steps,tolerance,step_scale,training_steps,device,devices):
        self.problem_fn=problem_fn
        self.phi_bounds=phi_bounds
        self.initial_phi=initial_phi
        self.max_steps=max_steps
        self.tolerance=tolerance
        self.step_scale=step_scale
        self.training_steps=training_steps
        self.device=device
        self.devices=devices

    def run_optimization(self):        
        eval_env = RL_muons_env(self.problem_fn, self.phi_bounds, self.initial_phi, self.max_steps, self.tolerance, self.step_scale)

        # Wrap with Monitor and DummyVecEnv for SB3:
        def make_train_env():
            return Monitor(RL_muons_env(self.problem_fn, self.phi_bounds, self.initial_phi, self.max_steps, self.tolerance, self.step_scale))
        vec_env = DummyVecEnv([make_train_env])  # single-env vectorization

        # Evaluation callback: evaluate every eval_freq timesteps
        eval_freq = max(1, int(0.05 * self.training_steps))  # similar to your det_eval interval
        callback = TrainingStatsCallback(eval_env=eval_env, eval_freq=eval_freq, verbose=1)

        # Policy kwargs: small MLP similar to what you want
        policy_kwargs = dict(
            net_arch=[dict(pi=[256, 256], vf=[256, 256])],
            activation_fn=torch.nn.ReLU
        )

        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=3e-4,
            n_steps=2048,           
            batch_size=64,
            n_epochs=10,
            gamma=0.9,#Need to use small values of gamma like 0.9, as 0.99 leads to unsuccessful training!
            verbose=1,
            tensorboard_log="outputs/PPO_tensorboard",
            policy_kwargs=policy_kwargs,
            device=dev,
        )

        # Train
        model.learn(total_timesteps=self.training_steps, callback=callback)

        # 5) Save / load
        model.save(f"outputs/RL_tests_5/ppo_model")

        print(f"Best evaluation achieved during training: x={callback.best_x}, f(x)={callback.best_f:.4f}")

        #Plots:
        fig=plt.figure()
        plt.plot(callback.episode_final_f,marker='o')
        plt.xlabel("Episode")
        plt.ylabel("Final f")
        plt.title("Final f per training episode")
        plt.grid(True)
        plt.savefig(f"outputs/RL_tests_5/training_final_f.png")
        plt.close(fig)

        fig=plt.figure()
        plt.plot(100*np.linspace(0, 1, len(callback.eval_scores)),callback.eval_scores,marker='o')
        plt.xlabel("Training %")
        plt.ylabel("Deterministic evaluation final f")
        plt.title("Periodic deterministic evaluations")
        plt.grid(True)
        plt.savefig(f"outputs/RL_tests_5/deterministic_final_f.png")
        plt.close(fig)

        fig=plt.figure()
        plt.plot(callback.total_steps_history,callback.best_f_history,marker='o')
        plt.xlabel("Training steps")
        plt.ylabel("Best training f")
        plt.title("Best training evaluation")
        plt.grid(True)
        plt.savefig(f"outputs/RL_tests_5/best_training_f.png")
        plt.close(fig)

        return callback.best_x,callback.best_f

def plot_frame(f, x_range, y_range, obs, historic_best_obs, filename, step):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    # mesh grid
    X = np.linspace(x_range[0], x_range[1], 100)
    Y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    # compute f for the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            phi = torch.tensor([[X[i,j], Y[i,j]]], dtype=torch.float32)
            Z[i,j] = f(phi,noise=False)
    # surface
    ax.plot_surface(X, Y, -Z, cmap='viridis', alpha=0.7)
    # blue dot for best historic observation during training
    x_obs, y_obs = historic_best_obs[:-1]
    z_obs = historic_best_obs[-1]
    ax.scatter(x_obs, y_obs, -z_obs, color='blue', s=50, label='Best evaluation during training')
    # red dot for current observation
    x_obs, y_obs = obs[:-1]
    z_obs = obs[-1]
    ax.scatter(x_obs, y_obs, -z_obs, color='red', s=50, label='Current observation')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)") 
    ax.set_title(f"Step {step}: f(x,y)={-z_obs:.3f}", fontsize=14)
    plt.suptitle(f"Best training score: f={-historic_best_obs[-1]:.3f}. True maximum: f={100*210.482/200:.3f}", y=0.95, fontsize=14)
    ax.legend()   
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:0')]
    phi_bounds=torch.tensor(((-2,-2),(2,2)))
    initial_phi=np.array([0,0], dtype=np.float32)
    RL_dict={}
    RL_dict["max_steps"]=50#50#TO_DO: Identify a proper value
    RL_dict["tolerance"]=-100*210/200#TO_DO: Identify a proper value
    RL_dict["step_scale"]=0.05
    RL_dict["training_steps"]=200000
    historic_best_x,historic_best_f=RL(problem_fn=f,
        phi_bounds=phi_bounds,
        initial_phi=initial_phi,
        max_steps=RL_dict["max_steps"],
        tolerance=RL_dict["tolerance"],
        step_scale=RL_dict["step_scale"],
        training_steps=RL_dict["training_steps"],
        device=dev,
        devices=devices).run_optimization()
    #Play an episode with trained agent:
    frames_dir = "outputs/RL_tests_5/frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)#Delete old frames
    os.makedirs(frames_dir, exist_ok=True)
    frame_files = []
    model = PPO.load("outputs/RL_tests_5/ppo_model", device=dev)
    play_env = RL_muons_env(f,phi_bounds,initial_phi,RL_dict["max_steps"],RL_dict["tolerance"],RL_dict["step_scale"])
    obs, _ = play_env.reset()
    done = False
    truncated = False
    frame_idx = 0
    low_bounds, high_bounds = phi_bounds.numpy()
    x_bounds=(low_bounds[0],high_bounds[0])
    y_bounds=(low_bounds[1],high_bounds[1])
    historic_best_obs_frame=np.concatenate([historic_best_x,np.array([historic_best_f], dtype=np.float32)])
    while not (done or truncated):
        action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
        obs, reward, done, truncated, info = play_env.step(action)
        # Create a frame
        obs_frame = np.concatenate([obs[:-1], [obs[-1]]])
        filename = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
        plot_frame(f, x_bounds, y_bounds, obs_frame, historic_best_obs_frame, filename, frame_idx)
        frame_files.append(filename)
        frame_idx += 1
    # Show final results
    print("Episode played with trained agent: ")
    play_env.render() 
    # Build GIF
    gif_path = f"outputs/RL_tests_5/episode.gif"
    with imageio.get_writer(gif_path, mode='I', format='GIF', fps=2) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

