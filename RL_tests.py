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

def f(phi):#Minimum is for x=y=5.4829 and f=-210.4823/200
    x,y=phi.squeeze().unbind()
    sum_x=0
    for n in range(1,6):
        sum_x+=n*math.cos(n+x*(n+1))
    sum_y=0
    for n in range(1,6):
        sum_y+=n*math.cos(n+y*(n+1))
    return -sum_x*sum_y/200#TO_DO: Add a noise gaussian with amplitude 0.05
    
def get_freest_gpu():
    import subprocess
    if not torch.cuda.is_available():
        return torch.device('cpu')
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        stdout=subprocess.PIPE, encoding='utf-8'
    )
    # Parse free memory for each GPU
    mem_free = [int(x) for x in result.stdout.strip().split('\n')]
    max_idx = mem_free.index(max(mem_free))
    return torch.device(f'cuda:{max_idx}')
if torch.cuda.is_available(): 
    dev=torch.device(f'cuda:0')
    torch.cuda.set_device(dev)

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

        self.historic_best_x=None
        self.historic_best_f=1e16#Huge number
        self.training_last_f=[]

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
        #TO_DO: Think about proper initialization
        #self.x=np.random.uniform(
        #    self.low_bounds, self.high_bounds#, size=(self.dim,)
        #).astype(np.float32)
        self.x=self.initial_phi.copy()
        #self.x=[self.problem_fn.initial_phi.tolist()[param_index]+0.0001*np.random.normal(0, self.step_scale[param_index]) for param_index in range(len(self.phi_bounds.T))]

        #self.x = np.random.uniform(
        #    -self.init_scale, self.init_scale, size=(self.dim,)
        #).astype(np.float32)

        phi=torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        y = self.problem_fn(phi)

        self.prev_f = y
        self.best_x = self.x.copy()
        self.best_f = self.prev_f
        self.steps = 0

        obs = np.concatenate([self.x, np.array([self.prev_f], dtype=np.float32)])
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        delta = action * self.step_scale
        self.x = np.clip(self.x + delta, self.low_bounds, self.high_bounds)

        phi=torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        y = self.problem_fn(phi)
        f_val=y
        reward = self.prev_f - f_val  #Improvement reward: Note that since we are minimizing the loss, the reward is the previous loss minus the current loss
        self.prev_f = f_val
        self.steps += 1

        # track best
        if f_val < self.best_f:
            self.best_f = f_val
            self.best_x = self.x.copy()

        done = bool(f_val <= self.tolerance)
        truncated = bool(self.steps >= self.max_steps)

        obs = np.concatenate([self.x, np.array([f_val], dtype=np.float32)])
        info = {"best_f": self.best_f, "best_x": self.best_x.copy()}

        if done or truncated:
            print(f"Done: {done}, truncated: {truncated}")
            self.render()
            if self.best_f<self.historic_best_f:#TO_DO: At some point I might parallelize the playing of training episodes, for which this check should be done carefully to avoid race conditions
                self.historic_best_f=self.best_f
                self.historic_best_x=self.best_x.copy()
            self.training_last_f.append(f_val)

        return obs, float(reward), done, truncated, info

    def render(self, mode="human"):
        print(f"step={self.steps}, x={self.x}, f(x)={self.prev_f:.4f}, best_x={self.best_x}, best_f={self.best_f:.4f}")

    def seed(self, seed=None):
        np.random.seed(seed)

class DeterministicEvaluator():
    def __init__(self, eval_env, step_interval):
        self.eval_env = eval_env
        self.step_interval = step_interval
        self.last_eval_step = 0
        self.scores = []

    def __call__(self, algo, epoch, total_step):
        if total_step - self.last_eval_step >= self.step_interval:
            obs, _ = self.eval_env.reset()
            done, truncated = False, False
            while not (done or truncated):
                action = algo.predict(obs.reshape(1, -1))[0]
                obs, reward, done, truncated, info = self.eval_env.step(action)
            final_f = obs[-1]
            self.scores.append(final_f)
            self.last_eval_step = total_step

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
        env = RL_muons_env(self.problem_fn, self.phi_bounds, self.initial_phi, self.max_steps, self.tolerance, self.step_scale)
        eval_env = RL_muons_env(self.problem_fn, self.phi_bounds, self.initial_phi, self.max_steps, self.tolerance, self.step_scale)

        det_eval = DeterministicEvaluator(eval_env, step_interval=int(0.05 * self.training_steps))

        # 2) Build IQN Q-function factory
        iqn_q_function = IQNQFunctionFactory(
            n_quantiles=32,         # number of quantile samples for learning
            n_greedy_quantiles=32,  # quantiles used for greedy action evaluation
            embed_size=64           # size of quantile embedding
        )

        # 3) Create SAC with IQN critic
        sac = d3rlpy.algos.SACConfig(
            q_func_factory=iqn_q_function,   # <-- plug IQN in here
            batch_size=256,
            n_critics=2,
            gamma=0.99,
            tau=5e-3,
            #learning_rate=3e-4,
            initial_temperature=0.1
        ).create(device="cuda:0")            # or device="cpu"
        sac.build_with_env(env)

        # 4) Train online (interacts with env)
        sac.fit_online(
            env,
            eval_env=eval_env,
            n_steps=self.training_steps,#20000,          # your total interaction steps
            #logdir="logs/iqn_sac",
            #eval_interval=10_000      # evaluate every N steps
            callback=det_eval
        )


        # 5) Save / load
        #sac.save_model(f"outputs/RL_tests/iqn_sac_model.d3")
        sac.save(f"outputs/RL_tests/iqn_sac_model.d3")

        print(f"Best evaluation achieved during training: x={env.historic_best_x}, f(x)={env.historic_best_f:.4f}")

        #Plots:
        fig=plt.figure()
        plt.plot(env.training_last_f,marker='o')
        plt.xlabel("Episode")
        plt.ylabel("Final f")
        plt.title("Final f per training episode")
        plt.grid(True)
        plt.savefig(f"outputs/RL_tests/training_final_f.png")
        plt.close(fig)

        fig=plt.figure()
        plt.plot(100*np.linspace(0, 1, len(det_eval.scores)),det_eval.scores,marker='o')
        plt.xlabel("Training %")
        plt.ylabel("Deterministic evaluation final f")
        plt.title("Periodic deterministic evaluations")
        plt.grid(True)
        plt.savefig(f"outputs/RL_tests/deterministic_final_f.png")
        plt.close(fig)

        return env.historic_best_x,env.historic_best_f

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
            Z[i,j] = f(phi)
    # surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7)
    # blue dot for best historic observation during training
    x_obs, y_obs = historic_best_obs[:-1]
    z_obs = historic_best_obs[-1]
    ax.scatter(x_obs, y_obs, z_obs, color='blue', s=50, label='Best evaluation during training')
    # red dot for current observation
    x_obs, y_obs = obs[:-1]
    z_obs = obs[-1]
    ax.scatter(x_obs, y_obs, z_obs, color='red', s=50, label='Current observation')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)") 
    ax.set_title(f"Step {step}: f(x,y)={z_obs:.3f}", fontsize=14)
    plt.suptitle(f"Best training score: f={historic_best_obs[-1]:.3f}. True minimum: f={-210.482/200:.3f}", y=0.95, fontsize=14)
    ax.legend()   
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:0')]
    phi_bounds=torch.tensor(((-2,-2),(2,2)))
    initial_phi=np.array([0.5,0.5], dtype=np.float32)
    RL_dict={}
    RL_dict["max_steps"]=50#TO_DO: Identify a proper value
    RL_dict["tolerance"]=-210/200#TO_DO: Identify a proper value
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
    frames_dir = "outputs/RL_tests/frames"
    if os.path.exists(frames_dir):
        shutil.rmtree(frames_dir)#Delete old frames
    os.makedirs(frames_dir, exist_ok=True)
    frame_files = []
    sac=d3rlpy.load_learnable("outputs/RL_tests/iqn_sac_model.d3")
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
        action = sac.predict(np.array(obs, dtype=np.float32).reshape(1, -1))[0]#This computes the action deterministically
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
    gif_path = f"outputs/RL_tests/episode.gif"
    with imageio.get_writer(gif_path, mode='I', duration=1.0) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)

