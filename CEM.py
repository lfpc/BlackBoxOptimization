#Code below uses: cross-entropy-method library (pip install cross-entropy-method)

import math
import numpy as np
import matplotlib.pyplot as plt
import shutil
import torch
import imageio

# Your function
def f(phi, noise=True):
    x, y = float(phi[0]), float(phi[1])/10

    sum_x = sum(n * math.cos(n + x * (n + 1)) for n in range(1, 6))
    sum_y = sum(n * math.cos(n + y * (n + 1)) for n in range(1, 6))

    val = -sum_x * sum_y / 200

    if noise:
        return val + np.random.normal(0, 0.03)
    else:
        return val

# Expects a minimization problem. 
def objective(phi):
    return f(phi, noise=True) 

def plot_frame(f, x_range, y_range, obs, filename, step):
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    # mesh grid
    X = np.linspace(x_range[0], x_range[1], 30)#0)
    Y = np.linspace(y_range[0], y_range[1], 30)#0)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    # compute f for the grid
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            phi = torch.tensor([[X[i,j], Y[i,j]]], dtype=torch.float32).squeeze(0)
            Z[i,j] = f(phi,noise=False)
    # surface
    ax.plot_surface(X, Y, -Z, cmap='viridis', alpha=0.7)
    # blue dot for best historic observation during training
    #x_obs, y_obs = historic_best_obs[:-1]
    #z_obs = historic_best_obs[-1]
    #ax.scatter(x_obs, y_obs, -z_obs, color='blue', s=50, label='Best evaluation during training')
    # red dot for current observation
    x_obs, y_obs = obs[:-1]
    z_obs = obs[-1]
    ax.scatter(x_obs, y_obs, -z_obs, color='red', s=50, label='Current observation')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)") 
    ax.set_title(f"Step {step}: f(x,y)={-z_obs:.3f}", fontsize=14)
    plt.suptitle(f"True maximum: f={210.482/2:.3f}", y=0.95, fontsize=14)
    ax.legend()   
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

dim = 2
x0 = np.array([0.0, 0.0])

lower_bounds = np.array([-2, -20])
upper_bounds = np.array([2, 20])

population_size = 40
elite_frac = 0.2
n_elite = max(1, int(population_size * elite_frac))

# Initial distribution
mean = x0.copy()
std = (upper_bounds - lower_bounds) * 0.3   # similar scale to sigma_vector

generations = 25
best_fitness_hist = 1e9
best_solution_hist = None


###
import os
frames_dir = "frames_borrar"
if os.path.exists(frames_dir):
    shutil.rmtree(frames_dir)#Delete old frames
os.makedirs(frames_dir, exist_ok=True)
frame_files = []
frame_idx = 0
x_bounds=(lower_bounds[0],upper_bounds[0])
y_bounds=(lower_bounds[1],upper_bounds[1])
###


for generation in range(generations):

    # 1) Sample population
    solutions = np.random.randn(population_size, dim) * std + mean

    # Clip to bounds
    solutions = np.clip(solutions, lower_bounds, upper_bounds)

    # 2) Evaluate population
    fitnesses = np.array([objective(s) for s in solutions])
    print(fitnesses)

    # 3) Select elite set
    elite_idx = np.argsort(fitnesses)[:n_elite]
    elites = solutions[elite_idx]

    # 4) Update mean and std
    mean = elites.mean(axis=0)
    std = elites.std(axis=0) + 1e-9  # avoid division by zero

    # 5) Track historical best
    gen_best = fitnesses[elite_idx[0]]
    if gen_best < best_fitness_hist:
        best_fitness_hist = gen_best
        best_solution_hist = solutions[elite_idx[0]].copy()
        print(f"Best solution in generation {generation}: {best_fitness_hist}")

    print(f"Generation {generation}: best = {gen_best}, hist_best = {best_fitness_hist}")


    ###
    obs_frame = np.concatenate([solutions[elite_idx[0]].copy(), [gen_best]])
    filename = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
    plot_frame(f, x_bounds, y_bounds, obs_frame, filename, frame_idx)
    frame_files.append(filename)
    frame_idx += 1
    ###

print("\n==================== RESULTS ====================")
print("Best historical solution:", best_solution_hist)
print("Best historical fitness:", best_fitness_hist)


###
gif_path = f"frames_borrar/evolution.gif"
with imageio.get_writer(gif_path, mode='I', format='GIF', fps=2) as writer:
    for filename in frame_files:
        image = imageio.imread(filename)
        writer.append_data(image)
print("GIF generated")
###