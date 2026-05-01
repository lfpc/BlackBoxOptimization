import torch
from encoders import DeepSetEncoder, ConvolutionalEncoder
from cuda_muons_ship import run_from_params as simulate_muon_shield
import h5py
import numpy as np
import gymnasium as gym


HA_magnet = torch.tensor([0.0,115.50,50.00, 50.00, 119.00, 119.00, 2.00, 2.00, 50.00, 50.00, 50.00, 50.00, 0.00, 0.00, 1.9])
default_z_space = 10
class MuonShieldDiscreteEnvironment(gym.Env):
    ''' In this environment, the action is discrete and corresponds to 3 possible configurations of the magnet:
    - Action 0: No magnet
    - Action 1: Magnet with upper polarity (By in the core = 1.9)
    - Action 2: Magnet with down polarity (By in the yoke = 1.9)
    The magnet shape is fixed and corresponds to the HA magnet. 
    '''
    def __init__(self, latent_dim = 32,n_samples = 500_000_000):
        self.kwargs = {'NI_from_B': True,
                       'simulate_fields': False,
                       "add_cavern": True,
                       "use_diluted": False,
                       "return_all": False,
                       "histogram_dir": "/disk/users/lprate/projects/MuonsAndMatter/cuda_muons/data",
                       }
        self.detector = [{"dz": 0.02,"dx": 4,"dy": 6,"position": 82}, \
                        {"dz": 0.02,"dx": 4,"dy": 6,"position": 91}]

        self.default_magnet = HA_magnet
        self.muons_file =  "/home/hep/lprate/projects/MuonsAndMatter/data/muons/full_sample_after_target.h5"
        self.max_steps = 30
        self.max_Z = 30
        self.n_samples = n_samples
        self.encoder = DeepSetEncoder(input_dim=7, hidden_dim=32, output_dim=latent_dim)
        self.obs_dim = latent_dim
        #self.encoder = ConvolutionalEncoder(grid_size=(32,32), x_dim=7, encoder_hidden_dim=32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
    def sample_muons(self, seed = None):
        print('Sampling muons from file')
        idx = slice(None)

        n = self.n_samples
        start, stop, step = idx.indices(n)
        n = (stop - start + (step - 1)) // step
        read_slice = slice(start, stop, step) 
        
        x = np.empty((n, 7), dtype=np.float32)
        with h5py.File(self.muons_file, "r") as f:
            for j, feat in enumerate(["px", "py", "pz", "x", "y", "z", "pdg"]):
                col = np.empty(n, dtype=np.float32)  
                f[feat].read_direct(col, read_slice)
                x[:, j] = col     
                
        x = torch.from_numpy(x)
        return x
    def get_new_sens_plane(self, phi):
        dist_z = phi[:,:2].sum() + self.z_space + 2 #2cm between magnet and z plane
        dist_z /= 100
        return dist_z.item()
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        self.muons = self.sample_muons(seed = seed)
        self.current_magnet = torch.empty(
            (0, self.default_magnet.numel()), dtype=self.default_magnet.dtype
        )
        self.z_space = 0
        info = {}
        obs = self.encoder(self.muons.unsqueeze(0)).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return obs, info
    def get_phi_from_action(self, action):
        phi = self.default_magnet.clone()
        if action == 0:
            self.z_space += self.default_magnet[1]
        else:   
            phi[-1] = phi[-1]*action #B_goal 1.9 or -1.9
            if self.z_space == 0 and self.current_magnet.size(0) > 0:
                self.z_space = default_z_space
            phi[0] = self.z_space
            self.z_space = 0
        return phi.view(1,-1)
    def step(self, action):
        phi = self.get_phi_from_action(action)
        self.current_magnet = torch.cat((self.current_magnet, phi), dim=0)
        Z = self.get_new_sens_plane(self.current_magnet)
        if self.current_magnet.size(0) >= self.max_steps+1 or Z >= self.max_Z:
            sens_plane = self.detector
            terminated = True
        else:
            terminated = False
            sens_plane = {'dz': 0.02, 'dx': 20, 'dy': 20, 'position': Z}
        muons_output = simulate_muon_shield(self.current_magnet, self.muons, 
                                            sensitive_plane = sens_plane, 
                                            **self.kwargs)
        self.muons = torch.stack((muons_output['px'], muons_output['py'], muons_output['pz'], 
                               muons_output['x'], muons_output['y'], muons_output['z'], 
                               muons_output['pdg_id']),dim=1)
        if terminated:
            reward = float((-1)*self.muons.shape[0])
        else:
            reward = 0

        obs = self.encoder(self.muons.unsqueeze(0)).squeeze(0).detach().cpu().numpy().astype(np.float32)
        
        info = {}
        truncated = False
        return obs, reward, terminated, truncated, info

class MuonShieldContinuousEnvironment(MuonShieldDiscreteEnvironment):
    '''In this environment, the action is continuous and corresponds to three magnet parameters:
    - First dimension: between 5 and 500
    - Second dimension: between 5 and 300
    - Third dimension (B_goal): between -1.9 and 1.9

    The sequence of magnets is fixed to 7 magnets separated by 10 cm.
    All the magnets have regular shape, meaning yoke with same width as the core and no V shapes.
    '''
    def __init__(self, latent_dim = 32,n_samples = 500_000_000):
        super().__init__(latent_dim, n_samples)
        self.action_space = gym.spaces.Box(
            low=np.array([5.0, 5.0, -1.9], dtype=np.float32),
            high=np.array([500.0, 300.0, 1.9], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
    def get_phi_from_action(self, action):
        phi = self.default_magnet.clone()
        params = np.asarray(action, dtype=np.float32).reshape(-1)
        phi[1] = float(params[0])
        phi[2] = phi[3] = phi[8] = phi[9] = phi[10] = phi[11] = float(params[1])
        phi[-1] = float(params[2])
        return phi.view(1,-1)

class MuonShieldMixedEnvironment(MuonShieldDiscreteEnvironment):
    '''Hybrid action space:
    - place (Discrete): 0 = no magnet, 1 = place magnet.
    - params (Box[3]): [length, width, B_goal].
    - z_space (Box[1]): spacing parameter used when place == 0.

    Continuous params are only applied when place == 1.
    '''
    def __init__(self, latent_dim = 32,n_samples = 500_000_000):
        super().__init__(latent_dim, n_samples)
        self.action_space = gym.spaces.Dict({
            'place': gym.spaces.Discrete(2),
            'params': gym.spaces.Box(
                low=np.array([5.0, 5.0, -1.9], dtype=np.float32),
                high=np.array([500.0, 300.0, 1.9], dtype=np.float32),
                shape=(3,),
                dtype=np.float32,
            ),
            'z_space': gym.spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([500.0], dtype=np.float32),
                shape=(1,),
                dtype=np.float32,
            )
        })
    def get_phi_from_action(self, action):
        phi = self.default_magnet.clone()
        place = int(np.asarray(action['place']).item())

        if place == 0:
            z_space = float(np.asarray(action['z_space'], dtype=np.float32).reshape(-1)[0])
            phi[0] = z_space
            phi[1:] = 0.0
        else:
            params = np.asarray(action['params'], dtype=np.float32)
            phi[1] = float(params[0])
            phi[2] = phi[3] = phi[8] = phi[9] = phi[10] = phi[11] = float(params[1])
            phi[-1] = float(params[2])

        return phi.view(1,-1)