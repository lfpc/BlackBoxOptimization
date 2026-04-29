import torch

class DeepSetEncoder(torch.nn.Module):
    def __init__(self, input_dim = 7, hidden_dim = 64, output_dim = 32, pool = 'sum'):
        super(DeepSetEncoder, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )
        if pool == 'sum':
            self.pool = torch.sum
        elif pool == 'mean':
            self.pool = torch.mean
        elif pool == 'max':
            self.pool = torch.max
        else:
            raise ValueError("Unsupported pooling method: {}".format(pool))
    def forward(self, x):
        # x is of shape (batch_size, num_particles, input_dim)
        phi_x = self.mlp(x)  # shape (batch_size, num_particles, output_dim)
        obs = self.pool(phi_x, dim=1)  # shape (batch_size, output_dim)

        return obs

class ConvolutionalEncoder(torch.nn.Module):
    def __init__(self, grid_size = (32,32), x_dim = 7, encoder_hidden_dim = 64):
        super(ConvolutionalEncoder, self).__init__()
        self.grid_size = grid_size
        self.encoder_hidden_dim = encoder_hidden_dim
        self.local_encoder = DeepSetEncoder(x_dim, encoder_hidden_dim, encoder_hidden_dim)
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(encoder_hidden_dim, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )

    def get_x_in_grid(self, x, **kwargs):
        pass

    def divide_bins(self, x):
        batch_size, _, _ = x.shape
        grid_h, grid_w = self.grid_size

        # Build regular grid indices from each batch element's coordinate range.
        x_min = x[:, :, 0].amin(dim=1, keepdim=True)
        x_max = x[:, :, 0].amax(dim=1, keepdim=True)
        y_min = x[:, :, 1].amin(dim=1, keepdim=True)
        y_max = x[:, :, 1].amax(dim=1, keepdim=True)

        x_span = (x_max - x_min).clamp_min(1e-8)
        y_span = (y_max - y_min).clamp_min(1e-8)

        x_norm = (x[:, :, 0] - x_min) / x_span
        y_norm = (x[:, :, 1] - y_min) / y_span

        grid_x = torch.clamp((x_norm * grid_h).long(), min=0, max=grid_h - 1)
        grid_y = torch.clamp((y_norm * grid_w).long(), min=0, max=grid_w - 1)

        latent_grid = x.new_zeros((batch_size, grid_h, grid_w, self.encoder_hidden_dim))

        for b in range(batch_size):
            for i in range(grid_h):
                for j in range(grid_w):
                    x_cell = self.get_x_in_grid(
                        x,
                        batch_idx=b,
                        cell_x=i,
                        cell_y=j,
                        grid_x=grid_x,
                        grid_y=grid_y
                    )

                    if x_cell is None:
                        mask = (grid_x[b] == i) & (grid_y[b] == j)
                        x_cell = x[b, mask]

                    if x_cell.numel() == 0:
                        continue

                    if x_cell.dim() == 2:
                        x_cell = x_cell.unsqueeze(0)

                    cell_latent = self.local_encoder(x_cell)
                    if isinstance(cell_latent, tuple):
                        cell_latent = cell_latent[0]

                    latent_grid[b, i, j] = cell_latent.squeeze(0)

        return latent_grid