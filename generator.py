class Generator(nn.Module):
    def __init__(self, g_depth: int, image_channels: int, latent_dimension: int) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dimension, 64 * g_depth, (4, 4), (1, 1), (0, 0)), # 4x4
            nn.LeakyReLU(0.25, True),
            # Upsample

            nn.Conv2d(64 * g_depth, 32 * g_depth, (3, 3), (1, 1), (1, 1)), # 8x8  
            nn.LeakyReLU(0.25, True),
            # Upsample

            nn.Conv2d(32 * g_depth, 16 * g_depth, (3, 3), (1, 1), (1, 1)), # 16x16  
            nn.LeakyReLU(0.25, True),
            # Upsample

            nn.Conv2d(16 * g_depth, 8 * g_depth, (3, 3), (1, 1), (1, 1)), # 32x32  
            nn.LeakyReLU(0.25, True),
            # Upsample

            nn.Conv2d(8 * g_depth, 4 * g_depth, (3, 3), (1, 1), (1, 1)), # 64x64  
            nn.LeakyReLU(0.25, True),

            nn.Conv2d(4 * g_depth, image_channels, (1, 1), (1, 1), (0, 0))
            nn.Tanh(True)
        )