class Discriminator(nn.Module):
    def __init__(self, g_depth: int, image_channels: int, latent_dimension: int) -> None:
        super().__init__()

        self.main = nn.Sequential(
            nn.Conv2d(image_channels, 4 * d_depth, (3, 3), (1, 1), (1, 1)) # 64x64
            nn.LeakyReLU(0.25, True),
            nn.AvgPool(2, 2)

            nn.Conv2d(4 * d_depth, 8 * d_depth, (3, 3), (1, 1), (1, 1)) # 32x32
            nn.LeakyReLU(0.25, True),
            nn.AvgPool(2, 2)

            nn.Conv2d(8 * d_depth, 16 * d_depth, (3, 3), (1, 1), (1, 1)) # 16x16
            nn.LeakyReLU(0.25, True),
            nn.AvgPool(2, 2)

            nn.Conv2d(16 * d_depth, 32 * d_depth, (3, 3), (1, 1), (1, 1)) # 8x8
            nn.LeakyReLU(0.25, True),
            nn.AvgPool(2, 2)

            nn.Conv2d(32 * d_depth, 64 * d_depth, (3, 3), (1, 1), (1, 1)) # 4x4
            nn.LeakyReLU(0.25, True),

            nn.Conv2d(64 * d_depth, 1, (1, 1), (1, 1), (0, 0))
        )