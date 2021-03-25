from torch import nn


class Generator(nn.Module):
    def __init__(self, g_depth: int, image_channels: int, latent_dimension: int) -> None:
        super().__init__()

        def block(in_channels, out_channels, size):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                # nn.LayerNorm([out_channels, size, size])
            )

        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dimension, 8 * g_depth, (4, 4), (1, 1), (0, 0), bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8 * g_depth),
            # nn.LayerNorm([8 * g_depth, 4, 4]),
        )
        self.block2 = block(8 * g_depth, 4 * g_depth, 8)
        self.block3 = block(4 * g_depth, 2 * g_depth, 16)
        self.block4 = block(2 * g_depth, g_depth, 32)
        self.block5 = nn.Sequential(
            nn.ConvTranspose2d(g_depth, image_channels, (4, 4), (2, 2), (1, 1), bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x
