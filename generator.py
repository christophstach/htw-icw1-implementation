from torch import nn


class Generator(nn.Module):
    def __init__(self, g_depth: int, image_channels: int, latent_dimension: int) -> None:
        super().__init__()

        def block(in_channels, out_channels, size):
            return nn.Sequential(
                # nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
                # nn.Conv2d(in_channels, out_channels, (3, 3), (1, 1), (1, 1), bias=False),
                nn.ConvTranspose2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
                nn.ReLU(),
                nn.BatchNorm2d(out_channels),
                # nn.LayerNorm([out_channels, size, size])
            )

        def weights_init(m):
            class_name = m.__class__.__name__

            if class_name.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif class_name.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

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
            # nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=False),
            # nn.Conv2d(2 * g_depth, image_channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x
