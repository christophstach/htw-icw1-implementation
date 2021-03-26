from torch import nn


class Discriminator(nn.Module):
    def __init__(self, d_depth: int, image_channels: int) -> None:
        super().__init__()

        def block(in_channels, out_channels, size):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, (4, 4), (2, 2), (1, 1), bias=False),
                nn.LeakyReLU(0.2),
                nn.LayerNorm([out_channels, size, size]),
                # nn.InstanceNorm2d(out_channels, affine=True)
            )

        def weights_init(m):
            class_name = m.__class__.__name__

            if class_name.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif class_name.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        self.block1 = block(image_channels, d_depth, 32)
        self.block2 = block(d_depth, 2 * d_depth, 16)
        self.block3 = block(2 * d_depth, 4 * d_depth, 8)
        self.block4 = block(4 * d_depth, 8 * d_depth, 4)
        self.block5 = nn.Conv2d(8 * d_depth, 1, (4, 4), (1, 1), (0, 0), bias=False)

        self.apply(weights_init)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        return x
