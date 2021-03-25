import torch

from discrimnator import Discriminator
from generator import Generator

torch.manual_seed(0)

generator = Generator(1, 3, 128)
discriminator = Discriminator(1, 3)

z = torch.randn(32, 128, 1, 1)

x = generator(z)
print(x.shape)

score = discriminator(x)

print(score.shape)