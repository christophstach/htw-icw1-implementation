


class MsgGANTrail(PyTorchTrial):
    def __init__(self, context: PyTorchTrialContext) -> None:
        super().__init__(context)

        self.context = context
        self.logger = TorchWriter()

        self.dataset = self.context.get_hparam('dataset')
        self.image_channels = self.context.get_hparam('image_channels')
        self.latent_dimension = self.context.get_hparam('latent_dimension')

        self.g_depth = self.context.get_hparam('g_depth')
        self.d_depth = self.context.get_hparam('d_depth')

        self.generator = Generator()
        self.discriminator = Discriminator()
