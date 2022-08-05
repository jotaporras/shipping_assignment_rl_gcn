import wandb
import numpy as np
import pytorch_lightning as ptl
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader


class TestModule(ptl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch):
        print("Logging to wandb")
        self.log("loss", np.random.randn())
        self.log("total_interplants", np.random.randint(0, 255))

    def train_dataloader(self):
        return DataLoader(dataset=list(range(100)))


if __name__ == "__main__":
    wandb.init(
        project="rl_warehouse_assignment",
        name="deleteme_wandb_test",
        tags=["debug", "deleteme"],
    )

    # for i in range(10):
    #     wandb.log({"loss": np.random.randn()}, commit=False)
    #     wandb.log({"total_interplants": np.random.randint(0, 255)}, commit=False)
    #     wandb.log({"episodes": i}, commit=True)
    model = TestModule()
    wandb_logger = WandbLogger(
        project="rl_warehouse_assignment",
        name="deleteme_wandb_test",
        tags=["debug", "deleteme"],
    )
    ptl.Trainer(
        # max_epochs=hparams["max_episodes"], # old way.
        # New way TODO test compare diff agents make sure they work.
        max_epochs=5,
        # early_stop_callback=False,
        logger=wandb_logger,
        progress_bar_refresh_rate=0,  # Todo maybe parameterize
        log_every_n_steps=1,
        # row_log_interval=1,  # the default of this may leave info behind.
        callbacks=[],
    )
