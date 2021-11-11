from typing import List, Dict

import pandas as pd
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl

from bike_sharing_demand.models.common import calculate_metric
from bike_sharing_demand.models.dl_models.dataset import DataModule
from bike_sharing_demand.models.dl_models.models.mlp import SimpleMLP


def run_dl_models(x: pd.DataFrame, y: pd.DataFrame) -> Dict:
    pl.seed_everything(42)
    dm = DataModule(x, y)
    dm.setup()
    models_collections = [SimpleMLP()]
    models_score: Dict = dict()
    for model in models_collections:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath="./models",
            filename=str(model) + "_{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
        )
        early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=1, mode="min")
        csv_logger = CSVLogger('./models', name=str(model), version='0'),
        trainer = Trainer(
            logger=csv_logger,
            max_epochs=50,
            progress_bar_refresh_rate=2,
            callbacks=[checkpoint_callback, early_stop_callback]
        )
        trainer.fit(model, dm)
        trainer.test(model)
        predictions = []
        for data, target in dm.test_dataloader():
            predictions.extend(model(x).detach().numpy().squeeze())

    return models_score
