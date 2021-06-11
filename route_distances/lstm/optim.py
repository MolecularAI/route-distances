""" Module containing an objective class for Optuna optimization """
import torch
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer

from route_distances.lstm.data import TreeDataModule
from route_distances.lstm.models import RouteDistanceModel

EPOCHS = 20
SPLIT_PART = 0.2


class OptunaObjective:
    """
    Representation of an objective function for Optuna


    :param filename: the path to a pickle file with pre-processed trees
    """

    def __init__(self, filename: str) -> None:
        self._filename = filename

    def __call__(self, trial: optuna.trial.Trial) -> float:
        data = TreeDataModule(
            self._filename,
            batch_size=trial.suggest_int("batch_size", 32, 160, 32),
        )
        kwargs = {
            "lstm_size": trial.suggest_categorical("lstm_size", [512, 1024, 2048]),
            "dropout_prob": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
            "learning_rate": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-3, 1e-1, log=True),
        }
        model = RouteDistanceModel(**kwargs)

        gpus = int(torch.cuda.is_available())
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor="val_monitor")
        trainer = Trainer(
            gpus=gpus,
            logger=True,  # become a tensorboard logger
            checkpoint_callback=False,
            callbacks=[pruning_callback],  # type: ignore
            max_epochs=EPOCHS,
        )
        trainer.fit(model, datamodule=data)
        return trainer.callback_metrics["val_monitor"].item()
