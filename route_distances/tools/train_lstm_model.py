""" Module for CLI tool to train LSTM-based model """
import argparse

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import route_distances.lstm.defaults as defaults
from route_distances.lstm.data import TreeDataModule
from route_distances.lstm.models import RouteDistanceModel
from route_distances.lstm.utils import accumulate_stats


def _get_args():
    parser = argparse.ArgumentParser(
        "Tool to train an LSTM-based model for route distances"
    )
    parser.add_argument("--trees", required=True)
    parser.add_argument("--epochs", type=int, default=defaults.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=defaults.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=defaults.LEARNING_RATE)
    parser.add_argument("--weight_decay", type=float, default=defaults.WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=defaults.DROPOUT_PROB)
    parser.add_argument("--fp_size", type=int, default=defaults.FP_SIZE)
    parser.add_argument("--lstm_size", type=int, default=defaults.LSTM_SIZE)
    parser.add_argument("--split_part", type=float, default=defaults.SPLIT_PART)
    parser.add_argument("--split_seed", type=float, default=defaults.SPLIT_SEED)
    return parser.parse_args()


def main(seed=None) -> None:
    """Entry-point for CLI tool"""
    args = _get_args()
    print(str(args).replace("Namespace", "Arguments used = "))

    if seed is not None:
        seed_everything(seed)

    data = TreeDataModule(
        args.trees,
        batch_size=args.batch_size,
        split_part=args.split_part,
        split_seed=args.split_seed,
    )

    kwargs = {
        "fp_size": args.fp_size,
        "lstm_size": args.lstm_size,
        "dropout_prob": args.dropout,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
    }
    model = RouteDistanceModel(**kwargs)

    gpus = int(torch.cuda.is_available())
    tb_logger = TensorBoardLogger("tb_logs", name=f"route-dist")
    csv_logger = CSVLogger("csv_logs", name=f"route-dist")
    checkpoint = ModelCheckpoint(monitor="val_monitor", save_last=True)
    trainer = Trainer(
        gpus=gpus,
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint],
        max_epochs=args.epochs,
        deterministic=seed is not None,
    )
    trainer.fit(model, datamodule=data)

    ret = trainer.test(datamodule=data)
    print("=== Test results === ")
    accum = accumulate_stats(ret)
    for key, value in accum.items():
        print(f"{key}: {value:0.4f}")


if __name__ == "__main__":
    main()
