import argparse
import logging
import logging.config
import os

import torch
from dotenv import load_dotenv

# important: load env variables before any file imports!
load_dotenv(".env")

from train import train
from utils.experiments import (
    create_experiment,
    load_config,
    setup_loggers,
    store_config,
)
from utils.logging_config import LOGGING_CONFIG


def main():
    """CL interface
    """
    parser = argparse.ArgumentParser(description="PSGN")

    # use experiment
    parser.add_argument(
        "-n",
        "--name",
        nargs="?",
        type=str,
        help="name of the experiment directory to use",
        default=None,
    )
    parser.add_argument('--resume', action='store_true', default=False, help='Resume training specified by the exp_name')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    # resume from checkpoint
    parser.add_argument(
        "-from",
        "--from_epoch",
        nargs="?",
        help="use model from checkpoint at epoch",
        default=0,
    )
    # for how many epochs to train
    parser.add_argument(
        "-epochs",
        "--num_epochs",
        nargs="?",
        type=str,
        help="how many epochs the model should be trained",
        default=None,
    )

    subparsers = parser.add_subparsers(help="commands", dest="command")

    # training command
    train_parser = subparsers.add_parser("train", help="run training")

    overfit_parser = subparsers.add_parser("overfit", help="run overfitting")

    # NOTE: you can add other commands here

    args = parser.parse_args()
    # configure logging
    logging.config.dictConfig(LOGGING_CONFIG)

    # root logger only logs to console
    log = logging.getLogger("root")

    # if no command specified
    if args.command == None:
        log.warning("No command specified")
        return

    # setup device if --use-cuda has been specified
    # log.info(f"CUDA_VISIBLE_DEVICES: {str(os.environ['CUDA_VISIBLE_DEVICES'])}")
    if torch.cuda.is_available():
        device = "cuda:0"
        log.info(f"Using device: {device}")
    else:
        device = "cpu"
        log.warning("Using cpu!")

    config = load_config(args.name)
    config.device = device

    if args.from_epoch != 0:
        # resume from checkpoint
        if isinstance(args.from_epoch, int):
            config.checkpoint = int(args.from_epoch)
        else:
            config.checkpoint = str(args.from_epoch)
    else:
        config.checkpoint = 0

    if args.num_epochs is not None:
        # train/overfit for num_epochs
        config.max_epochs = int(args.num_epochs)

    if args.command == "train":
        config.overfit = False
    elif args.command == "overfit":
        config.overfit = True

    # create experiment
    if not args.resume:
        config = create_experiment(config)

    # NOTE: do not modify config after this
    store_config(config)

    # NOTE: train and overfit are sharing the same logger for convenience, can change this in the future
    setup_loggers("train", log_path=os.path.join(config.exp_dir, "train.log"))

    train(config, args)


if __name__ == "__main__":
    main()
