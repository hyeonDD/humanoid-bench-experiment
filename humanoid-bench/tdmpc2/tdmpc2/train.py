import os
import sys
import signal  # 추가
import traceback

if sys.platform != "darwin":
    os.environ["MUJOCO_GL"] = "egl"

os.environ["LAZY_LEGACY_OP"] = "0"
import warnings

warnings.filterwarnings("ignore")
import torch

import hydra
from termcolor import colored

from tdmpc2.common.parser import parse_cfg
from tdmpc2.common.seed import set_seed
from tdmpc2.common.buffer import Buffer
from tdmpc2.envs import make_env
from tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.trainer.offline_trainer import OfflineTrainer
from tdmpc2.trainer.online_trainer import OnlineTrainer
from tdmpc2.common.logger import Logger

torch.backends.cudnn.benchmark = True

# 모델 저장 로직
def save_model(trainer):
    print("\nSIGTERM received. Saving the model...")
    trainer.logger.save_agent(trainer.agent)
    print("Model saved successfully.")

@hydra.main(config_name="config", config_path=".")
def train(cfg: dict):
    """
    Script for training single-task / multi-task TD-MPC2 agents.
    """
    assert cfg.steps > 0, "Must train for at least 1 step."
    cfg = parse_cfg(cfg)
    set_seed(cfg.seed)
    print(colored("Work dir:", "yellow", attrs=["bold"]), cfg.work_dir)

    trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainer
    trainer = trainer_cls(
        cfg=cfg,
        env=make_env(cfg),
        agent=TDMPC2(cfg),
        buffer=Buffer(cfg),
        logger=Logger(cfg),
    )

    # SIGTERM 핸들러 등록
    def handle_sigterm(signum, frame):
        save_model(trainer)
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_sigterm)

    try:
        trainer.train()
        print("\nTraining completed successfully")
    except KeyboardInterrupt:
        save_model(trainer)
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    train()
