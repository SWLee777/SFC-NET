import random
import shutil
import time
import torch
# from torch.utils.tensorboard import SummaryWriter

from utils.visualization import *
from loguru import logger

def get_optimizer_from_args(model, lr, weight_decay, **kwargs) -> torch.optim.Optimizer:
    return torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                             weight_decay=weight_decay)


def get_lr_schedule(optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_dir_from_args(root_dir, class_name, dataset, k_shot, experiment_indx):
    """根据固定参数生成目录路径"""
    # 生成实验名称
    exp_name = f"{dataset}-k-{k_shot}"

    # CSV路径
    csv_dir = os.path.join(root_dir, 'csv')
    csv_path = os.path.join(csv_dir, f"{exp_name}-indx-{experiment_indx}.csv")

    # 模型、图像、日志目录
    model_dir = os.path.join(root_dir, exp_name, 'models')
    img_dir = os.path.join(root_dir, exp_name, 'imgs')
    logger_dir = os.path.join(root_dir, exp_name, 'logger', class_name)

    # 日志文件路径（带时间戳）
    log_file_name = os.path.join(
        logger_dir,
        f'log_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))}.log'
    )

    # 创建目录
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(logger_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # 初始化日志
    logger.start(log_file_name)
    logger.info(f"==> Root dir: {logger_dir}")

    # 返回所有路径及模型名称
    return model_dir, img_dir, logger_dir, f"{class_name}", csv_path