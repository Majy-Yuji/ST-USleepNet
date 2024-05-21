import wandb
import yaml
from main import main

# 读取sweep.yaml文件
with open("configs//sweep.yaml") as file:
    sweep_config = yaml.safe_load(file)

# 初始化Sweep
sweep_id = wandb.sweep(sweep_config, project ='Sleep Unet')

# 运行Sweep
wandb.agent(sweep_id, function = main)
