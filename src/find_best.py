import wandb

# 设置项目名称
project_name = "Sleep Unet"

# 初始化WandB API
api = wandb.Api()

# 获取项目中的所有运行
runs = api.runs(project_name)

# 找到具有最小验证损失的运行
best_run = None
best_loss = float('inf')
for run in runs:
    # 假设我们记录的指标名称是 "validation_loss"
    val_loss = run.summary.get('test_acc')
    if val_loss is not None and val_loss < best_loss:
        best_loss = val_loss
        best_run = run

# 打印最佳运行的配置信息
if best_run:
    print(f"Best run ID: {best_run.id}")
    print(f"Best validation loss: {best_loss}")
    print("Best configuration:")
    for key, value in best_run.config.items():
        print(f"  {key}: {value}")
else:
    print("No runs found.")
