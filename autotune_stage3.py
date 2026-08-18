import os
import re
import json
import shutil
import subprocess
import optuna
import logging

# ==============================================================================
# OpenVLA-AlignFlow 单卡 4090 自适应调参引擎 (Stage 3 DPO 专用)
# 作用: 自动搜索最佳的物理超参数，并在不修改原代​​码的情况下调用原流水线
# ==============================================================================

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "vla", "configs", "config.py")
CONFIG_BAK_PATH = os.path.join(PROJECT_ROOT, "vla", "configs", "config.py.bak")
REPORT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "benchmark_report_latest.json")

# 设置日志
logging.basicConfig(level=logging.INFO, format='[AutoTune] %(asctime)s - %(message)s')

def backup_config():
    """备份原始配置文件"""
    if not os.path.exists(CONFIG_BAK_PATH):
        shutil.copy2(CONFIG_PATH, CONFIG_BAK_PATH)
        logging.info("已备份原始 config.py")

def restore_config():
    """恢复原始配置文件"""
    if os.path.exists(CONFIG_BAK_PATH):
        shutil.copy2(CONFIG_BAK_PATH, CONFIG_PATH)
        logging.info("已恢复原始 config.py")

def update_config_file(params):
    """
    使用正则表达式动态修改 config.py 中的超参数值。
    这种方法绝对安全，不需要修改原工程的导入逻辑。
    """
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()

    for key, value in params.items():
        # 正则匹配形如: key = 0.75
        # 支持整数、浮点数、科学计数法
        pattern = rf"^(\s*{key}\s*=\s*)[0-9eE\.\-]+(.*)$"
        # 替换为新的值
        content = re.sub(pattern, rf"\g<1>{value}\g<2>", content, flags=re.MULTILINE)

    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logging.info(f"已将超参数临时注入 config.py: {params}")

def extract_fitness_score():
    """
    解析评测生成的 JSON 报告，计算综合适应度得分 (Fitness Score)
    目标是最大化这个得分。
    """
    if not os.path.exists(REPORT_PATH):
        logging.error("未找到评测报告，当前 Trial 失败。")
        return float('-inf')
    
    try:
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            report = json.load(f)
            
        # 提取关键物理与感知指标 (注意: 这里的键名需根据您真实 json 中的字段名调整)
        # 假设提取出来的是纯数值
        recall_r1 = report.get("Sub-Goal Recall R@1", 0.0)      # 越大越好 (例如 55.0)
        rer = report.get("Resonance Energy Ratio", 100.0)       # 越小越好 (例如 26.5)
        jerk = report.get("Physical Mean Jerk", 50.0)           # 越小越好 (例如 10.6)
        cod = report.get("Contact Offset Distance", 10.0)       # 越小越好 (例如 1.6)
        violation = report.get("Workspace Violation Rate", 100.0) # 越小越好 (例如 24.8)

        # ---------------------------------------------------------
        # 【核心奖励函数设计 (Reward Shaping)】
        # 权重设计：极度鼓励 Recall，重度惩罚 RER 和 越界率
        # ---------------------------------------------------------
        fitness_score = (
            (recall_r1 * 2.0)        # 召回率是核心能力，权重最高
            - (rer * 1.5)            # 共振能量会导致物理损毁，重度惩罚
            - (jerk * 0.5)           # Jerk 在 25 以下即可，轻度惩罚
            - (cod * 2.0)            # 抓取偏差 (mm) 越小越好
            - (violation * 1.0)      # 越界率惩罚
        )
        
        logging.info(f"指标解析 -> Recall: {recall_r1}%, RER: {rer}%, Jerk: {jerk}, COD: {cod}mm")
        logging.info(f"计算得出综合适应度得分 (Fitness Score): {fitness_score:.4f}")
        return fitness_score

    except Exception as e:
        logging.error(f"解析报告时发生错误: {e}")
        return float('-inf')

def objective(trial):
    """Optuna 优化的目标函数"""
    
    # 1. 让 Optuna 采样这一轮要尝试的超参数 (重点搜索 Stage 3 物理参数)
    sampled_params = {
        # 动能阻尼权重 (决定了接触时的软着陆强度，太小震荡，太大保守)
        "energy_damping_weight": trial.suggest_float("energy_damping_weight", 0.5, 1.8),
        # 旋转测地线损失权重 (平衡旋转与平移)
        "rot_geodesic_weight": trial.suggest_float("rot_geodesic_weight", 0.4, 1.0),
        # DPO 偏好边际 Beta (控制策略偏离参考策略的容忍度)
        "dpo_beta": trial.suggest_float("dpo_beta", 5.0, 15.0),
    }
    
    logging.info(f"\n====================== 启动 Trial {trial.number} ======================")
    
    # 2. 将采样的参数写入 config.py
    update_config_file(sampled_params)
    
    # 3. 通过子进程安全调用 Stage 3 训练 (彻底隔离显存，防止 OOM)
    # 注意: 这里使用 subprocess 是为了保证单卡 4090 在每次跑完后彻底清空 VRAM
    logging.info(">>> 正在启动 Stage 3 (Trajectory-DPO) 训练...")
    train_cmd = ["python", "run_pipeline.py", "--start_stage", "3"]
    result_train = subprocess.run(train_cmd, capture_output=False) # 允许输出打印到控制台
    
    if result_train.returncode != 0:
        logging.warning("训练阶段崩溃 (可能是参数导致 NaN)，剪枝此 Trial。")
        raise optuna.exceptions.TrialPruned()

    # 4. 调用 Offline Benchmark 进行评测
    # 建议自适应调参时，将样本数适当调小 (如 150) 以节约时间，寻优结束后再用 500 样本终测
    eval_samples = "150" 
    logging.info(f">>> 正在启动 4D 评测 (样本数: {eval_samples})...")
    eval_cmd = ["python", "run_pipeline.py", "--start_stage", "eval", "--eval_num_samples", eval_samples]
    result_eval = subprocess.run(eval_cmd, capture_output=False)
    
    if result_eval.returncode != 0:
        logging.warning("评测阶段崩溃，剪枝此 Trial。")
        raise optuna.exceptions.TrialPruned()

    # 5. 计算并返回得分
    score = extract_fitness_score()
    return score

def main():
    # 确保当前在项目根目录
    os.chdir(PROJECT_ROOT)
    
    logging.info("初始化自适应调参引擎 (基于 Optuna)...")
    backup_config()
    
    try:
        # 创建 SQLite 数据库来保存调参进度。这样即使断电，下次也能接着调！
        study_name = "openvla_stage3_optimization"
        storage_name = f"sqlite:///{study_name}.db"
        
        # 目标是最大化适应度得分 (maximize)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            load_if_exists=True
        )
        
        # 开始寻优，设置最大尝试次数 n_trials (例如挂机跑 20 次)
        logging.info("开始自动化寻优，您可以随时按 Ctrl+C 终止，进度会自动保存在数据库中。")
        study.optimize(objective, n_trials=20)
        
        # 输出最优结果
        logging.info("\n====================== 寻优完成 ======================")
        logging.info(f"最佳综合得分 (Best Value): {study.best_value}")
        logging.info("最佳物理参数组合 (Best Params):")
        for key, value in study.best_params.items():
            logging.info(f"    {key}: {value}")
            
    except KeyboardInterrupt:
        logging.info("\n收到中断信号，已安全停止调参。")
    finally:
        # 无论调参成功还是异常中断，都强制还原 config.py，保证不弄脏原工程
        restore_config()
        logging.info("自适应调参引擎安全退出。")

if __name__ == "__main__":
    main()
