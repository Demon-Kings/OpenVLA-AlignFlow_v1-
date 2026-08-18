import os
import re
import json
import shutil
import subprocess
import logging
import time
import csv

# ==============================================================================
# OpenVLA-AlignFlow 单卡 4090 极客版：动态 PID 巡航训练引擎 (v3: 全量 20 维全景指标溯源)
# 包含功能: 自动分块训练、核心 5 维 PID 动态反馈、全量 20 维指标存表、🏆 最佳参数自动锁存
# ==============================================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "vla", "configs", "config.py")
CONFIG_BAK_PATH = os.path.join(PROJECT_ROOT, "vla", "configs", "config.py.bak")
REPORT_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "benchmark_report_latest.json")
HISTORY_LOG_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "pid_tuning_history_full.csv")
BEST_PARAMS_PATH = os.path.join(PROJECT_ROOT, "checkpoints", "best_pid_parameters.json")

logging.basicConfig(level=logging.INFO, format='[PID-Engine] %(asctime)s - %(message)s')

# 全量指标映射字典，用于确保 CSV 的列序与 JSON 数据对应
METRIC_KEYS_MAP = {
    "Recall_R1": "Sub-Goal Recall R@1",
    "Recall_R5": "Sub-Goal Recall R@5",
    "COD_mm": "Contact Offset Distance",
    "RER_pct": "Resonance Energy Ratio",
    "Jerk": "Physical Mean Jerk",
    "Violation_pct": "Workspace Violation Rate",
    "Min_L1": "Min-of-5 Trajectory L1",
    "Single_L1": "Single-Rollout L1",
    "Traj_MSE": "Trajectory MSE",
    "SO3_Rad": "SO(3) Geodesic Error",
    "Hausdorff_cm": "Hausdorff 3D Envelope",
    "FAD": "Frechet Action Distance",
    "Mode_Entropy": "Mode Coverage Entropy",
    "Momentum_Surge": "Contact Momentum Surge",
    "Manipulability": "Manipulability Index",
    "Kinetic_Smoothness": "Kinetic Energy Smoothness",
    "Affordance_IoU": "Spatial Affordance Attention IoU",
    "Kendall_Tau": "Causal Kendall's Tau",
    "DTW": "DTW Temporal Causality Distance",
    "CBF_Margin": "CBF Safety Barrier Margin"
}

class DynamicPIDScheduler:
    def __init__(self, initial_weight=0.75):
        self.current_damping_weight = initial_weight
        self.kp_rer = 0.02      
        self.kp_recall = 0.01   
        self.target_rer = 15.0  
        self.target_recall = 50.0 

    def calculate_next_weight(self, current_rer, current_recall):
        error_rer = max(0.0, current_rer - self.target_rer)
        error_recall = max(0.0, self.target_recall - current_recall)
        delta_weight = (self.kp_rer * error_rer) - (self.kp_recall * error_recall)
        self.current_damping_weight += delta_weight
        self.current_damping_weight = max(0.1, min(2.5, self.current_damping_weight))
        logging.info(f"PID 动态修正幅值 = {delta_weight:+.4f}")
        return self.current_damping_weight

def update_config_file(param_name, value):
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    pattern = rf"^(\s*{param_name}\s*=\s*)[0-9eE\.\-]+(.*)$"
    content = re.sub(pattern, rf"\g<1>{value:.4f}\g<2>", content, flags=re.MULTILINE)
    with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
        f.write(content)

def parse_full_report():
    """不再只解析 5 个指标，而是把整个评测报告的 JSON 字典全部读出"""
    if not os.path.exists(REPORT_PATH):
        return None
    try:
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"解析全量报告失败: {e}")
        return None

def init_history_log():
    """生成包含 20 多列的巨型 CSV 追踪表"""
    os.makedirs(os.path.dirname(HISTORY_LOG_PATH), exist_ok=True)
    if not os.path.exists(HISTORY_LOG_PATH):
        with open(HISTORY_LOG_PATH, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            # 基础信息 + 适应度得分 + 所有指标键名
            headers = ["Chunk", "Timestamp", "Energy_Damping_Weight", "Fitness_Score"] + list(METRIC_KEYS_MAP.keys())
            writer.writerow(headers)

def append_history_log(chunk, weight, fitness, report_dict):
    """将整整 20 个指标全部写入 CSV 存档"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    row = [chunk, timestamp, f"{weight:.4f}", f"{fitness:.4f}"]
    
    # 按照字典顺序依次提取指标值，防范缺失值
    for short_key, full_key in METRIC_KEYS_MAP.items():
        val = report_dict.get(full_key, 0.0)
        # 针对可能存在的字符串或者奇怪格式做数值安全转化
        if isinstance(val, (int, float)):
            row.append(f"{val:.4f}")
        else:
            row.append(str(val))
            
    with open(HISTORY_LOG_PATH, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def save_best_params(chunk, weight, fitness, report_dict):
    """保存全局最优参数，并完整存下此时的所有 20 个指标"""
    data = {
        "best_chunk": chunk,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "best_energy_damping_weight": round(weight, 4),
        "Comprehensive_Fitness_Score": round(fitness, 4),
        "full_metrics_record": report_dict,
        "description": "基于五维全景公式锁存的最优配置，附带所有 20 维物理/时序指标的留档。"
    }
    with open(BEST_PARAMS_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    logging.info(f"🏆 发现新的五维全局最佳参数! (Fitness: {fitness:.2f}) 已全量锁存。")

def main():
    os.chdir(PROJECT_ROOT)
    
    if not os.path.exists(CONFIG_BAK_PATH):
        shutil.copy2(CONFIG_PATH, CONFIG_BAK_PATH)
    
    init_history_log()
    pid_scheduler = DynamicPIDScheduler(initial_weight=0.75)
    
    total_chunks = 9
    epochs_per_chunk = 5  
    global_best_fitness = float('-inf')
    
    logging.info("🚀 启动 4090 动态 PID 引擎 (采用五维适应度打分，并执行 20 维全量存档)")
    
    try:
        for chunk in range(1, total_chunks + 1):
            logging.info(f"\n[{chunk}/{total_chunks}] 正在使用 Damping Weight: {pid_scheduler.current_damping_weight:.4f} 进行训练...")
            
            update_config_file("energy_damping_weight", pid_scheduler.current_damping_weight)
            update_config_file("stage3_epochs", epochs_per_chunk) 
            
            subprocess.run(["python", "run_pipeline.py", "--start_stage", "3"], capture_output=False)
            subprocess.run(["python", "run_pipeline.py", "--start_stage", "eval", "--eval_num_samples", "100"], capture_output=False)
            
            report_dict = parse_full_report()
            
            if report_dict:
                # -------------------------------------------------------------
                # PID 控制与适应度打分，依然只提取最核心、防作弊的 5 个硬指标
                # -------------------------------------------------------------
                recall = report_dict.get("Sub-Goal Recall R@1", 0.0)
                cod = report_dict.get("Contact Offset Distance", 10.0)
                rer = report_dict.get("Resonance Energy Ratio", 100.0)
                jerk = report_dict.get("Physical Mean Jerk", 50.0)
                violation = report_dict.get("Workspace Violation Rate", 100.0)
                
                current_fitness = (
                    (recall * 1.0)       # 召回率 (通常在 30-80) -> 正向基底分
                    - (cod * 5.0)        # 抓取偏置 (通常 1.5mm) -> 重度精度惩罚
                    - (rer * 0.5)        # 震颤率 (通常 15-30%)  -> 平滑度惩罚
                    - (jerk * 0.5)       # 加加速度 (通常 10-20) -> 电机磨损惩罚
                    - (violation * 1.5)  # 越界率 (通常 0-30%)   -> 绝对安全红线惩罚
                )
                
                logging.info(f"本次核心评测 -> Recall:{recall}%, COD:{cod}mm, RER:{rer}%, Jerk:{jerk}, Violation:{violation}%")
                logging.info(f"📊 综合得分 (Fitness): {current_fitness:.2f}")
                
                # 更新最佳参数
                if current_fitness > global_best_fitness:
                    global_best_fitness = current_fitness
                    save_best_params(chunk, pid_scheduler.current_damping_weight, current_fitness, report_dict)
                
                # PID 计算下一次权重
                pid_scheduler.calculate_next_weight(rer, recall)
                
                # 💾 执行巨型 20 维全量存档 (写入 CSV)
                append_history_log(chunk, pid_scheduler.current_damping_weight, current_fitness, report_dict)
            else:
                logging.warning("未检测到有效报告，维持上一轮参数。")
                
            time.sleep(2)
            
        logging.info("\n🎉 动态 PID 巡航训练完成！")
        
    except KeyboardInterrupt:
        logging.info("收到中断信号，中止巡航引擎...")
    finally:
        if os.path.exists(CONFIG_BAK_PATH):
            shutil.copy2(CONFIG_BAK_PATH, CONFIG_PATH)
        logging.info("已安全还原 config.py。")

if __name__ == "__main__":
    main()
