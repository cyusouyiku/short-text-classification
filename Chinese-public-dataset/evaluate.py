import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc, precision_recall_curve
)

def compute_metrics(predictions, labels, class_names=None):
    """
    计算评估指标，包括整体指标和各类别详细指标
    
    Args:
        predictions: 模型预测的类别
        labels: 真实标签
        class_names: 类别名称列表，默认为None，会使用数字索引
        
    Returns:
        包含各种评估指标的字典
    """
    # 转换为numpy数组以进行评估
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # 如果没有提供类别名称，使用数字索引
    if class_names is None:
        class_names = [str(i) for i in range(max(max(predictions), max(labels)) + 1)]
    
    # 计算整体指标
    accuracy = accuracy_score(labels, predictions)
    precision_micro = precision_score(labels, predictions, average='micro')
    precision_macro = precision_score(labels, predictions, average='macro')
    recall_micro = recall_score(labels, predictions, average='micro')
    recall_macro = recall_score(labels, predictions, average='macro')
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    # 计算各类别的指标
    class_report = classification_report(labels, predictions, output_dict=True)
    
    # 计算每个类别的精确率、召回率和F1值
    per_class_metrics = {}
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(labels, predictions, average=None)
    
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1-score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
    
    # 构建返回的指标字典
    metrics = {
        'accuracy': float(accuracy),
        'precision': {
            'micro': float(precision_micro),
            'macro': float(precision_macro),
            'per_class': {class_name: per_class_metrics[class_name]['precision'] 
                          for class_name in class_names if class_name in per_class_metrics}
        },
        'recall': {
            'micro': float(recall_micro),
            'macro': float(recall_macro),
            'per_class': {class_name: per_class_metrics[class_name]['recall'] 
                          for class_name in class_names if class_name in per_class_metrics}
        },
        'f1': {
            'micro': float(f1_micro),
            'macro': float(f1_macro),
            'weighted': float(f1_weighted),
            'per_class': {class_name: per_class_metrics[class_name]['f1-score'] 
                          for class_name in class_names if class_name in per_class_metrics}
        },
        'support': {class_name: per_class_metrics[class_name]['support'] 
                    for class_name in class_names if class_name in per_class_metrics},
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
        'class_report': class_report
    }
    
    # 添加Actor-Critic特定的评估指标
    # 这里将在evaluate_model函数中计算
    
    return metrics

def plot_metrics(metrics, output_path=None):
    """
    可视化评估指标
    
    Args:
        metrics: 评估指标字典
        output_path: 保存图像的路径，如果为None则显示图像
    """
    # 创建一个2x2的子图布局
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # 绘制混淆矩阵
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix', fontsize=14)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    
    # 绘制各类别F1分数
    f1_per_class = metrics['f1']['per_class']
    class_names = list(f1_per_class.keys())
    f1_scores = list(f1_per_class.values())
    
    # 按F1分数排序
    sorted_indices = np.argsort(f1_scores)
    sorted_class_names = [class_names[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]
    
    # 绘制条形图
    axes[0, 1].barh(sorted_class_names, sorted_f1_scores, color='skyblue')
    axes[0, 1].set_title('F1 Score per Class', fontsize=14)
    axes[0, 1].set_xlabel('F1 Score', fontsize=12)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].grid(axis='x', linestyle='--', alpha=0.6)
    
    # 绘制精确率和召回率对比
    precision_per_class = [metrics['precision']['per_class'][class_name] for class_name in class_names]
    recall_per_class = [metrics['recall']['per_class'][class_name] for class_name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, precision_per_class, width, label='Precision', color='skyblue')
    axes[1, 0].bar(x + width/2, recall_per_class, width, label='Recall', color='lightgreen')
    axes[1, 0].set_title('Precision vs Recall per Class', fontsize=14)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(class_names, rotation=45)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.6)
    
    # 绘制支持度条形图
    support = [metrics['support'][class_name] for class_name in class_names]
    axes[1, 1].bar(class_names, support, color='lightcoral')
    axes[1, 1].set_title('Support per Class', fontsize=14)
    axes[1, 1].set_xlabel('Class', fontsize=12)
    axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
    axes[1, 1].set_xticklabels(class_names, rotation=45)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示图像
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model, dataloader, device, class_names=None, output_path=None, compute_rl_metrics=True):
    """
    评估模型性能，计算多种评估指标
    
    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备(CPU/GPU)
        class_names: 类别名称列表
        output_path: 输出路径，用于保存可视化结果
        compute_rl_metrics: 是否计算强化学习相关指标
        
    Returns:
        评估指标字典
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_policy_probs = []
    all_rewards = []
    all_true_rewards = []
    
    # 禁用梯度计算来加速推理
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # 准备数据
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            positions = batch['positions'].to(device) if 'positions' in batch else None
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(input_ids, attention_mask, positions)
            logits = outputs['logits']
            policy_probs = outputs['policy_probs']
            
            # 获取预测结果
            predictions = torch.argmax(logits, dim=-1)
            
            # 如果计算强化学习指标，获取奖励预测
            if compute_rl_metrics and 'predicted_reward' in outputs:
                predicted_rewards = outputs['predicted_reward']
                # 计算真实奖励（预测正确为1，错误为0）
                true_rewards = (predictions == labels).float()
                
                all_rewards.extend(predicted_rewards.cpu().numpy())
                all_true_rewards.extend(true_rewards.cpu().numpy())
                all_policy_probs.extend(policy_probs.cpu().numpy())
            
            # 存储预测结果和标签
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算标准分类指标
    metrics = compute_metrics(all_predictions, all_labels, class_names)
    
    # 如果计算强化学习相关指标
    if compute_rl_metrics and all_rewards:
        # 转换为numpy数组
        all_rewards = np.array(all_rewards)
        all_true_rewards = np.array(all_true_rewards)
        all_policy_probs = np.array(all_policy_probs)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # 计算奖励预测MSE
        reward_mse = ((all_rewards - all_true_rewards) ** 2).mean()
        
        # 计算平均预测置信度（按正确/错误分组）
        correct_indices = (all_predictions == all_labels)
        incorrect_indices = ~correct_indices
        
        # 为每个预测计算其对应类别的置信度
        prediction_confidences = np.array([all_policy_probs[i, pred] for i, pred in enumerate(all_predictions)])
        
        # 计算正确和错误预测的平均置信度
        avg_confidence_correct = prediction_confidences[correct_indices].mean() if any(correct_indices) else 0
        avg_confidence_incorrect = prediction_confidences[incorrect_indices].mean() if any(incorrect_indices) else 0
        
        # 添加到指标字典
        metrics['actor_critic'] = {
            'reward_mse': float(reward_mse),
            'avg_confidence': float(prediction_confidences.mean()),
            'avg_confidence_correct': float(avg_confidence_correct),
            'avg_confidence_incorrect': float(avg_confidence_incorrect),
            'confidence_gap': float(avg_confidence_correct - avg_confidence_incorrect)
        }
        
        # 计算奖励预测的相关性
        correlation = np.corrcoef(all_rewards, all_true_rewards)[0, 1]
        metrics['actor_critic']['reward_correlation'] = float(correlation)
    
    # 可视化结果
    if output_path:
        plot_metrics(metrics, output_path)
    
    return metrics

def predict(model, dataloader, device):
    """
    使用模型对数据集进行预测
    
    Args:
        model: 训练好的模型
        dataloader: 数据加载器
        device: 设备 (CPU/GPU)
        
    Returns:
        predictions: 预测标签列表
        probabilities: 预测概率列表
        rewards: 如果模型支持，返回预测的奖励值
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_rewards = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="预测"):
            # 如果批次为None或为空，则跳过
            if batch is None:
                continue
                
            try:
                # 确保batch是一个字典
                if not isinstance(batch, dict):
                    print(f"批次不是字典: {type(batch)}")
                    continue
                
                # 确保必要的键存在
                if 'input_ids' not in batch:
                    print(f"批次缺少input_ids字段，可用的键: {batch.keys()}")
                    continue
                
                # 前向传播
                if hasattr(model, 'predict_with_rewards'):
                    # 解包批次字典
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    positions = batch.get('positions')
                    if positions is not None:
                        positions = positions.to(device)
                    
                    logits, rewards = model.predict_with_rewards(input_ids, attention_mask, positions)
                    all_rewards.extend(rewards.cpu().numpy())
                else:
                    # 解包批次字典
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    positions = batch.get('positions')
                    if positions is not None:
                        positions = positions.to(device)
                    
                    # 调用模型时传递解包后的参数
                    outputs = model(input_ids, attention_mask, positions)
                    # 从输出字典中提取logits
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                        if 'predicted_reward' in outputs:
                            all_rewards.extend(outputs['predicted_reward'].cpu().numpy())
                    else:
                        logits = outputs  # 如果输出不是字典，则假设它就是logits
                
                # 将logits转换为概率
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = np.argmax(probs, axis=1)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probs)
            except Exception as e:
                print(f"批次处理错误: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(all_rewards) > 0:
        return all_predictions, all_probabilities, all_rewards
    else:
        return all_predictions, all_probabilities

def generate_evaluation_report(metrics, output_path=None):
    """
    生成评估报告的Markdown文档
    
    Args:
        metrics: 评估指标字典
        output_path: 保存报告的路径
        
    Returns:
        report_md: Markdown格式的报告
    """
    report_md = "# 模型评估报告\n\n"
    
    # 主要指标
    report_md += "## 主要评估指标\n\n"
    report_md += f"- **准确率 (Accuracy)**: {metrics['accuracy']:.4f}\n"
    report_md += f"- **F1 (微平均)**: {metrics['f1']['micro']:.4f}\n"
    report_md += f"- **F1 (宏平均)**: {metrics['f1']['macro']:.4f}\n"
    report_md += f"- **F1 (加权平均)**: {metrics['f1']['weighted']:.4f}\n\n"
    
    # 每个类别的指标
    report_md += "## 每个类别的指标\n\n"
    report_md += "| 类别 | F1 | 精确率 | 召回率 | 支持数 |\n"
    report_md += "|------|----|---------|---------|---------|\n"
    
    for class_name in metrics['support'].keys():
        f1 = metrics['f1']['per_class'].get(class_name, 0)
        precision = metrics['precision']['per_class'].get(class_name, 0)
        recall = metrics['recall']['per_class'].get(class_name, 0)
        support = metrics['support'].get(class_name, 0)
        
        report_md += f"| {class_name} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {support} |\n"
    
    report_md += "\n"
    
    # 置信度指标（如果存在）
    if 'confidence_metrics' in metrics:
        report_md += "## 置信度指标\n\n"
        report_md += f"- **平均预测置信度**: {metrics['confidence_metrics']['avg_confidence']:.4f}\n"
        report_md += f"- **正确预测平均置信度**: {metrics['confidence_metrics']['avg_confidence_correct']:.4f}\n"
        report_md += f"- **错误预测平均置信度**: {metrics['confidence_metrics']['avg_confidence_incorrect']:.4f}\n"
        report_md += f"- **置信度差距**: {metrics['confidence_metrics']['confidence_gap']:.4f}\n\n"
    
    # Actor-Critic指标（如果有）
    if 'actor_critic' in metrics:
        report_md += "## Actor-Critic指标\n\n"
        report_md += f"- **奖励预测MSE**: {metrics['actor_critic']['reward_mse']:.4f}\n"
        report_md += f"- **奖励预测相关性**: {metrics['actor_critic']['reward_correlation']:.4f}\n\n"
    
    # 保存报告
    if output_path:
        report_dir = os.path.dirname(output_path)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        print(f"评估报告已保存到 {output_path}")
    
    return report_md

def calculate_metrics(y_true, y_pred, y_probs, class_names=None):
    """
    计算各种分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_probs: 预测概率
        class_names: 类别名称列表
        
    Returns:
        metrics: 包含各种指标的字典
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # 如果没有提供类别名称，则使用索引
    if class_names is None:
        class_names = [str(i) for i in range(max(max(y_true), max(y_pred)) + 1)]
    
    # 计算基本指标
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算每个类别的精确率、召回率、F1和支持度
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None)
    
    # 将类别指标转换为字典形式
    precision_dict = {class_names[i]: precision_per_class[i] for i in range(len(class_names)) if i < len(precision_per_class)}
    recall_dict = {class_names[i]: recall_per_class[i] for i in range(len(class_names)) if i < len(recall_per_class)}
    f1_dict = {class_names[i]: f1_per_class[i] for i in range(len(class_names)) if i < len(f1_per_class)}
    support_dict = {class_names[i]: int(support_per_class[i]) for i in range(len(class_names)) if i < len(support_per_class)}
    
    # 创建混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 计算每个类别的ROC曲线和AUC（一对其他）
    roc_curves = {}
    for i in range(len(class_names)):
        if i < y_probs.shape[1]:  # 确保类别索引在概率矩阵范围内
            # 创建二分类问题：当前类别 vs 其他类别
            y_true_binary = (y_true == i).astype(int)
            y_score = y_probs[:, i]
            
            try:
                # 计算ROC曲线和AUC
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                roc_curves[class_names[i]] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc
                }
            except Exception as e:
                print(f"计算类别 {i} 的ROC曲线时出错: {e}")
                roc_curves[class_names[i]] = {
                    'fpr': [],
                    'tpr': [],
                    'auc': 0.0
                }
    
    # 计算置信度相关指标
    confidences = np.max(y_probs, axis=1)
    
    # 计算正确预测和错误预测的平均置信度
    correct_indices = (y_pred == y_true)
    avg_confidence = np.mean(confidences)
    avg_confidence_correct = np.mean(confidences[correct_indices]) if np.any(correct_indices) else 0
    avg_confidence_incorrect = np.mean(confidences[~correct_indices]) if np.any(~correct_indices) else 0
    confidence_gap = avg_confidence_correct - avg_confidence_incorrect
    
    # 整合所有指标
    metrics = {
        'accuracy': float(accuracy),
        'precision': {
            'micro': float(precision_micro),
            'macro': float(precision_macro),
            'weighted': float(precision_weighted),
            'per_class': precision_dict
        },
        'recall': {
            'micro': float(recall_micro),
            'macro': float(recall_macro),
            'weighted': float(recall_weighted),
            'per_class': recall_dict
        },
        'f1': {
            'micro': float(f1_micro),
            'macro': float(f1_macro),
            'weighted': float(f1_weighted),
            'per_class': f1_dict
        },
        'support': support_dict,
        'confusion_matrix': cm.tolist(),
        'roc_curves': roc_curves,
        'confidence_metrics': {
            'avg_confidence': float(avg_confidence),
            'avg_confidence_correct': float(avg_confidence_correct),
            'avg_confidence_incorrect': float(avg_confidence_incorrect),
            'confidence_gap': float(confidence_gap)
        }
    }
    
    return metrics

def generate_class_distributions(y_true, y_pred, class_names, output_path):
    """
    生成类别分布的可视化
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        output_path: 保存图表的路径
    """
    # 计算每个类别的真实和预测分布
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    
    # 创建DataFrame用于可视化
    df = pd.DataFrame({
        '类别': class_names,
        '真实分布': true_counts,
        '预测分布': pred_counts
    })
    
    # 转换为长格式用于绘图
    df_melted = pd.melt(df, id_vars=['类别'], value_vars=['真实分布', '预测分布'], 
                       var_name='分布类型', value_name='样本数')
    
    # 绘制分布对比图
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='类别', y='样本数', hue='分布类型', data=df_melted)
    
    plt.title('真实分布 vs 预测分布')
    plt.xlabel('类别')
    plt.ylabel('样本数')
    
    # 在条形上方显示数值
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"类别分布可视化已保存到 {output_path}")
