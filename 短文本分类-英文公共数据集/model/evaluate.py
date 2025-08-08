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
    precision_recall_fscore_support
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
    
    # 计算各类别的指标
    class_report = classification_report(labels, predictions, output_dict=True)
    
    # 计算每个类别的精确率、召回率和F1值
    per_class_metrics = {}
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(labels, predictions, average=None)
    
    # 计算micro、macro和weighted指标
    precision_micro = precision_score(labels, predictions, average='micro')
    precision_macro = precision_score(labels, predictions, average='macro')
    precision_weighted = precision_score(labels, predictions, average='weighted')
    
    recall_micro = recall_score(labels, predictions, average='micro')
    recall_macro = recall_score(labels, predictions, average='macro')
    recall_weighted = recall_score(labels, predictions, average='weighted')
    
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')
    f1_weighted = f1_score(labels, predictions, average='weighted')
    
    # 确保所有类别都被包含在per_class_metrics中
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
            'weighted': float(precision_weighted),
            'per_class': {class_name: per_class_metrics[class_name]['precision'] 
                          for class_name in class_names if class_name in per_class_metrics}
        },
        'recall': {
            'micro': float(recall_micro),
            'macro': float(recall_macro),
            'weighted': float(recall_weighted),
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