import os
import torch
import argparse
import json
import pickle
import numpy as np
from data_process import DataProcessor
from model.reward_model import TextClassificationModel
from evaluate import (
    predict, evaluate_model, generate_evaluation_report, plot_metrics,
    calculate_metrics, generate_class_distributions
)
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import logging
from torchviz import make_dot
import torch.nn as nn
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义类别映射（使用README中的信息）
CLASS_NAMES = [
    "民生",     # 100 news_story
    "文化",     # 101 news_culture
    "娱乐",     # 102 news_entertainment
    "体育",     # 103 news_sports
    "财经",     # 104 news_finance
    "房产",     # 106 news_house
    "汽车",     # 107 news_car
    "教育",     # 108 news_edu
    "科技",     # 109 news_tech
    "军事",     # 110 news_military
    "旅游",     # 112 news_travel
    "国际",     # 113 news_world
    "证券",     # 114 stock
    "农业",     # 115 news_agriculture
    "电竞",     # 116 news_game
]

def parse_args():
    parser = argparse.ArgumentParser(description='Test a trained text classification model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--test_file', type=str, default='data/test/test.txt', help='Path to test file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory with model and results')
    parser.add_argument('--predictions_file', type=str, default='predictions.json', help='File to save predictions')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=256, help='Dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes (15 for this dataset)')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length')
    
    # Testing arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for testing')
    parser.add_argument('--model_path', type=str, default='output/best_model.pth', help='Path to saved model')
    parser.add_argument('--f1_model_path', type=str, default='output/best_model_f1.pth', 
                        help='Path to model saved with best F1 score')
    parser.add_argument('--use_f1_model', action='store_true', 
                        help='Use the model saved with best F1 score instead of best accuracy')
    parser.add_argument('--error_analysis', action='store_true', 
                        help='Perform detailed error analysis')
    parser.add_argument('--visualize_roc', action='store_true',
                        help='Generate ROC curves for each class')
    parser.add_argument('--visualize_distributions', action='store_true',
                        help='Generate class distribution visualizations')
    
    return parser.parse_args()

def perform_error_analysis(results, output_dir):
    """
    Perform error analysis to identify classes that the model tends to confuse
    
    Args:
        results: List of prediction results
        output_dir: Output directory
    """
    logger.info("Performing error analysis...")
    
    # 确保CLASS_NAMES长度足够
    max_label = max(max(item['true_label'] for item in results), max(item['predicted_label'] for item in results))
    if max_label >= len(CLASS_NAMES):
        logger.warning(f"Warning: Class index {max_label} exceeds CLASS_NAMES length {len(CLASS_NAMES)}. Using index values as class names.")
        # 扩展CLASS_NAMES以适应所有类别
        extended_class_names = CLASS_NAMES.copy()
        for i in range(len(CLASS_NAMES), max_label + 1):
            extended_class_names.append(f"Class{i}")
        class_names = extended_class_names
    else:
        class_names = CLASS_NAMES
    
    # 构建混淆矩阵数据
    num_classes = max_label + 1
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    for item in results:
        true_label = item['true_label']
        pred_label = item['predicted_label']
        
        # 确保索引在有效范围内
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            confusion[true_label][pred_label] += 1
        else:
            logger.warning(f"Skipping invalid labels: true={true_label}, pred={pred_label}")
    
    # 找出最常见的错误类型
    error_types = []
    for true_class in range(num_classes):
        for pred_class in range(num_classes):
            if true_class != pred_class:
                error_count = confusion[true_class][pred_class]
                if error_count > 0:
                    true_class_name = class_names[true_class] if true_class < len(class_names) else f"Class{true_class}"
                    pred_class_name = class_names[pred_class] if pred_class < len(class_names) else f"Class{pred_class}"
                    
                    error_types.append({
                        'true_class': true_class,
                        'pred_class': pred_class,
                        'count': int(error_count),
                        'true_class_name': true_class_name,
                        'pred_class_name': pred_class_name,
                    })
    
    # 按错误数量排序
    error_types.sort(key=lambda x: x['count'], reverse=True)
    
    # 输出前10个最常见错误
    print("\nMost Common Classification Errors:")
    print("True Class -> Predicted Class: Count")
    for i, error in enumerate(error_types[:10]):
        print(f"{error['true_class_name']} ({error['true_class']}) -> {error['pred_class_name']} ({error['pred_class']}): {error['count']}")
    
    # 找出每个类别最容易混淆的类别
    print("\nMost Confused Classes for Each Category:")
    for true_class in range(num_classes):
        if true_class >= len(class_names):
            continue
            
        errors = []
        for pred_class in range(num_classes):
            if true_class != pred_class and confusion[true_class][pred_class] > 0:
                errors.append((pred_class, confusion[true_class][pred_class]))
        
        if errors:
            errors.sort(key=lambda x: x[1], reverse=True)
            most_confused = errors[0]
            most_confused_class_name = class_names[most_confused[0]] if most_confused[0] < len(class_names) else f"Class{most_confused[0]}"
            print(f"{class_names[true_class]} ({true_class}) most often confused as: {most_confused_class_name} ({most_confused[0]}), count: {most_confused[1]}")
    
    # 分析具有高置信度但预测错误的样本
    high_confidence_errors = []
    for item in results:
        if item['true_label'] != item['predicted_label']:
            if isinstance(item['probabilities'], list) and 0 <= item['predicted_label'] < len(item['probabilities']):
                confidence = item['probabilities'][item['predicted_label']]
                if confidence > 0.8:  # High confidence threshold
                    true_class_name = class_names[item['true_label']] if item['true_label'] < len(class_names) else f"Class{item['true_label']}"
                    pred_class_name = class_names[item['predicted_label']] if item['predicted_label'] < len(class_names) else f"Class{item['predicted_label']}"
                    
                    high_confidence_errors.append({
                        'text': item['text'],
                        'true_label': item['true_label'],
                        'pred_label': item['predicted_label'],
                        'true_class_name': true_class_name,
                        'pred_class_name': pred_class_name,
                        'confidence': float(confidence)
                    })
    
    # 按置信度排序
    high_confidence_errors.sort(key=lambda x: x['confidence'], reverse=True)
    
    # 输出高置信度错误样本
    if high_confidence_errors:
        print("\nHigh Confidence Error Samples:")
        for i, error in enumerate(high_confidence_errors[:5]):
            print(f"\n{i+1}. Text: {error['text'][:100]}..." if len(error['text']) > 100 else f"\n{i+1}. Text: {error['text']}")
            print(f"   True Class: {error['true_class_name']} ({error['true_label']})")
            print(f"   Predicted Class: {error['pred_class_name']} ({error['pred_label']})")
            print(f"   Confidence: {error['confidence']:.4f}")
    
    # 保存错误分析结果
    error_analysis = {
        'confusion_matrix': confusion.tolist(),
        'common_errors': error_types[:20],
        'high_confidence_errors': high_confidence_errors[:20] if high_confidence_errors else []
    }
    
    try:
        with open(os.path.join(output_dir, 'error_analysis.json'), 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"\nError analysis results saved to {os.path.join(output_dir, 'error_analysis.json')}")
    except Exception as e:
        logger.error(f"Error saving error analysis results: {str(e)}")
    
    return error_analysis

def plot_roc_curves(metrics, output_path):
    """
    Plot ROC curves for all classes
    
    Args:
        metrics: Dictionary containing roc_curves information
        output_path: Path to save the plot
    """
    try:
        if 'roc_curves' not in metrics:
            logger.warning("ROC curve data not found, cannot plot ROC curves")
            return
        
        roc_curves = metrics['roc_curves']
        if not roc_curves:
            logger.warning("ROC curve data is empty, cannot plot")
            return
        
        n_classes = len(roc_curves)
        plt.figure(figsize=(10, 8))
        
        for class_name, roc_data in roc_curves.items():
            if 'fpr' in roc_data and 'tpr' in roc_data and len(roc_data['fpr']) > 0:
                plt.plot(
                    roc_data['fpr'], 
                    roc_data['tpr'], 
                    lw=2,
                    label=f'{class_name} (AUC = {roc_data["auc"]:.3f})'
                )
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Classes')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"ROC curves saved to {output_path}")
    except Exception as e:
        logger.error(f"Error plotting ROC curves: {str(e)}")

def visualize_model_architecture(model, input_size, output_dir):
    """
    Visualize the model architecture using torchviz
    
    Args:
        model: The PyTorch model
        input_size: Input tensor size
        output_dir: Directory to save the visualization
    """
    try:
        # Create dummy input
        x = torch.randn(input_size).to(next(model.parameters()).device)
        
        # Forward pass
        y = model(x)
        
        # Create dot graph
        dot = make_dot(y, params=dict(model.named_parameters()))
        
        # Save visualization
        viz_path = os.path.join(output_dir, 'model_architecture.png')
        dot.render(viz_path, format='png')
        print(f"Model architecture visualization saved to {viz_path}")
        
        # Generate and save model summary
        summary_path = os.path.join(output_dir, 'model_summary.txt')
        with open(summary_path, 'w') as f:
            f.write('Model Architecture Summary:\n\n')
            f.write('Layer Structure:\n')
            for name, module in model.named_children():
                f.write(f'\n{name}:\n')
                f.write(str(module))
                f.write('\n')
            
            f.write('\nParameter Count:\n')
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            f.write(f'Total parameters: {total_params:,}\n')
            f.write(f'Trainable parameters: {trainable_params:,}\n')
            
        print(f"Model summary saved to {summary_path}")
        
    except Exception as e:
        logger.error(f"Error visualizing model architecture: {str(e)}")

def main():
    try:
        args = parse_args()
        device = torch.device(args.device)
        
        # 创建输出目录
        os.makedirs(args.output_dir, exist_ok=True)
        
        # 加载词汇表
        vocab_path = os.path.join(args.output_dir, 'vocab.pkl')
        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                vectorizer = pickle.load(f)
        else:
            raise ValueError(f"Vocabulary file not found: {vocab_path}")
        
        # 初始化数据处理器
        processor = DataProcessor(
            data_dir=args.data_dir,
            max_seq_len=args.max_seq_len,
        )
        processor.vectorizer = vectorizer
        
        # 加载测试数据集
        logger.info("Loading test dataset...")
        try:
            test_dataset = processor.test_dataset if hasattr(processor, 'test_dataset') and processor.test_dataset else \
                processor.load_datasets()[1]
        except Exception as e:
            logger.error(f"Failed to load test dataset: {str(e)}")
            raise
        
        # 创建测试数据加载器
        logger.info("Creating data loader...")
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            collate_fn=lambda batch: processor.preprocess_batch(
                {'tokens': [item['tokens'] for item in batch], 
                 'labels': torch.tensor([item['label'] for item in batch])},
                device=args.device
            )
        )
        
        # 模型配置
        model_config = {
            'vocab_size': len(vectorizer.word2idx),
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_classes': args.num_classes,
            'num_heads': args.num_heads,
            'num_layers': args.num_layers,
            'max_seq_len': args.max_seq_len
        }
        
        # 初始化模型
        logger.info("Initializing model...")
        model = TextClassificationModel(model_config)
        
        # 选择要加载的模型路径
        model_path = args.f1_model_path if args.use_f1_model and os.path.exists(args.f1_model_path) else args.model_path
        
        # 加载模型权重
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Model loaded from {model_path}")
        else:
            raise ValueError(f"Model file not found: {model_path}")
        
        model.to(device)
        model.eval()
        
        # Visualize model architecture
        logger.info("Generating model architecture visualization...")
        try:
            # 创建示例输入
            batch_size = args.batch_size
            seq_length = args.max_seq_len
            example_input = {
                'input_ids': torch.randint(0, len(vectorizer.word2idx), (batch_size, seq_length)).to(device),
                'attention_mask': torch.ones(batch_size, seq_length).to(device),
                'positions': torch.arange(seq_length).unsqueeze(0).repeat(batch_size, 1).to(device),
                'labels': torch.randint(0, args.num_classes, (batch_size,)).to(device)
            }
            
            # 使用示例输入进行一次前向传播并获取损失
            outputs = model(**example_input)
            if isinstance(outputs, dict):
                # 如果输出是字典，使用其中的一个张量值
                viz_tensor = outputs['loss'] if 'loss' in outputs else list(outputs.values())[0]
            else:
                viz_tensor = outputs
            
            # 生成模型摘要（这部分不依赖Graphviz）
            summary_path = os.path.join(args.output_dir, 'model_summary.txt')
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write('Model Architecture Summary:\n\n')
                f.write('Layer Structure:\n')
                for name, module in model.named_children():
                    f.write(f'\n{name}:\n')
                    f.write(str(module))
                    f.write('\n')
                
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                f.write(f'\nTotal parameters: {total_params:,}\n')
                f.write(f'Trainable parameters: {trainable_params:,}\n')
            logger.info(f"Model summary saved to {summary_path}")
            
            try:
                # 尝试使用torchviz生成模型图（这部分依赖Graphviz）
                viz_path = os.path.join(args.output_dir, 'model_architecture')
                dot = make_dot(viz_tensor, params=dict(model.named_parameters()))
                dot.render(viz_path, format='png')
                logger.info(f"Model architecture visualization saved to {viz_path}.png")
            except Exception as viz_error:
                if "ExecutableNotFound" in str(viz_error):
                    logger.warning("Graphviz not installed. Skipping visualization. To enable visualization, please install Graphviz.")
                else:
                    logger.warning(f"Failed to generate visualization: {str(viz_error)}")
                logger.debug(traceback.format_exc())
            
        except Exception as e:
            logger.error(f"Error in model analysis: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 评估模型
        logger.info("Evaluating model...")
        test_metrics = evaluate_model(
            model, 
            test_dataloader, 
            device,
            class_names=[str(i) for i in range(len(CLASS_NAMES))],
            output_path=os.path.join(args.output_dir, 'test_metrics.png')
        )
        
        # 生成评估报告
        logger.info("Generating evaluation report...")
        report = generate_evaluation_report(
            test_metrics, 
            output_path=os.path.join(args.output_dir, 'test_report.md')
        )
        
        # 打印主要评估指标
        print("\nTest Set Evaluation Results:")
        print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  - F1 (Micro): {test_metrics['f1']['micro']:.4f}")
        print(f"  - F1 (Macro): {test_metrics['f1']['macro']:.4f}")
        print(f"  - F1 (Weighted): {test_metrics['f1']['weighted']:.4f}")
        
        # 打印每个类别的F1得分
        print("\nF1 Scores by Class:")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_key = str(class_idx)
            if class_key in test_metrics['f1']['per_class']:
                f1 = test_metrics['f1']['per_class'][class_key]
                precision = test_metrics['precision']['per_class'][class_key]
                recall = test_metrics['recall']['per_class'][class_key]
                support = test_metrics['support'][class_key]
                print(f"  - {class_name} (Class {class_idx}): F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, Samples={support}")
        
        # 如果有Actor-Critic指标，也打印出来
        if 'actor_critic' in test_metrics:
            ac_metrics = test_metrics['actor_critic']
            print("\nActor-Critic Evaluation Metrics:")
            print(f"  - Reward Prediction MSE: {ac_metrics['reward_mse']:.4f}")
            print(f"  - Reward Prediction Correlation: {ac_metrics['reward_correlation']:.4f}")
            print(f"  - Average Prediction Confidence: {ac_metrics['avg_confidence']:.4f}")
            print(f"  - Average Confidence (Correct): {ac_metrics['avg_confidence_correct']:.4f}")
            print(f"  - Average Confidence (Incorrect): {ac_metrics['avg_confidence_incorrect']:.4f}")
            print(f"  - Confidence Gap: {ac_metrics['confidence_gap']:.4f}")
        
        # 进行预测
        logger.info("Generating prediction results...")
        try:
            predictions, probabilities, rewards = predict(model, test_dataloader, device)
            has_rewards = True
        except Exception as e:
            logger.info(f"Reward prediction not supported, switching to standard prediction: {str(e)}")
            predictions, probabilities = predict(model, test_dataloader, device)
            rewards = None
            has_rewards = False
        
        # 确保预测结果和概率是numpy数组
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)
        
        # 获取原始文本和标签
        texts = []
        labels = []
        
        for item in test_dataset:
            if 'text' in item:
                texts.append(item['text'])
            else:
                # 如果没有原始文本，使用tokens的字符串表示作为替代
                texts.append(' '.join(item['tokens']) if 'tokens' in item else "")
            
            labels.append(item['label'])
        
        # 确保标签是numpy数组
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        
        # 创建结果字典
        results = []
        for i in range(len(texts)):
            if i >= len(predictions) or i >= len(probabilities):
                logger.warning(f"Index {i} out of prediction result range, skipping")
                continue
                
            # 确保索引在有效范围内
            if i < len(labels) and i < len(predictions) and i < len(probabilities):
                # 确保probabilities[i]是一维数组
                probs = probabilities[i]
                if len(probs.shape) > 1:
                    logger.warning(f"Prediction probability {i} not one-dimensional, shape: {probs.shape}")
                    probs = probs.flatten()
                
                result = {
                    'text': texts[i],
                    'true_label': int(labels[i]),
                    'predicted_label': int(predictions[i]),
                    'probabilities': probs.tolist() if isinstance(probs, np.ndarray) else probs,
                    'true_class': CLASS_NAMES[int(labels[i])] if int(labels[i]) < len(CLASS_NAMES) else f"Class{int(labels[i])}",
                    'predicted_class': CLASS_NAMES[int(predictions[i])] if int(predictions[i]) < len(CLASS_NAMES) else f"Class{int(predictions[i])}",
                    'is_correct': int(predictions[i]) == int(labels[i])
                }
                
                if has_rewards and rewards is not None and i < len(rewards):
                    result['predicted_reward'] = float(rewards[i])
                
                results.append(result)
        
        # 保存预测结果
        predictions_path = os.path.join(args.output_dir, args.predictions_file)
        with open(predictions_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Prediction results saved to {predictions_path}")
        
        # 计算准确率
        correct = sum(1 for i in range(len(labels)) if i < len(predictions) and predictions[i] == labels[i])
        accuracy = correct / len(labels)
        print(f"Test Accuracy: {accuracy:.4f} ({correct}/{len(labels)})")
        
        # 计算更详细的评估指标
        logger.info("Calculating detailed metrics...")
        try:
            y_true = labels.astype(np.int64)
            y_pred = predictions.astype(np.int64)
            
            # 确保概率数组形状正确
            if len(probabilities.shape) == 1:
                # 如果是一维的，重塑为二维，假设只有一个类别
                logger.warning("Probability array is one-dimensional, reshaping to two-dimensional")
                y_probs = np.zeros((len(y_pred), max(2, np.max(y_pred)+1)))
                for i, pred in enumerate(y_pred):
                    y_probs[i, pred] = probabilities[i]
            else:
                y_probs = probabilities
            
            detailed_metrics = calculate_metrics(
                y_true, 
                y_pred, 
                y_probs, 
                class_names=CLASS_NAMES
            )
            
            # 保存详细指标
            with open(os.path.join(args.output_dir, 'detailed_metrics.json'), 'w', encoding='utf-8') as f:
                json.dump(detailed_metrics, f, ensure_ascii=False, indent=2)
            
            # 打印置信度指标
            if 'confidence_metrics' in detailed_metrics:
                conf_metrics = detailed_metrics['confidence_metrics']
                print("\nConfidence Analysis:")
                print(f"  - Average Prediction Confidence: {conf_metrics['avg_confidence']:.4f}")
                print(f"  - Average Confidence (Correct): {conf_metrics['avg_confidence_correct']:.4f}")
                print(f"  - Average Confidence (Incorrect): {conf_metrics['avg_confidence_incorrect']:.4f}")
                print(f"  - Confidence Gap: {conf_metrics['confidence_gap']:.4f}")
            
            # 可视化ROC曲线
            if args.visualize_roc and 'roc_curves' in detailed_metrics:
                roc_path = os.path.join(args.output_dir, 'roc_curves.png')
                plot_roc_curves(detailed_metrics, roc_path)
            
            # 可视化类别分布
            if args.visualize_distributions:
                dist_path = os.path.join(args.output_dir, 'class_distributions.png')
                generate_class_distributions(
                    y_true, 
                    y_pred, 
                    CLASS_NAMES,
                    dist_path
                )
        except Exception as e:
            logger.error(f"Error calculating detailed metrics: {str(e)}")
            logger.error(traceback.format_exc())
        
        # 如果需要，执行错误分析
        if args.error_analysis:
            error_analysis = perform_error_analysis(results, args.output_dir)

        logger.info("Test completed!")
        
    except Exception as e:
        logger.error(f"Error executing: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
