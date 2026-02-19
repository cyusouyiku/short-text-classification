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
    Compute evaluation metrics, including overall and per-class metrics.
    
    Args:
        predictions: Model predicted classes
        labels: Ground truth labels
        class_names: List of class names, default None uses numeric indices
        
    Returns:
        Dictionary containing various evaluation metrics
    """
    # Convert to numpy arrays for evaluation
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    # Use numeric indices if class names not provided
    if class_names is None:
        class_names = [str(i) for i in range(max(max(predictions), max(labels)) + 1)]
    
    # Compute overall metrics
    accuracy = accuracy_score(labels, predictions)
    
    # Compute per-class metrics
    class_report = classification_report(labels, predictions, output_dict=True)
    
    # Compute precision, recall and F1 for each class
    per_class_metrics = {}
    precision_per_class, recall_per_class, f1_per_class, support_per_class = \
        precision_recall_fscore_support(labels, predictions, average=None)
    
    # Ensure all classes are included in per_class_metrics
    total_support = 0
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            per_class_metrics[class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1-score': float(f1_per_class[i]),
                'support': int(support_per_class[i])
            }
            total_support += support_per_class[i]
    
    # Manually compute macro and weighted metrics
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    total_support = max(total_support, 1)  # Avoid division by zero
    weighted_precision = np.sum(precision_per_class * support_per_class) / total_support
    weighted_recall = np.sum(recall_per_class * support_per_class) / total_support
    weighted_f1 = np.sum(f1_per_class * support_per_class) / total_support
    
    # Build metrics dictionary to return
    metrics = {
        'accuracy': float(accuracy),
        'precision': {
            'micro': float(accuracy),  # In multiclass, micro precision equals accuracy
            'macro': float(macro_precision),
            'weighted': float(weighted_precision),
            'per_class': {class_name: per_class_metrics[class_name]['precision'] 
                          for class_name in class_names if class_name in per_class_metrics}
        },
        'recall': {
            'micro': float(accuracy),  # In multiclass, micro recall equals accuracy
            'macro': float(macro_recall),
            'weighted': float(weighted_recall),
            'per_class': {class_name: per_class_metrics[class_name]['recall'] 
                          for class_name in class_names if class_name in per_class_metrics}
        },
        'f1': {
            'micro': float(accuracy),  # In multiclass, micro F1 equals accuracy
            'macro': float(macro_f1),
            'weighted': float(weighted_f1),
            'per_class': {class_name: per_class_metrics[class_name]['f1-score'] 
                          for class_name in class_names if class_name in per_class_metrics}
        },
        'support': {class_name: per_class_metrics[class_name]['support'] 
                    for class_name in class_names if class_name in per_class_metrics},
        'confusion_matrix': confusion_matrix(labels, predictions).tolist(),
        'class_report': class_report
    }
    
    # Add Actor-Critic specific metrics (computed in evaluate_model)
    
    return metrics

def plot_metrics(metrics, output_path=None):
    """
    Visualize evaluation metrics.
    
    Args:
        metrics: Evaluation metrics dictionary
        output_path: Path to save the figure, or None to display
    """
    # Create 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues')
    axes[0, 0].set_title('Confusion Matrix', fontsize=14)
    axes[0, 0].set_xlabel('Predicted Label', fontsize=12)
    axes[0, 0].set_ylabel('True Label', fontsize=12)
    
    # Get F1 scores for all classes from class_report
    f1_per_class = {}
    for class_name, class_metrics in metrics['class_report'].items():
        if isinstance(class_metrics, dict) and 'f1-score' in class_metrics:
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                f1_per_class[class_name] = class_metrics['f1-score']
    
    # Sort by F1 score
    sorted_items = sorted(f1_per_class.items(), key=lambda x: x[1])
    sorted_class_names = [item[0] for item in sorted_items]
    sorted_f1_scores = [item[1] for item in sorted_items]
    
    # Plot F1 score bar chart
    axes[0, 1].barh(sorted_class_names, sorted_f1_scores, color='skyblue')
    axes[0, 1].set_title('F1 Score per Class', fontsize=14)
    axes[0, 1].set_xlabel('F1 Score', fontsize=12)
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].grid(axis='x', linestyle='--', alpha=0.6)
    
    # Plot precision vs recall comparison
    class_names = sorted_class_names
    precision_per_class = [metrics['class_report'][name]['precision'] for name in class_names]
    recall_per_class = [metrics['class_report'][name]['recall'] for name in class_names]
    
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
    
    # Plot support bar chart
    support = [metrics['class_report'][name]['support'] for name in class_names]
    axes[1, 1].bar(class_names, support, color='lightcoral')
    axes[1, 1].set_title('Support per Class', fontsize=14)
    axes[1, 1].set_xlabel('Class', fontsize=12)
    axes[1, 1].set_ylabel('Number of Samples', fontsize=12)
    axes[1, 1].set_xticklabels(class_names, rotation=45)
    axes[1, 1].grid(axis='y', linestyle='--', alpha=0.6)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save or display figure
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()

def evaluate_model(model, dataloader, device, class_names=None, output_path=None, compute_rl_metrics=True):
    """
    Evaluate model performance and compute various metrics.
    
    Args:
        model: Model to evaluate
        dataloader: Data loader
        device: Device (CPU/GPU)
        class_names: List of class names
        output_path: Output path for saving visualizations
        compute_rl_metrics: Whether to compute reinforcement learning metrics
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_policy_probs = []
    all_rewards = []
    all_true_rewards = []
    
    # Disable gradient computation for faster inference
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            positions = batch['positions'].to(device) if 'positions' in batch else None
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask, positions)
            logits = outputs['logits']
            policy_probs = outputs['policy_probs']
            
            # Get predictions
            predictions = torch.argmax(logits, dim=-1)
            
            # Get reward predictions if computing RL metrics
            if compute_rl_metrics and 'predicted_reward' in outputs:
                predicted_rewards = outputs['predicted_reward']
                # Compute true reward (1 if correct, 0 if wrong)
                true_rewards = (predictions == labels).float()
                
                all_rewards.extend(predicted_rewards.cpu().numpy())
                all_true_rewards.extend(true_rewards.cpu().numpy())
                all_policy_probs.extend(policy_probs.cpu().numpy())
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute standard classification metrics
    metrics = compute_metrics(all_predictions, all_labels, class_names)
    
    # Compute reinforcement learning metrics if requested
    if compute_rl_metrics and all_rewards:
        # Convert to numpy arrays
        all_rewards = np.array(all_rewards)
        all_true_rewards = np.array(all_true_rewards)
        all_policy_probs = np.array(all_policy_probs)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        # Compute reward prediction MSE
        reward_mse = ((all_rewards - all_true_rewards) ** 2).mean()
        
        # Compute average prediction confidence (by correct/incorrect groups)
        correct_indices = (all_predictions == all_labels)
        incorrect_indices = ~correct_indices
        
        # Compute confidence for each prediction's predicted class
        prediction_confidences = np.array([all_policy_probs[i, pred] for i, pred in enumerate(all_predictions)])
        
        # Compute average confidence for correct and incorrect predictions
        avg_confidence_correct = prediction_confidences[correct_indices].mean() if any(correct_indices) else 0
        avg_confidence_incorrect = prediction_confidences[incorrect_indices].mean() if any(incorrect_indices) else 0
        
        # Add to metrics dictionary
        metrics['actor_critic'] = {
            'reward_mse': float(reward_mse),
            'avg_confidence': float(prediction_confidences.mean()),
            'avg_confidence_correct': float(avg_confidence_correct),
            'avg_confidence_incorrect': float(avg_confidence_incorrect),
            'confidence_gap': float(avg_confidence_correct - avg_confidence_incorrect)
        }
        
        # Compute reward prediction correlation
        correlation = np.corrcoef(all_rewards, all_true_rewards)[0, 1]
        metrics['actor_critic']['reward_correlation'] = float(correlation)
    
    # Visualize results
    if output_path:
        plot_metrics(metrics, output_path)
    
    return metrics

def predict(model, dataloader, device):
    """
    Use model to predict on the dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device (CPU/GPU)
        
    Returns:
        predictions: List of predicted labels
        probabilities: List of prediction probabilities
        rewards: Predicted reward values if model supports it
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_rewards = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Skip if batch is None or empty
            if batch is None:
                continue
                
            try:
                # Ensure batch is a dictionary
                if not isinstance(batch, dict):
                    print(f"Batch is not a dict: {type(batch)}")
                    continue
                
                # Ensure required keys exist
                if 'input_ids' not in batch:
                    print(f"Batch missing input_ids field, available keys: {batch.keys()}")
                    continue
                
                # Forward pass
                if hasattr(model, 'predict_with_rewards'):
                    # Unpack batch dictionary
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    positions = batch.get('positions')
                    if positions is not None:
                        positions = positions.to(device)
                    
                    logits, rewards = model.predict_with_rewards(input_ids, attention_mask, positions)
                    all_rewards.extend(rewards.cpu().numpy())
                else:
                    # Unpack batch dictionary
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    positions = batch.get('positions')
                    if positions is not None:
                        positions = positions.to(device)
                    
                    # Call model with unpacked parameters
                    outputs = model(input_ids, attention_mask, positions)
                    # Extract logits from output dict
                    if isinstance(outputs, dict) and 'logits' in outputs:
                        logits = outputs['logits']
                        if 'predicted_reward' in outputs:
                            all_rewards.extend(outputs['predicted_reward'].cpu().numpy())
                    else:
                        logits = outputs  # If output is not dict, assume it is logits
                
                # Convert logits to probabilities
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                predictions = np.argmax(probs, axis=1)
                
                all_predictions.extend(predictions)
                all_probabilities.extend(probs)
            except Exception as e:
                print(f"Batch processing error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    if len(all_rewards) > 0:
        return all_predictions, all_probabilities, all_rewards
    else:
        return all_predictions, all_probabilities

def generate_evaluation_report(metrics, output_path=None):
    """
    Generate evaluation report as Markdown document.
    
    Args:
        metrics: Evaluation metrics dictionary
        output_path: Path to save the report
        
    Returns:
        report_md: Report in Markdown format
    """
    report_md = "# Model Evaluation Report\n\n"
    
    # Main metrics
    report_md += "## Main Evaluation Metrics\n\n"
    report_md += f"- **Accuracy**: {metrics['accuracy']:.4f}\n"
    report_md += f"- **F1 (Micro)**: {metrics['f1']['micro']:.4f}\n"
    report_md += f"- **F1 (Macro)**: {metrics['f1']['macro']:.4f}\n"
    report_md += f"- **F1 (Weighted)**: {metrics['f1']['weighted']:.4f}\n\n"
    
    # Per-class metrics
    report_md += "## Per-Class Metrics\n\n"
    report_md += "| Class | F1 | Precision | Recall | Support |\n"
    report_md += "|------|----|---------|---------|---------|\n"
    
    for class_name in metrics['support'].keys():
        f1 = metrics['f1']['per_class'].get(class_name, 0)
        precision = metrics['precision']['per_class'].get(class_name, 0)
        recall = metrics['recall']['per_class'].get(class_name, 0)
        support = metrics['support'].get(class_name, 0)
        
        report_md += f"| {class_name} | {f1:.4f} | {precision:.4f} | {recall:.4f} | {support} |\n"
    
    report_md += "\n"
    
    # Confidence metrics (if present)
    if 'confidence_metrics' in metrics:
        report_md += "## Confidence Metrics\n\n"
        report_md += f"- **Avg Prediction Confidence**: {metrics['confidence_metrics']['avg_confidence']:.4f}\n"
        report_md += f"- **Avg Confidence (Correct)**: {metrics['confidence_metrics']['avg_confidence_correct']:.4f}\n"
        report_md += f"- **Avg Confidence (Incorrect)**: {metrics['confidence_metrics']['avg_confidence_incorrect']:.4f}\n"
        report_md += f"- **Confidence Gap**: {metrics['confidence_metrics']['confidence_gap']:.4f}\n\n"
    
    # Actor-Critic metrics (if present)
    if 'actor_critic' in metrics:
        report_md += "## Actor-Critic Metrics\n\n"
        report_md += f"- **Reward Prediction MSE**: {metrics['actor_critic']['reward_mse']:.4f}\n"
        report_md += f"- **Reward Prediction Correlation**: {metrics['actor_critic']['reward_correlation']:.4f}\n\n"
    
    # Save report
    if output_path:
        report_dir = os.path.dirname(output_path)
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_md)
        
        print(f"Evaluation report saved to {output_path}")
    
    return report_md

def calculate_metrics(y_true, y_pred, y_probs, class_names=None):
    """
    Compute various classification evaluation metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_probs: Prediction probabilities
        class_names: List of class names
        
    Returns:
        metrics: Dictionary containing various metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    # Use indices if class names not provided
    if class_names is None:
        class_names = [str(i) for i in range(max(max(y_true), max(y_pred)) + 1)]
    
    # Compute basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    # Compute precision, recall, F1 and support per class
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro')
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    
    # Compute per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None)
    
    # Convert per-class metrics to dict format
    precision_dict = {class_names[i]: precision_per_class[i] for i in range(len(class_names)) if i < len(precision_per_class)}
    recall_dict = {class_names[i]: recall_per_class[i] for i in range(len(class_names)) if i < len(recall_per_class)}
    f1_dict = {class_names[i]: f1_per_class[i] for i in range(len(class_names)) if i < len(f1_per_class)}
    support_dict = {class_names[i]: int(support_per_class[i]) for i in range(len(class_names)) if i < len(support_per_class)}
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Compute ROC curve and AUC per class (one-vs-rest)
    roc_curves = {}
    for i in range(len(class_names)):
        if i < y_probs.shape[1]:  # Ensure class index is within prob matrix range
            # Create binary classification: current class vs others
            y_true_binary = (y_true == i).astype(int)
            y_score = y_probs[:, i]
            
            try:
                # Compute ROC curve and AUC
                fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                roc_curves[class_names[i]] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc
                }
            except Exception as e:
                print(f"Error computing ROC curve for class {i}: {e}")
                roc_curves[class_names[i]] = {
                    'fpr': [],
                    'tpr': [],
                    'auc': 0.0
                }
    
    # Compute confidence-related metrics
    confidences = np.max(y_probs, axis=1)
    
    # Compute average confidence for correct and incorrect predictions
    correct_indices = (y_pred == y_true)
    avg_confidence = np.mean(confidences)
    avg_confidence_correct = np.mean(confidences[correct_indices]) if np.any(correct_indices) else 0
    avg_confidence_incorrect = np.mean(confidences[~correct_indices]) if np.any(~correct_indices) else 0
    confidence_gap = avg_confidence_correct - avg_confidence_incorrect
    
    # Integrate all metrics
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
    Generate visualization of class distributions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the chart
    """
    # Compute true and predicted distribution per class
    true_counts = np.bincount(y_true, minlength=len(class_names))
    pred_counts = np.bincount(y_pred, minlength=len(class_names))
    
    # Create DataFrame for visualization
    df = pd.DataFrame({
        'Class': class_names,
        'True Distribution': true_counts,
        'Predicted Distribution': pred_counts
    })
    
    # Convert to long format for plotting
    df_melted = pd.melt(df, id_vars=['Class'], value_vars=['True Distribution', 'Predicted Distribution'], 
                       var_name='Distribution Type', value_name='Sample Count')
    
    # Plot distribution comparison
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Class', y='Sample Count', hue='Distribution Type', data=df_melted)
    
    plt.title('True Distribution vs Predicted Distribution')
    plt.xlabel('Class')
    plt.ylabel('Sample Count')
    
    # Display values above bars
    for container in ax.containers:
        ax.bar_label(container)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Class distribution visualization saved to {output_path}")
