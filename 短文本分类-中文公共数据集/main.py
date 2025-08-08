import os
import torch
import argparse
import json
import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import Adam
from utils import set_seed, TextDataset, create_dataloader
from data_process import DataProcessor
from model.reward_model import TextClassificationModel
from evaluate import evaluate_model, compute_metrics, generate_evaluation_report, plot_metrics
from tqdm import tqdm

# 定义类别映射（使用README中的信息）
CLASS_NAMES = [
    "民生",    # 100
    "文化",    # 101
    "娱乐",    # 102
    "体育",    # 103
    "财经",    # 104
    "房产",    # 106
    "汽车",    # 107
    "教育",    # 108
    "科技",    # 109
    "军事",    # 110
    "旅游",    # 112
    "国际",    # 113
    "证券",    # 114
    "农业",    # 115
    "电竞"     # 116
]

def parse_args():
    parser = argparse.ArgumentParser(description='Train a text classification model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='data', help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save model and results')
    
    # Model arguments
    parser.add_argument('--embedding_dim', type=int, default=256, help='Dimension of word embeddings')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Dimension of hidden layers')
    parser.add_argument('--num_classes', type=int, default=15, help='Number of classes')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout probability')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--max_len', type=int, default=128, help='Maximum sequence length')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.0003, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--max_seq_len', type=int, default=128, help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                        help='Device to use for training')
    parser.add_argument('--eval_steps', type=int, default=0, 
                        help='Evaluate every n steps. If 0, evaluate once per epoch')
    
    return parser.parse_args()

def train(model, train_dataloader, val_dataloader, optimizer, device, num_epochs, output_dir, eval_steps=0):
    """Train the model"""
    # Create output directory if it doesn't exist
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    best_val_acc = 0.0
    best_val_f1_macro = 0.0
    global_step = 0
    total_steps = num_epochs * len(train_dataloader)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        # Progress bar for batches
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            positions = batch['positions'].to(device) if 'positions' in batch else None
            labels = batch['labels'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, attention_mask, positions, labels)
            loss = outputs['loss']
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            global_step += 1
            train_loss += loss.item()
            progress_bar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'step': f"{global_step}/{total_steps}"
            })
            
            # Evaluate if needed
            if eval_steps > 0 and global_step % eval_steps == 0:
                # Evaluate on validation set
                val_metrics = evaluate_model(
                    model, val_dataloader, device, 
                    class_names=CLASS_NAMES,
                    output_path=os.path.join(output_dir, f'eval_step_{global_step}_metrics.png')
                )
                
                print(f"Step {global_step} Validation Metrics:")
                print(f"  - Accuracy: {val_metrics['accuracy']:.4f}")
                print(f"  - F1 (macro): {val_metrics['f1']['macro']:.4f}")
                print(f"  - F1 (weighted): {val_metrics['f1']['weighted']:.4f}")
                
                # Save model if it's the best so far (by F1 macro)
                if val_metrics['f1']['macro'] > best_val_f1_macro:
                    best_val_f1_macro = val_metrics['f1']['macro']
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_f1.pth'))
                    print(f"New best model (F1) saved with validation F1-macro: {best_val_f1_macro:.4f}")
                
                # Also save based on accuracy for backward compatibility
                if val_metrics['accuracy'] > best_val_acc:
                    best_val_acc = val_metrics['accuracy']
                    torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
                    print(f"New best model (Acc) saved with validation accuracy: {best_val_acc:.4f}")
                
                # Switch back to training mode
                model.train()
        
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs} - Avg. Training Loss: {avg_train_loss:.4f}")
        
        # Evaluate after each epoch
        print("\nEvaluating on validation set...")
        val_metrics = evaluate_model(
            model, val_dataloader, device, 
            class_names=CLASS_NAMES,
            output_path=os.path.join(output_dir, f'epoch_{epoch+1}_metrics.png')
        )
        
        # Print key metrics
        print(f"Epoch {epoch+1} Validation Metrics:")
        print(f"  - Accuracy: {val_metrics['accuracy']:.4f}")
        print(f"  - F1 (macro): {val_metrics['f1']['macro']:.4f}")
        print(f"  - F1 (weighted): {val_metrics['f1']['weighted']:.4f}")
        
        # Print F1 scores for each class
        print("\nF1 scores per class:")
        for class_idx, class_name in enumerate(CLASS_NAMES):
            if str(class_idx) in val_metrics['f1']['per_class']:
                f1 = val_metrics['f1']['per_class'][str(class_idx)]
                print(f"  - {class_name} (Class {class_idx}): {f1:.4f}")
        
        # Print Actor-Critic specific metrics if available
        if 'actor_critic' in val_metrics:
            ac_metrics = val_metrics['actor_critic']
            print("\nActor-Critic Metrics:")
            print(f"  - Reward MSE: {ac_metrics['reward_mse']:.4f}")
            print(f"  - Reward Correlation: {ac_metrics['reward_correlation']:.4f}")
            print(f"  - Confidence Gap (correct vs incorrect): {ac_metrics['confidence_gap']:.4f}")
        
        # Save detailed metrics to file
        epoch_metrics_path = os.path.join(output_dir, f'epoch_{epoch+1}_metrics.json')
        with open(epoch_metrics_path, 'w') as f:
            json.dump(val_metrics, f, indent=2)
        
        # Generate and save evaluation report
        report = generate_evaluation_report(
            val_metrics, 
            output_path=os.path.join(output_dir, f'epoch_{epoch+1}_report.md')
        )
        
        # Save model if it's the best so far (by F1 macro)
        if val_metrics['f1']['macro'] > best_val_f1_macro:
            best_val_f1_macro = val_metrics['f1']['macro']
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_f1.pth'))
            print(f"New best model (F1) saved with validation F1-macro: {best_val_f1_macro:.4f}")
        
        # Also save based on accuracy for backward compatibility
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"New best model (Acc) saved with validation accuracy: {best_val_acc:.4f}")
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation F1-macro: {best_val_f1_macro:.4f}")
    
    return model

def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load data
    processor = DataProcessor(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
    )
    
    train_dataset, test_dataset = processor.load_datasets()
    vectorizer = processor.build_vocab(save_path=os.path.join(args.output_dir, 'vocab.pkl'))
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=lambda batch: processor.preprocess_batch(
            {'tokens': [item['tokens'] for item in batch], 
             'labels': torch.tensor([item['label'] for item in batch])},
            device=args.device
        )
    )
    
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
    
    # Model configuration
    model_config = {
        'vocab_size': len(vectorizer.word2idx),
        'embedding_dim': args.embedding_dim,
        'hidden_dim': args.hidden_dim,
        'num_classes': args.num_classes,
        'dropout': args.dropout,
        'num_heads': args.num_heads,
        'num_layers': args.num_layers,
        'max_seq_len': args.max_len
    }
    
    # Initialize model
    model = TextClassificationModel(model_config)
    model.to(args.device)
    
    # Print model architecture
    print(model)
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")
    
    # Initialize optimizer
    optimizer = Adam(model.parameters(), lr=args.lr)
    
    # Train model
    model = train(
        model, 
        train_dataloader, 
        test_dataloader, 
        optimizer, 
        args.device, 
        args.num_epochs, 
        args.output_dir,
        args.eval_steps
    )
    
    # Final evaluation on test set
    print("\nPerforming final evaluation on test set...")
    test_metrics = evaluate_model(
        model, 
        test_dataloader, 
        args.device,
        class_names=CLASS_NAMES,
        output_path=os.path.join(args.output_dir, 'final_test_metrics.png')
    )
    
    # Print key metrics
    print("\nFinal Test Metrics:")
    print(f"  - Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  - F1 (macro): {test_metrics['f1']['macro']:.4f}")
    print(f"  - F1 (weighted): {test_metrics['f1']['weighted']:.4f}")
    
    # Generate and save final test report
    final_report = generate_evaluation_report(
        test_metrics, 
        output_path=os.path.join(args.output_dir, 'final_test_report.md')
    )
    
    # Save test metrics
    with open(os.path.join(args.output_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    print("\nDone! Final evaluation results saved to output directory.")

if __name__ == "__main__":
    main()
