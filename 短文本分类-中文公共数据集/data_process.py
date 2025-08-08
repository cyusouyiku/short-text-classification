import os
import torch
import pickle
import numpy as np
from utils import TextDataset, Vectorizer, set_seed
import torch.nn as nn

class DataProcessor:
    def __init__(self, data_dir='data', max_seq_len=128, vocab_size=50000, min_freq=3):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vectorizer = Vectorizer(vocab_size=vocab_size, min_freq=min_freq)
        
        # Initialize datasets and vectorizers
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
    
    def load_datasets(self):
        """Load datasets from files"""
        train_path = ('/Users/zhangzongyu/Desktop/短文本分类/短文本分类-中文公共数据集/data/train.txt')
        test_path = ('/Users/zhangzongyu/Desktop/短文本分类/短文本分类-中文公共数据集/data/test.txt')
        
        print(f"Loading training data from {train_path}")
        self.train_dataset = TextDataset(train_path, max_length=self.max_seq_len)
        
        print(f"Loading test data from {test_path}")
        self.test_dataset = TextDataset(test_path, max_length=self.max_seq_len)
        
        print(f"Train size: {len(self.train_dataset)}, Test size: {len(self.test_dataset)}")
        
        return self.train_dataset, self.test_dataset
    
    def build_vocab(self, save_path=None):
        """Build vocabulary from the training dataset"""
        if self.train_dataset is None:
            self.load_datasets()
            
        tokens_list = [item['tokens'] for item in self.train_dataset]
        print(f"Building vocabulary from {len(tokens_list)} examples")
        
        self.vectorizer.fit(tokens_list)
        vocab_size = len(self.vectorizer.word2idx)
        print(f"Vocabulary size: {vocab_size}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"Saved vocabulary to {save_path}")
        
        return self.vectorizer
    
    def preprocess_batch(self, batch, device='cpu'):
        """Process a batch of data for model input"""
        texts = batch['tokens']
        labels = batch['labels']
        
        # Convert tokens to indices
        input_ids = self.vectorizer.transform(texts, self.max_seq_len)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != 0).float()
        
        # Create positional indices
        positions = torch.arange(0, input_ids.size(1)).unsqueeze(0).repeat(input_ids.size(0), 1)
        
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'positions': positions.to(device),
            'labels': labels.to(device)
        }

def main():
    set_seed(42)
    
    # Initialize data processor
    processor = DataProcessor()
    
    # Load datasets
    train_dataset, test_dataset = processor.load_datasets()
    
    # Build vocabulary
    vectorizer = processor.build_vocab(save_path='data/vocab.pkl')
    
    print("Data processing complete")

if __name__ == "__main__":
    main()
