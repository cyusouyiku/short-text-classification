"""
Data processing with RoBERTa tokenization.
Per paper: "For RoBERTa embeddings we used the pretrained tokenizer with the same length and padding."
"""
import os
import torch
import pickle
import numpy as np
from utils import TextDataset, Vectorizer, set_seed


class DataProcessor:
    def __init__(self, data_dir='data', max_seq_len=128, vocab_size=50000, min_freq=3,
                 roberta_model='hfl/chinese-roberta-wwm-ext'):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.vectorizer = Vectorizer(vocab_size=vocab_size, min_freq=min_freq)
        self.roberta_model = roberta_model
        self.tokenizer = None  # Lazy load when needed
        
        self.train_dataset = None
        self.test_dataset = None
        self.val_dataset = None
    
    def _get_tokenizer(self):
        """Lazy load RoBERTa tokenizer"""
        if self.tokenizer is None:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.roberta_model)
        return self.tokenizer
    
    def load_datasets(self):
        """Load datasets from files"""
        train_path = os.path.join(self.data_dir, 'train.txt')
        test_path = os.path.join(self.data_dir, 'test.txt')
        if not os.path.exists(train_path):
            train_path = os.path.join(os.path.dirname(__file__), 'data', 'train.txt')
        if not os.path.exists(test_path):
            test_path = os.path.join(os.path.dirname(__file__), 'data', 'test.txt')
        
        print(f"Loading training data from {train_path}")
        self.train_dataset = TextDataset(train_path, max_length=self.max_seq_len)
        
        print(f"Loading test data from {test_path}")
        self.test_dataset = TextDataset(test_path, max_length=self.max_seq_len)
        
        print(f"Train size: {len(self.train_dataset)}, Test size: {len(self.test_dataset)}")
        return self.train_dataset, self.test_dataset
    
    def build_vocab(self, save_path=None):
        """Build vocabulary from training dataset (for Word2Vec loading if needed)"""
        if self.train_dataset is None:
            self.load_datasets()
            
        tokens_list = [item['tokens'] for item in self.train_dataset]
        print(f"Building vocabulary from {len(tokens_list)} examples")
        self.vectorizer.fit(tokens_list)
        print(f"Vocabulary size: {len(self.vectorizer.word2idx)}")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            print(f"Saved vocabulary to {save_path}")
        
        return self.vectorizer
    
    def preprocess_batch(self, batch, device='cpu'):
        """
        Process batch using RoBERTa tokenizer.
        Expects batch with 'texts' (raw strings) and 'labels'.
        """
        texts = batch['texts']
        labels = batch['labels']
        
        tokenizer = self._get_tokenizer()
        encodings = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt"
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask'].float()
        
        return {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'labels': labels.to(device)
        }


def main():
    set_seed(42)
    processor = DataProcessor()
    train_dataset, test_dataset = processor.load_datasets()
    vectorizer = processor.build_vocab(save_path='data/vocab.pkl')
    print("Data processing complete")


if __name__ == "__main__":
    main()
