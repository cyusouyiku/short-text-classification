import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
import jieba
import logging

jieba.setLogLevel(logging.INFO)

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer=None, max_length=128):
        self.texts = []
        self.labels = []
        self.max_length = max_length
        self.tokenizer = tokenizer if tokenizer else jieba.lcut
        
        # Define label mapping dictionary
        self.label_map = {
            100: 0,  # Civil/Livelihood
            101: 1,  # Culture
            102: 2,  # Entertainment
            103: 3,  # Sports
            104: 4,  # Finance
            106: 5,  # Real Estate
            107: 6,  # Automotive
            108: 7,  # Education
            109: 8,  # Technology
            110: 9,  # Military
            112: 10, # Travel
            113: 11, # International
            114: 12, # Securities
            115: 13, # Agriculture
            116: 14  # Esports
        }
        
        print(f"Loading data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Remove extra spaces and split
                parts = [p.strip() for p in line.strip().split('\t')]
                
                # Ensure at least 2 fields (label and title)
                if len(parts) >= 2:
                    try:
                        label = int(parts[0])
                        title = parts[2] if len(parts) > 2 else parts[1]
                        keywords = parts[3] if len(parts) > 3 else ""
                        
                        # Map original label to new index
                        mapped_label = self.label_map.get(label)
                        if mapped_label is not None:  # Only add labels in the mapping
                            self.labels.append(mapped_label)
                            # Combine title and keywords as text features
                            combined_text = title
                            if keywords.strip():  # Add keywords to text if not empty
                                combined_text += ' ' + keywords.replace(',', ' ')
                            self.texts.append(combined_text)
                    except ValueError as e:
                        print(f"Warning: Skipping invalid line: {line.strip()}")
                        continue
        
        print(f"Loaded {len(self.texts)} valid examples")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Tokenize text
        tokens = self.tokenizer(text)
        
        return {
            'text': text,
            'tokens': tokens[:self.max_length],
            'label': label
        }

class Vectorizer:
    def __init__(self, vocab_size=50000, min_freq=3):
        self.vocab_size = vocab_size
        self.min_freq = min_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_freq = {}
        self.embeddings = None
    
    def fit(self, texts):
        """Build vocabulary from texts"""
        # Count word frequencies
        for text in texts:
            for word in text:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1
        
        # Filter by frequency and limit vocab size
        words = [word for word, freq in self.word_freq.items() if freq >= self.min_freq]
        words = sorted(words, key=lambda x: self.word_freq[x], reverse=True)
        words = words[:self.vocab_size - 2]  # -2 for PAD and UNK
        
        # Build word to index mapping
        for word in words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
    
    def transform(self, texts, max_length=None):
        """Convert text to indices"""
        if max_length is None:
            max_length = max(len(text) for text in texts)
        
        indices = []
        for text in texts:
            text_indices = [self.word2idx.get(word, 1) for word in text]  # 1 for UNK
            if len(text_indices) < max_length:
                text_indices += [0] * (max_length - len(text_indices))  # Pad
            else:
                text_indices = text_indices[:max_length]  # Truncate
            indices.append(text_indices)
        
        return torch.tensor(indices)
    
    def fit_transform(self, texts, max_length=None):
        """Build vocabulary and transform texts"""
        self.fit(texts)
        return self.transform(texts, max_length)
    
    def vocab_size(self):
        return len(self.word2idx)

def load_embedding(word2idx, embedding_path, embedding_dim=300):
    """Load pre-trained word embeddings"""
    embeddings = np.random.normal(scale=0.6, size=(len(word2idx), embedding_dim))
    # Special tokens
    embeddings[0] = np.zeros(embedding_dim)  # PAD
    
    if embedding_path and os.path.exists(embedding_path):
        print(f"Loading embeddings from {embedding_path}")
        with open(embedding_path, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                if word in word2idx:
                    vector = np.array([float(val) for val in values[1:]])
                    embeddings[word2idx[word]] = vector
    
    return torch.FloatTensor(embeddings)

def create_dataloader(dataset, batch_size=32, shuffle=True):
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn
    )

def collate_fn(batch):
    """Custom collate function for padding sequences in a batch"""
    texts = [item['text'] for item in batch]
    tokens = [item['tokens'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch])
    
    # Find max length in batch
    max_length = max(len(t) for t in tokens)
    
    # Pad tokens to max_length
    padded_tokens = []
    for t in tokens:
        padded = t + ['<PAD>'] * (max_length - len(t))
        padded_tokens.append(padded)
    
    return {
        'texts': texts,
        'tokens': padded_tokens,
        'labels': labels
    }
