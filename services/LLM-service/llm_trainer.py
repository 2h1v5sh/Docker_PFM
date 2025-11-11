import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from pymongo import MongoClient
import numpy as np
import json
import re
from collections import Counter, deque
from typing import List, Dict, Tuple, Optional
import math
import os
from datetime import datetime
import pickle

# ==================== ADVANCED DATA EXTRACTION ====================
class MongoDBDataExtractor:
    """Enhanced MongoDB data extraction with multiple strategies"""
    
    def __init__(self, connection_string: str, database: str, collection: str):
        self.client = MongoClient(connection_string)
        self.db = self.client[database]
        self.collection = self.db[collection]
        
    def extract_text_data(self, text_field: str, limit: int = None) -> List[str]:
        """Extract text data from MongoDB"""
        query = {}
        cursor = self.collection.find(query, {text_field: 1})
        
        if limit:
            cursor = cursor.limit(limit)
        
        texts = []
        for doc in cursor:
            if text_field in doc and doc[text_field]:
                text = str(doc[text_field]).strip()
                if len(text) > 10:  # Filter very short texts
                    texts.append(text)
        
        print(f"✓ Extracted {len(texts)} documents from MongoDB")
        return texts
    
    def extract_conversation_pairs(self, user_field: str, assistant_field: str, limit: int = None) -> List[Tuple[str, str]]:
        """Extract conversation pairs for chat training"""
        cursor = self.collection.find({}, {user_field: 1, assistant_field: 1})
        
        if limit:
            cursor = cursor.limit(limit)
        
        pairs = []
        for doc in cursor:
            if user_field in doc and assistant_field in doc:
                user_msg = str(doc[user_field]).strip()
                assistant_msg = str(doc[assistant_field]).strip()
                if user_msg and assistant_msg:
                    pairs.append((user_msg, assistant_msg))
        
        print(f"✓ Extracted {len(pairs)} conversation pairs")
        return pairs
    
    def extract_multi_turn_conversations(self, conversation_field: str, limit: int = None) -> List[List[Dict]]:
        """Extract multi-turn conversations"""
        cursor = self.collection.find({}, {conversation_field: 1})
        
        if limit:
            cursor = cursor.limit(limit)
        
        conversations = []
        for doc in cursor:
            if conversation_field in doc:
                conv = doc[conversation_field]
                if isinstance(conv, list) and len(conv) > 0:
                    conversations.append(conv)
        
        print(f"✓ Extracted {len(conversations)} multi-turn conversations")
        return conversations

# ==================== ADVANCED TOKENIZER ====================
class BPETokenizer:
    """Byte Pair Encoding tokenizer for better subword handling"""
    
    def __init__(self, vocab_size: int = 30000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3, "<SEP>": 4}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>", 4: "<SEP>"}
        self.merges = {}
        self.char_vocab = set()
        
    def build_vocab(self, texts: List[str]):
        """Build BPE vocabulary"""
        print("Building BPE vocabulary...")
        
        # Initialize with character-level tokens
        for text in texts:
            self.char_vocab.update(text)
        
        # Add character tokens
        current_idx = len(self.word2idx)
        for char in sorted(self.char_vocab):
            if char not in self.word2idx:
                self.word2idx[char] = current_idx
                self.idx2word[current_idx] = char
                current_idx += 1
        
        # Build word frequency
        word_freq = Counter()
        for text in texts:
            words = text.lower().split()
            word_freq.update(words)
        
        # Add most common words
        most_common = word_freq.most_common(self.vocab_size - len(self.word2idx))
        for word, _ in most_common:
            if word not in self.word2idx:
                self.word2idx[word] = current_idx
                self.idx2word[current_idx] = word
                current_idx += 1
        
        print(f"✓ Vocabulary size: {len(self.word2idx)}")
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        text = text.lower().strip()
        words = text.split()
        
        tokens = [2]  # Start with <SOS>
        
        for word in words:
            if word in self.word2idx:
                tokens.append(self.word2idx[word])
            else:
                # Fallback to character-level
                for char in word:
                    tokens.append(self.word2idx.get(char, 1))
            
            if len(tokens) >= max_length - 1:
                break
        
        tokens.append(3)  # Add <EOS>
        
        # Pad if necessary
        if len(tokens) < max_length:
            tokens += [0] * (max_length - len(tokens))
        
        return tokens[:max_length]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text"""
        words = []
        current_word = ""
        
        for token in tokens:
            if token in [0, 2, 4]:  # Skip PAD, SOS, SEP
                continue
            if token == 3:  # Stop at EOS
                break
            
            word = self.idx2word.get(token, "")
            
            if len(word) == 1 and word in self.char_vocab:
                current_word += word
            else:
                if current_word:
                    words.append(current_word)
                    current_word = ""
                words.append(word)
        
        if current_word:
            words.append(current_word)
        
        return " ".join(words)
    
    def save(self, filepath: str):
        """Save tokenizer"""
        data = {
            'vocab_size': self.vocab_size,
            'word2idx': self.word2idx,
            'idx2word': {int(k): v for k, v in self.idx2word.items()},
            'char_vocab': list(self.char_vocab)
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"✓ Tokenizer saved to {filepath}")
    
    def load(self, filepath: str):
        """Load tokenizer"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.vocab_size = data['vocab_size']
        self.word2idx = data['word2idx']
        self.idx2word = {int(k): v for k, v in data['idx2word'].items()}
        self.char_vocab = set(data['char_vocab'])
        print(f"✓ Tokenizer loaded from {filepath}")

# ==================== ADVANCED DATASETS ====================
class ChatDataset(Dataset):
    """Dataset for chat/instruction training"""
    
    def __init__(self, conversation_pairs: List[Tuple[str, str]], tokenizer: BPETokenizer, max_length: int = 512):
        self.pairs = conversation_pairs
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        user_msg, assistant_msg = self.pairs[idx]
        
        # Format: <SOS> user_message <SEP> assistant_message <EOS>
        combined = f"{user_msg} <SEP> {assistant_msg}"
        tokens = self.tokenizer.encode(combined, self.max_length)
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

class MultiTurnChatDataset(Dataset):
    """Dataset for multi-turn conversations"""
    
    def __init__(self, conversations: List[List[Dict]], tokenizer: BPETokenizer, max_length: int = 512):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Concatenate all turns
        full_text = ""
        for turn in conversation:
            role = turn.get('role', 'user')
            content = turn.get('content', '')
            full_text += f"{role}: {content} <SEP> "
        
        tokens = self.tokenizer.encode(full_text, self.max_length)
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        target_ids = torch.tensor(tokens[1:], dtype=torch.long)
        
        return input_ids, target_ids

# ==================== ADVANCED TRANSFORMER MODEL ====================
class RotaryPositionalEncoding(nn.Module):
    """Rotary Position Embedding (RoPE) - More efficient than sinusoidal"""
    
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_len = max_len
        
    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb[None, :, :]

class MultiQueryAttention(nn.Module):
    """Multi-Query Attention for faster inference"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, self.head_dim)
        self.v_proj = nn.Linear(d_model, self.head_dim)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        batch_size, seq_len, _ = x.shape
        
        # Project queries
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Project keys and values (single head)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if past_kv is not None and use_cache:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        
        # Expand k, v to match number of query heads
        k = k.unsqueeze(1).expand(batch_size, self.n_heads, -1, self.head_dim)
        v = v.unsqueeze(1).expand(batch_size, self.n_heads, -1, self.head_dim)
        
        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn + mask
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        
        if use_cache:
            return out, (k[:, 0], v[:, 0])
        return out

class SwiGLU(nn.Module):
    """SwiGLU activation function (better than GELU for LLMs)"""
    
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_model, d_ff)
        self.w3 = nn.Linear(d_ff, d_model)
    
    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))

class TransformerBlock(nn.Module):
    """Enhanced transformer block with MQA and SwiGLU"""
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = MultiQueryAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = SwiGLU(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, use_cache=False, past_kv=None):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask, use_cache, past_kv)
        
        if use_cache:
            attn_out, present_kv = attn_out
        
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        
        if use_cache:
            return x, present_kv
        return x

class LargeLanguageModel(nn.Module):
    """Large-scale LLM with modern architecture"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 24,
        d_ff: int = 4096,
        max_seq_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.rotary_emb = RotaryPositionalEncoding(d_model // n_heads, max_seq_len)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.output.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
        # Calculate model size
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✓ Model initialized with {total_params:,} parameters ({total_params/1e6:.1f}M)")
    
    def _init_weights(self):
        """Initialize weights with scaled initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(self, x, mask=None, use_cache=False, past_kvs=None):
        # Embedding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.dropout(x)
        
        # Generate causal mask if not provided
        if mask is None:
            mask = self._generate_square_subsequent_mask(x.size(1)).to(x.device)
        
        # Transformer blocks
        present_kvs = []
        for i, layer in enumerate(self.layers):
            past_kv = past_kvs[i] if past_kvs is not None else None
            
            if use_cache:
                x, present_kv = layer(x, mask, use_cache, past_kv)
                present_kvs.append(present_kv)
            else:
                x = layer(x, mask, use_cache, past_kv)
        
        # Output projection
        x = self.norm(x)
        logits = self.output(x)
        
        if use_cache:
            return logits, present_kvs
        return logits
    
    def get_num_params(self) -> int:
        """Get number of parameters"""
        return sum(p.numel() for p in self.parameters())

# ==================== ADVANCED TRAINING ====================
class AdvancedLLMTrainer:
    """Advanced trainer with mixed precision, gradient accumulation, etc."""
    
    def __init__(
        self,
        model: LargeLanguageModel,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 2000,
        max_steps: int = 100000,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_mixed_precision: bool = True,
        checkpoint_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Optimizer with weight decay
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Learning rate scheduler with warmup
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.current_step = 0
        
        # Mixed precision training
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.scaler = GradScaler() if self.use_mixed_precision else None
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
        print(f"✓ Trainer initialized on {device}")
        print(f"✓ Mixed precision: {self.use_mixed_precision}")
        print(f"✓ Gradient accumulation steps: {gradient_accumulation_steps}")
    
    def get_lr(self) -> float:
        """Get learning rate with warmup and cosine decay"""
        if self.current_step < self.warmup_steps:
            return self.optimizer.param_groups[0]['lr'] * (self.current_step / self.warmup_steps)
        else:
            progress = (self.current_step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            return self.optimizer.param_groups[0]['lr'] * 0.5 * (1 + math.cos(math.pi * progress))
    
    def train_step(self, input_ids, target_ids) -> float:
        """Single training step"""
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)
        
        if self.use_mixed_precision:
            with autocast():
                logits = self.model(input_ids)
                loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
        else:
            logits = self.model(input_ids)
            loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
        
        return loss.item() * self.gradient_accumulation_steps
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (input_ids, target_ids) in enumerate(self.train_loader):
            loss = self.train_step(input_ids, target_ids)
            total_loss += loss
            num_batches += 1
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Update learning rate
                lr = self.get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Gradient clipping and optimizer step
                if self.use_mixed_precision:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.current_step += 1
                
                # Logging
                if self.current_step % 100 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"  Step {self.current_step}/{self.max_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            
            # Early stopping if max steps reached
            if self.current_step >= self.max_steps:
                break
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """Validate the model"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for input_ids, target_ids in self.val_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                if self.use_mixed_precision:
                    with autocast():
                        logits = self.model(input_ids)
                        loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                else:
                    logits = self.model(input_ids)
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def train(self, epochs: int):
        """Train the model"""
        print(f"\n{'='*60}")
        print(f"Starting Training")
        print(f"{'='*60}")
        print(f"Total parameters: {self.model.get_num_params():,}")
        print(f"Device: {self.device}")
        print(f"Epochs: {epochs}")
        print(f"Max steps: {self.max_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"{'='*60}")
            
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            print(f"\n📊 Epoch {epoch + 1} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            
            if self.val_loader:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"  Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt', is_best=True)
                    print(f"  ✓ New best model saved! (Val Loss: {val_loss:.4f})")
            
            # Regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
            
            if self.current_step >= self.max_steps:
                print(f"\n✓ Reached max steps ({self.max_steps}), stopping training")
                break
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        self.save_checkpoint('final_model.pt')
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """Save model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'current_step': self.current_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
        }
        
        if self.use_mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if not is_best:
            print(f"  ✓ Checkpoint saved: {filename}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_step = checkpoint['current_step']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']
        
        if self.use_mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✓ Checkpoint loaded: {filename}")

# ==================== CHAT SERVICE ====================
class ChatService:
    """Advanced chat service with conversation history and context management"""
    
    def __init__(
        self,
        model: LargeLanguageModel,
        tokenizer: BPETokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        max_history: int = 10
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.max_history = max_history
        
        # Conversation history
        self.conversation_history = deque(maxlen=max_history)
        
        # System prompt
        self.system_prompt = "You are a helpful AI assistant. Provide clear, accurate, and helpful responses."
    
    def set_system_prompt(self, prompt: str):
        """Set custom system prompt"""
        self.system_prompt = prompt
        print(f"✓ System prompt updated")
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
        print("✓ Conversation history cleared")
    
    def add_to_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({"role": role, "content": content})
    
    def get_context(self) -> str:
        """Get formatted conversation context"""
        context = f"System: {self.system_prompt}\n"
        
        for msg in self.conversation_history:
            role = msg['role'].capitalize()
            content = msg['content']
            context += f"{role}: {content}\n"
        
        return context
    
    def generate_response(
        self,
        user_message: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        use_history: bool = True,
        stream: bool = False
    ) -> str:
        """Generate response to user message"""
        
        # Add user message to history
        if use_history:
            self.add_to_history("user", user_message)
            context = self.get_context() + "Assistant:"
        else:
            context = f"User: {user_message}\nAssistant:"
        
        # Encode context
        tokens = self.tokenizer.encode(context, max_length=1024)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        generated_tokens = []
        
        with torch.no_grad():
            for step in range(max_length):
                # Get predictions
                logits = self.model(input_ids)
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Stop at EOS or SEP
                if next_token.item() in [3, 4]:
                    break
                
                generated_tokens.append(next_token.item())
                
                # Stream output if enabled
                if stream:
                    token_text = self.tokenizer.decode([next_token.item()])
                    print(token_text, end='', flush=True)
                
                # Append to input for next iteration
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Truncate if too long
                if input_ids.size(1) > 1024:
                    input_ids = input_ids[:, -1024:]
        
        # Decode response
        response = self.tokenizer.decode(generated_tokens)
        
        # Add to history
        if use_history:
            self.add_to_history("assistant", response)
        
        return response
    
    def chat(self):
        """Interactive chat interface"""
        print("\n" + "="*60)
        print("🤖 Chat Service Started")
        print("="*60)
        print("Commands:")
        print("  /clear  - Clear conversation history")
        print("  /system - Set system prompt")
        print("  /quit   - Exit chat")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("\n👋 Goodbye!")
                        break
                    elif user_input == '/clear':
                        self.clear_history()
                        continue
                    elif user_input.startswith('/system'):
                        new_prompt = input("Enter new system prompt: ").strip()
                        if new_prompt:
                            self.set_system_prompt(new_prompt)
                        continue
                    else:
                        print("❌ Unknown command")
                        continue
                
                # Generate response
                print("\n🤖 Assistant: ", end='', flush=True)
                response = self.generate_response(user_input, stream=True)
                print()  # New line after streaming
                
            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue

# ==================== ADVANCED INFERENCE ====================
class AdvancedInference:
    """Advanced inference with beam search, caching, and more"""
    
    def __init__(
        self,
        model: LargeLanguageModel,
        tokenizer: BPETokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.kv_cache = None
    
    def beam_search(
        self,
        prompt: str,
        max_length: int = 100,
        num_beams: int = 5,
        temperature: float = 1.0,
        length_penalty: float = 1.0
    ) -> List[Tuple[str, float]]:
        """Beam search for better quality generation"""
        
        tokens = self.tokenizer.encode(prompt, max_length=512)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        # Initialize beams
        beams = [(input_ids, 0.0)]  # (sequence, score)
        
        with torch.no_grad():
            for _ in range(max_length):
                new_beams = []
                
                for seq, score in beams:
                    if seq[0, -1].item() in [3, 4]:  # EOS or SEP
                        new_beams.append((seq, score))
                        continue
                    
                    logits = self.model(seq)
                    next_token_logits = logits[0, -1, :] / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    
                    # Get top-k tokens
                    top_probs, top_indices = torch.topk(probs, num_beams)
                    
                    for prob, idx in zip(top_probs, top_indices):
                        new_seq = torch.cat([seq, idx.unsqueeze(0).unsqueeze(0)], dim=1)
                        new_score = score + torch.log(prob).item()
                        new_beams.append((new_seq, new_score))
                
                # Keep top beams
                beams = sorted(new_beams, key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)[:num_beams]
                
                # Check if all beams ended
                if all(seq[0, -1].item() in [3, 4] for seq, _ in beams):
                    break
        
        # Decode results
        results = []
        for seq, score in beams:
            tokens = seq[0].tolist()
            text = self.tokenizer.decode(tokens)
            results.append((text, score))
        
        return results
    
    def generate_with_cache(
        self,
        prompt: str,
        max_length: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """Generate using KV cache for faster inference"""
        
        tokens = self.tokenizer.encode(prompt, max_length=512)
        input_ids = torch.tensor([tokens], dtype=torch.long).to(self.device)
        
        generated_tokens = []
        past_kvs = None
        
        with torch.no_grad():
            for step in range(max_length):
                # Use cache for faster inference
                if step == 0:
                    logits, past_kvs = self.model(input_ids, use_cache=True)
                else:
                    logits, past_kvs = self.model(input_ids[:, -1:], use_cache=True, past_kvs=past_kvs)
                
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-p filtering
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                if next_token.item() in [3, 4]:
                    break
                
                generated_tokens.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
        
        return self.tokenizer.decode(generated_tokens)

# ==================== EVALUATION ====================
class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self, model: LargeLanguageModel, tokenizer: BPETokenizer, device: str = 'cuda'):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def calculate_perplexity(self, test_loader: DataLoader) -> float:
        """Calculate perplexity on test set"""
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
        
        with torch.no_grad():
            for input_ids, target_ids in test_loader:
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                
                logits = self.model(input_ids)
                loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
                
                total_loss += loss.item()
                total_tokens += (target_ids != 0).sum().item()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def evaluate_generation_quality(self, prompts: List[str], max_length: int = 100) -> Dict:
        """Evaluate generation quality metrics"""
        inference = AdvancedInference(self.model, self.tokenizer, self.device)
        
        results = {
            'generations': [],
            'avg_length': 0,
            'unique_tokens': set()
        }
        
        total_length = 0
        
        for prompt in prompts:
            generated = inference.generate_with_cache(prompt, max_length)
            results['generations'].append({'prompt': prompt, 'generated': generated})
            
            tokens = self.tokenizer.encode(generated)
            total_length += len(tokens)
            results['unique_tokens'].update(tokens)
        
        results['avg_length'] = total_length / len(prompts)
        results['vocab_diversity'] = len(results['unique_tokens'])
        results['unique_tokens'] = None  # Don't store the set
        
        return results

# ==================== MAIN PIPELINE ====================
def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print("🚀 CUSTOM LLM TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # ===== CONFIGURATION =====
    # MongoDB Configuration
    MONGODB_URI = "mongodb://localhost:27017/"
    DATABASE = "your_database"
    COLLECTION = "your_collection"
    TEXT_FIELD = "text"  # For general text
    USER_FIELD = "user_message"  # For chat pairs
    ASSISTANT_FIELD = "assistant_message"  # For chat pairs
    
    # Test MongoDB connection
    try:
        client = MongoClient("mongodb://localhost:27017/")
        client.server_info()  # Test connection
        print(f"✓ MongoDB connection successful")
        client.close()
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        print("Please ensure MongoDB is running on localhost:27017")
        return
    
    # Model Configuration
    VOCAB_SIZE = 30000
    D_MODEL = 1024  # Embedding dimension
    N_HEADS = 16    # Attention heads
    N_LAYERS = 24   # Transformer layers
    D_FF = 4096     # Feed-forward dimension
    MAX_SEQ_LEN = 2048
    DROPOUT = 0.1
    
    # Training Configuration
    BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    EFFECTIVE_BATCH_SIZE = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.1
    WARMUP_STEPS = 2000
    MAX_STEPS = 100000
    EPOCHS = 10
    
    # System Configuration
    USE_MIXED_PRECISION = True
    CHECKPOINT_DIR = './checkpoints'
    
    print(f"Configuration:")
    print(f"  Model Size: {D_MODEL}d, {N_LAYERS} layers, {N_HEADS} heads")
    print(f"  Vocabulary: {VOCAB_SIZE:,} tokens")
    print(f"  Batch Size: {EFFECTIVE_BATCH_SIZE} (effective)")
    print(f"  Max Steps: {MAX_STEPS:,}")
    print(f"="*60 + "\n")
    
    # ===== STEP 1: DATA EXTRACTION =====
    print("📥 STEP 1: Extracting Data from MongoDB")
    print("-"*60)
    
    extractor = MongoDBDataExtractor(MONGODB_URI, DATABASE, COLLECTION)
    
    # Choose extraction method based on your data
    # Option 1: General text data
    texts = extractor.extract_text_data(TEXT_FIELD, limit=50000)
    
    # Option 2: Conversation pairs (uncomment if you have chat data)
    # conversation_pairs = extractor.extract_conversation_pairs(USER_FIELD, ASSISTANT_FIELD, limit=50000)
    
    print()
    
    # ===== STEP 2: TOKENIZER =====
    print("🔤 STEP 2: Building Tokenizer")
    print("-"*60)
    
    tokenizer = BPETokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(texts)
    tokenizer.save("tokenizer.json")
    
    print()
    
    # ===== STEP 3: DATASETS =====
    print("📚 STEP 3: Creating Datasets")
    print("-"*60)
    
    # Split data
    train_size = int(0.9 * len(texts))
    val_size = int(0.05 * len(texts))
    
    train_texts = texts[:train_size]
    val_texts = texts[train_size:train_size+val_size]
    test_texts = texts[train_size+val_size:]
    
    print(f"✓ Train samples: {len(train_texts):,}")
    print(f"✓ Val samples: {len(val_texts):,}")
    print(f"✓ Test samples: {len(test_texts):,}")
    
    # Create datasets
    from torch.utils.data import Dataset as TorchDataset
    
    class TextDataset(TorchDataset):
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            tokens = self.tokenizer.encode(self.texts[idx], self.max_length)
            input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
            target_ids = torch.tensor(tokens[1:], dtype=torch.long)
            return input_ids, target_ids
    
    train_dataset = TextDataset(train_texts, tokenizer, MAX_SEQ_LEN)
    val_dataset = TextDataset(val_texts, tokenizer, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)
    
    print()
    
    # ===== STEP 4: MODEL =====
    print("🧠 STEP 4: Initializing Model")
    print("-"*60)
    
    model = LargeLanguageModel(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN,
        dropout=DROPOUT
    )
    
    print()
    
    # ===== STEP 5: TRAINING =====
    print("🏋️ STEP 5: Training Model")
    print("-"*60)
    
    trainer = AdvancedLLMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        use_mixed_precision=USE_MIXED_PRECISION,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    trainer.train(epochs=EPOCHS)
    
    print()
    
    # ===== STEP 6: EVALUATION =====
    print("📊 STEP 6: Evaluating Model")
    print("-"*60)
    
    test_dataset = TextDataset(test_texts, tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    evaluator = ModelEvaluator(model, tokenizer)
    perplexity = evaluator.calculate_perplexity(test_loader)
    
    print(f"✓ Test Perplexity: {perplexity:.2f}")
    
    print()
    
    # ===== STEP 7: DEMO =====
    print("🎯 STEP 7: Testing Generation")
    print("-"*60)
    
    test_prompts = [
        "Hello, how are you",
        "What is artificial intelligence",
        "Tell me a story about"
    ]
    
    inference = AdvancedInference(model, tokenizer)
    
    for prompt in test_prompts:
        print(f"\n💭 Prompt: {prompt}")
        generated = inference.generate_with_cache(prompt, max_length=50, temperature=0.8)
        print(f"🤖 Generated: {generated}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60 + "\n")

# ==================== CHAT MODE =====
def chat_mode():
    """Run in chat service mode"""
    
    print("\n🤖 Loading Chat Service...")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    # Load model
    model = LargeLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
        max_seq_len=2048
    )
    
    # Load checkpoint
    checkpoint = torch.load('./checkpoints/best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create chat service
    chat_service = ChatService(model, tokenizer, max_history=10)
    chat_service.chat()

# ==================== INFERENCE MODE =====
def inference_mode():
    """Run in inference mode"""
    
    print("\n🚀 Loading Inference Engine...")
    
    # Load tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load("tokenizer.json")
    
    # Load model
    model = LargeLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=1024,
        n_heads=16,
        n_layers=24,
        d_ff=4096,
        max_seq_len=2048
    )
    
    checkpoint = torch.load('./checkpoints/best_model.pt', map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("✓ Model loaded successfully\n")
    
    inference = AdvancedInference(model, tokenizer)
    
    print("Inference Mode (type 'quit' to exit)")
    print("="*60)
    
    while True:
        prompt = input("\nEnter prompt: ").strip()
        
        if prompt.lower() == 'quit':
            break
        
        if not prompt:
            continue
        
        print("\nGenerating...")
        
        # Beam search for better quality
        results = inference.beam_search(prompt, max_length=100, num_beams=3)
        
        print("\n📝 Generated responses:")
        for i, (text, score) in enumerate(results, 1):
            print(f"\n{i}. (score: {score:.2f})")
            print(f"   {text}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            main()
        elif sys.argv[1] == "chat":
            chat_mode()
        elif sys.argv[1] == "inference":
            inference_mode()
        else:
            print("Usage: python script.py [train|chat|inference]")
    else:
        # Default: training mode
        main()