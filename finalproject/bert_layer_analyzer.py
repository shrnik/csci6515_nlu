import torch
from transformers import AutoModel, AutoTokenizer
import mteb
import numpy as np
from typing import List, Dict
import pandas as pd
from tqdm import tqdm
import os
import hashlib
import json
from pathlib import Path
import gc

class BertLayerAnalyzer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_layers = self.model.config.num_hidden_layers
        self._cache = {} # Cache for hidden states

    def clear_cache(self):
        """Clears the hidden state cache."""
        self._cache = {}
        gc.collect() # Suggest garbage collection
    def _get_all_hidden_states(self, texts: List[str]):
        """Runs the model forward pass and caches/retrieves hidden states."""
        # use hashlib to create a hash of each item in texts
        cache_key = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
        uncached_texts = [(key, text) for text, key in zip(texts, cache_key) if key not in self._cache]
        if uncached_texts:
            # Tokenize texts

            # create a batch of uncached texts
            batch_size = 16
            for i in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[i:i+batch_size]
                batch_keys = [key for key, _ in batch]
                batch_texts = [text for _, text in batch]
                encoded = self.tokenizer(batch_texts, padding="max_length", truncation=True, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Get embeddings with output from all layers
                with torch.no_grad():
                    outputs = self.model(**encoded, output_hidden_states=True)

                # Detach and move states to CPU to free GPU memory and allow caching
                hidden_states_cpu = [state.cpu() for state in outputs.hidden_states]
                attention_mask_cpu = encoded['attention_mask'].cpu()

                # Store in cache
                for idx, key in enumerate(batch_keys):
                    self._cache[key] = {
                        "hidden_states": [layer[idx] for layer in hidden_states_cpu],  # one tensor per layer for this sample
                        "attention_mask": attention_mask_cpu[idx]  # attention mask for this sample
                    }
                    
        # Return cached states (already on CPU)
        # Stack the hidden states and attention masks into batched tensors
        all_hidden_states = [torch.stack([self._cache[key]["hidden_states"][i] for key in cache_key]) 
                           for i in range(len(self._cache[cache_key[0]]["hidden_states"]))]
        all_attention_masks = torch.stack([self._cache[key]["attention_mask"] for key in cache_key])
        
        # Return tuple matching model output format
        return all_hidden_states, all_attention_masks

    def get_layer_embeddings(self, texts: List[str], layer_idx: int) -> np.ndarray:
        """Get embeddings from a specific transformer layer using cached hidden states."""
        hidden_states, attention_mask = self._get_all_hidden_states(texts)

        # Get the specific layer's output from the cached list
        # hidden_states[0] is embedding layer, hidden_states[1] is layer 0 output, etc.
        layer_output = hidden_states[layer_idx + 1]

        # Mean pooling (ensure mask is correctly shaped)
        attention_mask_unsqueezed = attention_mask.unsqueeze(-1) # Shape: [batch_size, seq_len, 1]

        # Ensure division by zero doesn't happen for empty sequences if any
        sum_mask = torch.sum(attention_mask_unsqueezed, dim=1)
        sum_mask = torch.clamp(sum_mask, min=1e-9) # Avoid division by zero

        embeddings = torch.sum(layer_output * attention_mask_unsqueezed, dim=1) / sum_mask

        embeddings_np = embeddings.numpy() # Already on CPU
        print(f"Embeddings shape: {embeddings_np.shape}")  # Debug print
        return embeddings_np

    def clear_cache(self):
        """Clears the hidden state cache."""
        self._cache = {}
        import gc
        gc.collect() # Suggest garbage collection

    def evaluate_layer(self, layer_idx: int, tasks: List[str] = None) -> Dict:
        """Evaluate a specific layer on MTEB benchmark tasks."""
        if tasks is None:
            return None

        class LayerEvaluator:
            def __init__(self, analyzer, layer_idx):
                self.analyzer = analyzer
                self.layer_idx = layer_idx

            def encode(
              self,
              sentences: list[str],
              *,
              task_name: str,
              prompt_type = None,
              **kwargs
            ):
                embeddings = self.analyzer.get_layer_embeddings(sentences, self.layer_idx)
                # Ensure 2D array
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)
                return embeddings

        evaluator = LayerEvaluator(self, layer_idx)
        benchmark = mteb.MTEB(tasks=tasks)
        results = benchmark.run(evaluator, output_folder=f'results/{layer_idx}')
        return results

def main():
    # Initialize analyzer
    analyzer = BertLayerAnalyzer()

    # Analyze each layer
    tasks = mteb.get_tasks(tasks=["BIOSSES"])

    for layer_idx in tqdm(range(analyzer.num_layers)):
        print(f"\nEvaluating layer {layer_idx}")
        results = analyzer.evaluate_layer(layer_idx, tasks)

main()
