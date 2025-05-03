import torch
from transformers import AutoModel, AutoTokenizer
import mteb
import numpy as np
from typing import List, Dict, Set
import pandas as pd
from tqdm import tqdm
import os
import hashlib
import json
from pathlib import Path
import gc


def mean_pooling_from_layer(hidden_state, attention_mask):
    """
    Mean pooling over token embeddings from a BERT layer (with batch support).

    Args:
        hidden_state: Tensor of shape (batch_size, seq_len, hidden_dim)
        attention_mask: Tensor of shape (batch_size, seq_len)

    Returns:
        mean_pooled: Tensor of shape (batch_size, hidden_dim)
    """
    # Expand attention mask for broadcasting
    attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)

    # Apply mask
    masked_embeddings = hidden_state * attention_mask

    # Sum over tokens, then divide by the number of valid tokens per sentence
    sum_embeddings = masked_embeddings.sum(dim=1)
    token_counts = attention_mask.sum(dim=1)  # (batch_size, 1)
    mean_pooled = sum_embeddings / token_counts

    return mean_pooled.numpy()


class BertLayerAnalyzer:
    def __init__(self, model_name: str = "bert-base-uncased"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.num_layers = self.model.config.num_hidden_layers
        self._processed_texts: Set[str] = set()  # Track processed texts
        self._cache = {}  # Cache for layer outputs

    def _get_embeddings(self, texts: List[str], layerIndex: int,  task_name: str):
        # use hashlib to create a hash of each item in texts
        cache_key = [hashlib.sha256(text.encode()).hexdigest()
                     for text in texts]
        uncached_texts = [(key, text) for text, key in zip(
            texts, cache_key) if key not in self._processed_texts]
        if uncached_texts:
            # Tokenize texts
            # create a batch of uncached texts
            batch_size = 8
            for i in tqdm(range(0, len(uncached_texts), batch_size), desc=f"Processing batches"):
                batch = uncached_texts[i:i+batch_size]
                batch_keys = [key for key, _ in batch]
                batch_texts = [text for _, text in batch]
                encoded = self.tokenizer(
                    batch_texts, padding="max_length", truncation=True, return_tensors="pt")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                # Get embeddings with output from all layers
                with torch.no_grad():
                    outputs = self.model(**encoded, output_hidden_states=True)

                # Detach and move states to CPU to free GPU memory and allow caching
                for ind, layer_output in enumerate(outputs.hidden_states):
                    # Shape: [batch_size, seq_len, 1]
                    embeddings_np = mean_pooling_from_layer(
                        layer_output.cpu(), encoded['attention_mask'].cpu())
                    # Save embeddings for each text in the batch
                    for key, embedding in zip(batch_keys, embeddings_np):
                        if ind not in self._cache:
                            self._cache[ind] = {}
                        self._cache[ind][key] = embedding
                        self._processed_texts.add(key)

        all_embeddings = []
        for key in cache_key:
            all_embeddings.append(self._cache[layerIndex][key])
        all_embeddings = np.array(all_embeddings)
        print(f"All embeddings shape: {all_embeddings.shape}")  # Debug print
        return all_embeddings

    def _get_layer_embeddings_uncached(self, texts: List[str], layer_idx: int) -> np.ndarray:
        batch_size = 128
        layer_outputs = []
        attention_masks = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Processing batches"):
            batch_texts = texts[i:i+batch_size]
            encoded = self.tokenizer(
                batch_texts, padding="max_length", truncation=True, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)

            # Only extract the desired layer and move it to CPU
            # [lyer, batch_size, seq_len, hidden_size]
            layer_hidden = outputs.hidden_states[layer_idx].cpu()
            attention_mask = encoded['attention_mask'].cpu()

            layer_outputs.extend(layer_hidden)
            attention_masks.extend(attention_mask)

        hidden_states = torch.stack(layer_outputs)
        attention_mask = torch.stack(attention_masks)
        return hidden_states, attention_mask

    def get_layer_embeddings(self, texts: List[str], layer_idx: int, task_name) -> np.ndarray:
        """Get embeddings from a specific transformer layer using cached hidden states."""

        return self._get_embeddings(texts, layerIndex=layer_idx, task_name=task_name)

    def clear_cache(self):
        """Clears the hidden state cache."""
        import gc
        self._cache = {}
        self._processed_texts = set()
        gc.collect()  # Suggest garbage collection

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
                prompt_type=None,
                **kwargs
            ):
                embeddings = self.analyzer.get_layer_embeddings(
                    sentences, self.layer_idx, task_name)
                # Ensure 2D array
                if len(embeddings.shape) == 1:
                    embeddings = embeddings.reshape(1, -1)
                return embeddings

        evaluator = LayerEvaluator(self, layer_idx)
        benchmark = mteb.MTEB(tasks=tasks)
        results = benchmark.run(
            evaluator, output_folder=f'results_latest/{layer_idx}')
        return results


tasks_by_categories = {
    "Classification": ["AmazonReviewsClassification", "Banking77Classification", "EmotionClassification", "MTOPDomainClassification", "MTOPIntentClassification", "MassiveIntentClassification", "MassiveScenarioClassification", "ToxicConversationsClassification", "TweetSentimentExtractionClassification"],

    "Clustering": ["ArxivClusteringS2S", "BiorxivClusteringS2S", "MedrxivClusteringS2S", "RedditClustering",
                   "StackExchangeClustering", "TwentyNewsgroupsClustering"],
    "Reranking": ["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR", "StackOverflowDupQuestions"],
    "STS": ["BIOSSES", "SICK-R", "STS12", "STS13", "STS14", "STS15", "STS16", "STS17", "STSBenchmark"]
}


def main():
    # Initialize analyzer
    analyzer = BertLayerAnalyzer()

    # Analyze each layer
    all_tasks = [task for sublist in tasks_by_categories.values()
                 for task in sublist]
    tasks = mteb.get_tasks(tasks=["BIOSSES"], languages=["eng"])

    for task in tasks:
        for layer_idx in tqdm(range(13)):
            print(f"\nEvaluating layer {layer_idx}")
            analyzer.evaluate_layer(layer_idx, [task])
        analyzer.clear_cache()


if __name__ == "__main__":
    main()
