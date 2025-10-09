from transformers import CLIPModel, CLIPProcessor
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image
import hashlib
import pickle
import os
from pathlib import Path
import time
import csv
import datetime


#abstract class
class EmbeddingModel:
    """Abstract class for embedding models."""
    @abstractmethod
    def embed_image(self, image_path):
        """Embed a single image."""
        pass


class ClipEmbeddingModel(EmbeddingModel):
    """Handles image and dataset embedding operations."""

    def __init__(self):
        self.embedding_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to("cuda")
        self.embedding_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise

        if self.embedding_processor is None or self.embedding_model is None:
            print("Error: CLIP model or processor is not initialized.")
            raise
        
        inputs = self.embedding_processor(images=image, return_tensors="pt").to("cuda")

        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = (
                self.embedding_model.get_image_features(**inputs).detach().cpu().numpy()
            )
        return image_embeddings


class EmbeddingManager:
    """Handles image and dataset embedding operations."""
    
    def __init__(self, embedding_model: EmbeddingModel = ClipEmbeddingModel()):
        self.embedding_model = embedding_model
    
    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        return self.embedding_model.embed_image(image_path)
        
    def embed_dataset(self, dataset_path, debug=False) -> npt.NDArray:
        """Embed all images in a dataset."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

        print(f"Embedding dataset from '{dataset_path}' ...")
        dataset_embeddings = []

        # IDD has both png and jpg images in train set
        image_files = list(dataset_path.rglob("*.png")) + list(dataset_path.rglob("*.jpg"))
    
        if not image_files:
            print(f"Warning: No images found in dataset path '{dataset_path}'.")
            return []

        for img in image_files:
            embedding = self.embed_image(img)
            if embedding is not None:
                dataset_embeddings.append(embedding)
            else:
                raise ValueError(f"Error embedding image '{img}'.")

        print("Finished embedding dataset.")
        return dataset_embeddings
        
    def embed_images_batch(self, image_paths: list[str], batch_size: int = 128) -> npt.NDArray:
        """Embed multiple images in batches for efficiency."""
        all_embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load all images in the batch
            for image_path in batch_paths:
                try:
                    image = Image.open(image_path).convert("RGB")
                    batch_images.append(image)
                except FileNotFoundError:
                    print(f"Error: Image file '{image_path}' not found.")
                    raise
                except Exception as e:
                    print(f"Error opening image '{image_path}': {e}")
                    raise
            
            # Process batch through CLIP
            if self.embedding_model.embedding_processor is None or self.embedding_model.embedding_model is None:
                print("Error: CLIP model or processor is not initialized.")
                raise
            
            inputs = self.embedding_model.embedding_processor(images=batch_images, return_tensors="pt").to("cuda")
            
            # Generate embeddings for the batch
            with torch.no_grad():
                batch_embeddings = (
                    self.embedding_model.embedding_model.get_image_features(**inputs).detach().cpu().numpy()
                )
            
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)

    def calculate_statistics(self, domain_name, domain_path, train_path):

        """
        Calculate or load domain statistics.
        Args:
            domain_name (str): The name of the domain.
            domain_path (Path): The path to the domain database where the statistics will be saved.
            train_path (Path): The path to the train set.
        Returns:
            dict: A dictionary containing the statistics.
        """
        suffix = "_statistics.npz"
        statistics_path = domain_path / f"{domain_name}{suffix}"
        stats_dict = {}

        print(f"Statistics file: {statistics_path}")
        if statistics_path.exists():  # Load the data if it exists
            try:
                print(f"Loading statistics from {domain_name}{suffix} ...")
                stats = np.load(statistics_path)
                stats_dict.update({
                    "train_average_embedding": stats["train_average_embedding"],
                })
                print(f"Statistics loaded from {domain_name}{suffix}")
                return stats_dict
            except Exception as e:
                print(f"Error loading statistics file '{statistics_path}': {e}")
                return None

        print(f"Statistics file {statistics_path} does not exist, calculating statistics for domain '{domain_name}' ...")
        train_dataset_embeddings = self.embed_dataset(train_path)

        if not train_dataset_embeddings:
            raise ValueError("No embeddings were generated for dataset.")

        try:
            train_average_embedding = np.mean(train_dataset_embeddings, axis=0)
        except Exception as e:
            print(f"Error computing mean embedding: {e}")
            raise

        stats_dict.update({
            "train_average_embedding": train_average_embedding
        })

        try:
            np.savez(
                statistics_path,
                train_average_embedding=train_average_embedding,
            )
            print(f"Statistics saved to {domain_name}{suffix}")
        except Exception as e:
            print(f"Error saving statistics file '{statistics_path}': {e}")
            raise
        return stats_dict

    def _generate_cache_key(self, domain_name: str, image_paths: list[str]) -> str:
        """Generate a unique cache key based on domain name and image paths."""
        # Create a hash of the sorted image paths for consistency
        paths_str = "|".join(sorted(image_paths))
        paths_hash = hashlib.md5(paths_str.encode()).hexdigest()
        return f"{domain_name}_{paths_hash}"

    def _get_cache_path(self, cache_key: str, cache_dir: str = "./embedding_cache") -> Path:
        """Get the cache file path for a given cache key."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{cache_key}.pkl"

    def _load_from_cache(self, cache_key: str, cache_dir: str = "./embedding_cache") -> npt.NDArray:
        """Load embeddings from cache if they exist."""
        cache_path = self._get_cache_path(cache_key, cache_dir)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    print(f"Loaded embeddings from cache: {cache_path}")
                    return cached_data['embeddings']
            except Exception as e:
                print(f"Error loading cache {cache_path}: {e}")
                return None
        return None

    def _save_to_cache(self, cache_key: str, embeddings: npt.NDArray, cache_dir: str = "./embedding_cache") -> None:
        """Save embeddings to cache."""
        cache_path = self._get_cache_path(cache_key, cache_dir)
        try:
            cache_data = {
                'embeddings': embeddings,
                'timestamp': time.time(),
                'domain_name': cache_key.split('_')[0]
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved embeddings to cache: {cache_path}")
        except Exception as e:
            print(f"Error saving cache {cache_path}: {e}")

    def embed_images_batch_with_cache(self, domain_name: str, image_paths: list[str], batch_size: int = 128, cache_dir: str = "./embedding_cache") -> npt.NDArray:
        """Embed multiple images in batches with caching support."""
        # Generate cache key
        cache_key = self._generate_cache_key(domain_name, image_paths)
        
        # Try to load from cache first
        cached_embeddings = self._load_from_cache(cache_key, cache_dir)
        if cached_embeddings is not None:
            return cached_embeddings
        
        # If not in cache, compute embeddings
        print(f"Computing embeddings for domain {domain_name} (not found in cache)")
        embeddings = self.embed_images_batch(image_paths, batch_size)
        
        # Save to cache
        self._save_to_cache(cache_key, embeddings, cache_dir)
        
        return embeddings

    def clear_cache(self, cache_dir: str = "./embedding_cache") -> None:
        """Clear all cached embeddings."""
        cache_path = Path(cache_dir)
        if cache_path.exists():
            for cache_file in cache_path.glob("*.pkl"):
                cache_file.unlink()
            print(f"Cleared all cache files in {cache_dir}")

    def get_cache_info(self, cache_dir: str = "./embedding_cache") -> dict:
        """Get information about cached embeddings."""
        cache_path = Path(cache_dir)
        cache_info = {}
        
        if cache_path.exists():
            for cache_file in cache_path.glob("*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        cache_info[cache_file.name] = {
                            'domain_name': cached_data.get('domain_name', 'unknown'),
                            'timestamp': cached_data.get('timestamp', 0),
                            'embedding_shape': cached_data['embeddings'].shape if 'embeddings' in cached_data else 'unknown'
                        }
                except Exception as e:
                    cache_info[cache_file.name] = {'error': str(e)}
        
        return cache_info

    def is_cached(self, domain_name: str, image_paths: list[str], cache_dir: str = "./embedding_cache") -> bool:
        """Check if embeddings for a domain are already cached."""
        cache_key = self._generate_cache_key(domain_name, image_paths)
        cache_path = self._get_cache_path(cache_key, cache_dir)
        return cache_path.exists()

    def _generate_paths_cache_key(self, domain_name: str, data_loader_info: str = None) -> str:
        """Generate a unique cache key for image paths based on domain name and data loader info."""
        # Use domain name and a hash of data loader info if available
        if data_loader_info:
            info_hash = hashlib.md5(data_loader_info.encode()).hexdigest()[:8]
            return f"{domain_name}_paths_{info_hash}"
        else:
            return f"{domain_name}_paths"

    def _get_paths_cache_path(self, cache_key: str, cache_dir: str = "./path_cache") -> Path:
        """Get the cache file path for image paths."""
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(exist_ok=True)
        return cache_dir / f"{cache_key}.pkl"

    def _load_paths_from_cache(self, domain_name: str, cache_dir: str = "./path_cache") -> list[str]:
        """Load image paths from cache if they exist."""
        # Try different cache key formats
        possible_keys = [
            f"{domain_name}_paths",
            f"{domain_name}_paths_latest"
        ]
        
        for cache_key in possible_keys:
            cache_path = self._get_paths_cache_path(cache_key, cache_dir)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        print(f"Loaded image paths from cache: {cache_path} ({len(cached_data['paths'])} paths)")
                        return cached_data['paths']
                except Exception as e:
                    print(f"Error loading paths cache {cache_path}: {e}")
                    continue
        
        return None

    def _save_paths_to_cache(self, domain_name: str, image_paths: list[str], cache_dir: str = "./path_cache") -> None:
        """Save image paths to cache."""
        cache_key = f"{domain_name}_paths_latest"
        cache_path = self._get_paths_cache_path(cache_key, cache_dir)
        
        try:
            cache_data = {
                'paths': image_paths,
                'timestamp': time.time(),
                'domain_name': domain_name,
                'path_count': len(image_paths)
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Saved {len(image_paths)} image paths to cache: {cache_path}")
        except Exception as e:
            print(f"Error saving paths cache {cache_path}: {e}")

    def get_cached_image_paths(self, domain_name: str, data_loader, cache_dir: str = "./path_cache", force_refresh: bool = False) -> list[str]:
        """Get image paths for a domain, using cache if available."""
        
        # Try to load from cache first (unless force refresh)
        if not force_refresh:
            cached_paths = self._load_paths_from_cache(domain_name, cache_dir)
            if cached_paths is not None:
                return cached_paths
        
        # If not in cache or force refresh, collect paths from data loader
        print(f"Collecting image paths for domain {domain_name} (not found in cache or force refresh enabled)")
        
        time_start = time.time()
        image_paths = []
        
        try:
            # Simple and reliable method
            for inputs in data_loader:
                if isinstance(inputs, (list, tuple)) and len(inputs) > 0:
                    if isinstance(inputs[0], dict) and 'file_name' in inputs[0]:
                        image_paths.append(inputs[0]['file_name'])
                    else:
                        # Handle different input formats
                        for item in inputs:
                            if isinstance(item, dict) and 'file_name' in item:
                                image_paths.append(item['file_name'])
        except Exception as e:
            print(f"Error collecting paths from data loader: {e}")
            return []
        
        time_end = time.time()
        print(f"Time taken to collect {len(image_paths)} image paths from domain {domain_name}: {time_end - time_start:.2f} seconds")
        
        # Save to cache
        if image_paths:
            self._save_paths_to_cache(domain_name, image_paths, cache_dir)
        
        return image_paths

    def clear_paths_cache(self, cache_dir: str = "./path_cache") -> None:
        """Clear all cached image paths."""
        cache_path = Path(cache_dir)
        if cache_path.exists():
            for cache_file in cache_path.glob("*_paths*.pkl"):
                cache_file.unlink()
            print(f"Cleared all path cache files in {cache_dir}")

    def get_paths_cache_info(self, cache_dir: str = "./path_cache") -> dict:
        """Get information about cached image paths."""
        cache_path = Path(cache_dir)
        cache_info = {}
        
        if cache_path.exists():
            for cache_file in cache_path.glob("*_paths*.pkl"):
                try:
                    with open(cache_file, 'rb') as f:
                        cached_data = pickle.load(f)
                        cache_info[cache_file.name] = {
                            'domain_name': cached_data.get('domain_name', 'unknown'),
                            'timestamp': cached_data.get('timestamp', 0),
                            'path_count': cached_data.get('path_count', 0),
                            'age_hours': (time.time() - cached_data.get('timestamp', 0)) / 3600
                        }
                except Exception as e:
                    cache_info[cache_file.name] = {'error': str(e)}
        
        return cache_info
