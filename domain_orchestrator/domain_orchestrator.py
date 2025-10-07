import time
from typing import Union, Dict, Callable
from typing import Any, Literal, Mapping
from pathlib import Path
from argparse import Namespace
from dataclasses import dataclass
import logging
import os
import csv
import json
from datetime import datetime

import numpy as np
import numpy.typing as npt
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

import peft

from .embedding import EmbeddingManager

logging.disable()

torch.set_float32_matmul_precision("high")

from .utils import custom_domain_args, get_domain_args, benchmark_catseg, load_catseg_model


def softmax(x: list[float], softmax_temperature) -> np.ndarray:
    """Compute softmax values for each sets of scores in x."""
    
    # Add error handling for division by zero
    if softmax_temperature == 0:
        softmax_temperature = 1e-6
    print(f"softmax_temperature being used is {softmax_temperature}")
    exp_x = np.exp(np.divide(x, softmax_temperature))
    return exp_x / np.sum(exp_x, axis=0)


@dataclass
class Domain:
    """A simple Domain class with a name attribute."""
    name: str
    args: Namespace
    train_dataset_path: Path
    train_average_embedding: npt.NDArray
    data_loader: Any
    evaluator: Any
    lora_path: Path = None



class MoEGatingNetwork(nn.Module):
    """Mixture of Experts Gating Network for LoRA adapter weights"""
    
    def __init__(self, input_dim: int, num_domains: int, hidden_dim: int = 512):
        super(MoEGatingNetwork, self).__init__()
        self.input_dim = input_dim
        self.num_domains = num_domains
        self.hidden_dim = hidden_dim
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_domains)
        )
        print("MoEGatingNetwork init params: --------------------------------")
        print(input_dim, num_domains, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the gating network."""
        return self.network(x)
    
    def get_top_k_weights(self, x: torch.Tensor, top_k: int, temperature: float = 2) -> tuple[torch.Tensor, torch.Tensor]:
        """Get top-k domain weights and indices."""
        # 1.5 (17.93), 10(18.24), 5(18.29), 4(18.30), 3(18.27), 2(18.11)
        logits = self.forward(x)
        scaled_logits = logits / temperature
        # Get top-k values and indices from raw logits
        top_k_logits, top_k_indices = torch.topk(scaled_logits, top_k, dim=-1)
        
        # Create mask of -inf for all logits except top-k
        mask = torch.full_like(scaled_logits, float('-inf'))
        mask.scatter_(1, top_k_indices, top_k_logits)
        
        # Apply softmax to masked logits
        probs = F.softmax(mask, dim=-1)
        print(f"probs are {probs}")
        
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1)
        
        return top_k_values, top_k_indices
        


class DomainObserver:
    """Observer class that holds and manages a collection of Domains."""

    def __init__(
        self,
    ) -> None:
        self.domain_prototypes = {}
        self.gating_network = None
        self.domain_to_index = {}
        self.index_to_domain = {}

    def add_domain_prototypes(
        self,
        domain: Domain,
        average_embedding: npt.NDArray,
    ) -> None:
        """
        Add the average embedding of a domain to the observer.
        """
        self.domain_prototypes.update({domain.name: average_embedding})

    def calculate_similarity_to_domains(
        self, 
        embedding: npt.NDArray, 
        domains: list[Domain],
        similarity_measure: Callable[[npt.NDArray, npt.NDArray], np.float64],
        sort_descending: bool = True
    ) -> Dict[str, float]:
        """
        Calculate the similarity between the target embedding and the domain prototypes.
        """
        similarities = []

        for domain in domains:
            prot = self.domain_prototypes[domain.name]
            similarity = similarity_measure(embedding, prot)
            similarities.append([domain.name, similarity])

        # sort similarities from lowest to highest
        similarities_dict = dict(sorted(similarities, key=lambda x: x[1], reverse=sort_descending))
        return similarities_dict
    
    def calculate_moe_similarity_to_domains(
        self,
        embedding: npt.NDArray,
        domains: list[Domain],
        top_k: int = 5,
        filename: str = None,
        log_file_handle = None
    ) -> Dict[str, float]:

        # convert embedding to tensor
        device = next(self.gating_network.parameters()).device
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32).to(device)
        top_k_weights, top_k_indices = self.gating_network.get_top_k_weights(embedding_tensor, top_k)
        #print(f"shape of top_k_weights are {top_k_weights.shape}")
        #print(f"shape of top_k_indices are {top_k_indices.shape}")

        top_k_weights = top_k_weights.squeeze(0).detach().cpu().numpy()
        # TODO: Check if this is correct way to normalize the weights
        # top_k_weights = top_k_weights / np.sum(top_k_weights)
        print(f"top_k_weights are {top_k_weights}")
        top_k_indices = top_k_indices.squeeze(0).detach().cpu().numpy()

        # Log to console
        print(f"\n=== MoE Weight Distribution ===")
        if filename:
            print(f"File: {filename}")
        print(f"Top-{top_k} selected domains and weights:")
        
        domain_weights = {}
        for i, (weight, idx) in enumerate(zip(top_k_weights, top_k_indices)):
            domain_name = self.index_to_domain[idx]
            domain_weights[domain_name] = float(weight)
            
            # Log to console
            print(f"  Rank {i+1}: {domain_name} (weight: {weight:.4f})")
            
            # Log to CSV if file handle is provided
            if log_file_handle:
                writer = csv.writer(log_file_handle)
                writer.writerow([
                    filename or "unknown",
                    i + 1,
                    domain_name,
                    float(weight),
                    top_k
                ])
        
        return domain_weights


    # TODO: try various ways of training the gating network
    def train_gating_network(
        self,
        domains: list[Domain],
        embedding_manager: EmbeddingManager,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2,
        hidden_dim: int = 512,
        log_dir: str = "./moe_training_logs"
    ) -> None:
        """
        Train the MoE gating network using data from all source domains.
        """
        print("Training MoE gating network...")

        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
        # csv_file = os.path.join(log_dir, f"training_metrics_{timestamp}.csv")

        # Initialize csv logging
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy'])
        
        # Set up domain mapping
        self.domain_to_index = {domain.name: i for i, domain in enumerate(domains)}
        self.index_to_domain = {i: domain.name for i, domain in enumerate(domains)}
        
        # Collect embeddings and labels from all domains
        all_embeddings = []
        all_labels = []
        
        for domain in domains:
            print(f"Collecting embeddings from domain: {domain.name}")
            
            # Get training data loader
            data_loader = domain.data_loader
            
            domain_embeddings = []
            for inputs in data_loader:
                input_path = inputs[0]["file_name"]
                embedding = embedding_manager.embed_image(input_path)
                domain_embeddings.append(embedding)
            
            # Convert to numpy arrays
            domain_embeddings = np.array(domain_embeddings)
            domain_labels = np.full(len(domain_embeddings), self.domain_to_index[domain.name])
            
            all_embeddings.append(domain_embeddings)
            all_labels.append(domain_labels)
        
        # Concatenate all data
        X = np.vstack(all_embeddings).squeeze(1) # remove first dimension (=1)
        y = np.concatenate(all_labels)
        print(f"shape of X and y are {X.shape} and {y.shape}")
        
        # Split into train/validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Convert to tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_val = torch.tensor(y_val, dtype=torch.long)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize gating network
        input_dim = X.shape[1]
        num_domains = len(domains)
        self.gating_network = MoEGatingNetwork(input_dim, num_domains, hidden_dim=768)
        
        # Set up training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gating_network.to(device)
        
        optimizer = optim.Adam(self.gating_network.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            self.gating_network.train()
            train_loss = 0.0
            train_accuracy = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                print(f"input shape is {batch_X.shape}")
                outputs = self.gating_network(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                train_accuracy += (predicted == batch_y).sum().item()
            
            # Validation
            self.gating_network.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = self.gating_network(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    val_accuracy += (predicted == batch_y).sum().item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_accuracy /= len(train_dataset)
            val_accuracy /= len(val_dataset)
            print(f"length of train_loader is {len(train_loader)}")
            print(f"length of val_loader is {len(val_loader)}")
            print(f"length of train_dataset is {len(train_dataset)}")
            print(f"length of val_dataset is {len(val_dataset)}")
            
            print(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}\n")            

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print("MoE gating network training completed!")

    def train_moe_with_adapters(
        self,
        source_domains: list[str],
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 8,  # Small batch size as requested
        validation_split: float = 0.2,
        hidden_dim: int = 512,
        log_dir: str = "./moe_training_logs"
    ) -> None:
        """
        Train the MoE gating network jointly with all LoRA adapters using segmentation loss.
        Training is done sequentially on each source domain dataset.
        """
        
        print("Training MoE gating network with LoRA adapters using segmentation loss...")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Set up logging files
        log_file = os.path.join(log_dir, f"moe_training_log_{timestamp}.txt")
        csv_file = os.path.join(log_dir, f"moe_training_metrics_{timestamp}.csv")
        
        # Initialize CSV logging
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['domain', 'epoch', 'train_loss', 'val_loss'])
        
        # Get domain objects
        domains = [self._source_domains[name] for name in source_domains]
        
        # Initialize gating network
        embedding_dim = 768  # CLIP embedding dimension
        num_domains = len(domains)
        self.observer.gating_network = MoEGatingNetwork(embedding_dim, num_domains, hidden_dim)
        
        # Set up domain mapping
        self.observer.domain_to_index = {domain.name: i for i, domain in enumerate(domains)}
        self.observer.index_to_domain = {i: domain.name for i, domain in enumerate(domains)}
        
        # Initialize model with all adapters loaded
        self._initialize_model_with_adapters(domains)
        
        # Training loop - sequential training on each domain
        for domain_idx, domain in enumerate(domains):
            print(f"\n=== Training on domain: {domain.name} ({domain_idx + 1}/{len(domains)}) ===")
            
            # Get data loader for this domain
            data_loader = domain.data_loader
            
            # Split data for validation
            total_samples = len(data_loader.dataset)
            val_size = int(total_samples * validation_split)
            train_size = total_samples - val_size
            
            train_dataset, val_dataset = torch.utils.data.random_split(
                data_loader.dataset, [train_size, val_size]
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
            
            # Set up training for this domain
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.current_model.to(device)
            self.observer.gating_network.to(device)
            
            # Optimizer for both gating network and all LoRA adapters
            optimizer = optim.Adam(
                list(self.observer.gating_network.parameters()) + 
                list(self.current_model.parameters()),
                lr=learning_rate
            )
            
            # Training loop for this domain
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training
                self.current_model.train()
                self.observer.gating_network.train()
                train_loss = 0.0
                
                for batch_data in train_loader:
                    # Move data to device
                    if isinstance(batch_data, list):
                        for item in batch_data:
                            if isinstance(item, dict):
                                for key in item:
                                    if torch.is_tensor(item[key]):
                                        item[key] = item[key].to(device)
                    else:
                        batch_data = batch_data.to(device)
                    
                    # Get CLIP embeddings for gating
                    images = [x["image"].to(device) for x in batch_data]
                    clip_images = [(x - self.current_model.pixel_mean) / self.current_model.pixel_std for x in images]
                    clip_images_resized = F.interpolate(
                        torch.stack(clip_images), 
                        size=self.current_model.clip_resolution, 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Get CLIP embeddings for gating network
                    with torch.no_grad():
                        clip_features = self.current_model.sem_seg_head.predictor.clip_model.encode_image(
                            clip_images_resized, dense=True
                        )
                        # Use CLS token for gating
                        gating_embeddings = clip_features[:, 0, :]  # [batch_size, 768]
                    
                    # Get gating weights
                    gating_weights = self.observer.gating_network(gating_embeddings)
                    gating_probs = F.softmax(gating_weights, dim=-1)
                    
                    # Forward pass through model
                    loss_dict = self.current_model(batch_data)
                    losses = sum(loss_dict.values())
                    
                    # Weight the loss by gating probabilities
                    # For this domain, we want to maximize the corresponding gating weight
                    domain_weight = gating_probs[:, domain_idx].mean()
                    weighted_loss = losses * domain_weight
                    
                    # Backward pass
                    optimizer.zero_grad()
                    weighted_loss.backward()
                    optimizer.step()
                    
                    train_loss += weighted_loss.item()
                
                # Validation
                self.current_model.eval()
                self.observer.gating_network.eval()
                val_loss = 0.0
                
                with torch.no_grad():
                    for batch_data in val_loader:
                        # Move data to device
                        if isinstance(batch_data, list):
                            for item in batch_data:
                                if isinstance(item, dict):
                                    for key in item:
                                        if torch.is_tensor(item[key]):
                                            item[key] = item[key].to(device)
                        else:
                            batch_data = batch_data.to(device)
                        
                        # Get CLIP embeddings for gating
                        images = [x["image"].to(device) for x in batch_data]
                        clip_images = [(x - self.current_model.pixel_mean) / self.current_model.pixel_std for x in images]
                        clip_images_resized = F.interpolate(
                            torch.stack(clip_images), 
                            size=self.current_model.clip_resolution, 
                            mode='bilinear', 
                            align_corners=False
                        )
                        
                        # Get CLIP embeddings for gating network
                        clip_features = self.current_model.sem_seg_head.predictor.clip_model.encode_image(
                            clip_images_resized, dense=True
                        )
                        gating_embeddings = clip_features[:, 0, :]
                        
                        # Get gating weights
                        gating_weights = self.observer.gating_network(gating_embeddings)
                        gating_probs = F.softmax(gating_weights, dim=-1)
                        
                        # Forward pass through model
                        loss_dict = self.current_model(batch_data)
                        losses = sum(loss_dict.values())
                        
                        # Weight the loss by gating probabilities
                        domain_weight = gating_probs[:, domain_idx].mean()
                        weighted_loss = losses * domain_weight
                        
                        val_loss += weighted_loss.item()
                
                # Calculate average losses
                train_loss /= len(train_loader)
                val_loss /= len(val_loader)
                
                # Log results
                print(f"Domain {domain.name} - Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Log to files
                with open(log_file, 'a') as f:
                    f.write(f"Domain {domain.name} - Epoch {epoch+1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}\n")
                
                with open(csv_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([domain.name, epoch + 1, train_loss, val_loss])
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping for domain {domain.name} at epoch {epoch+1}")
                        break
            
            print(f"Completed training on domain: {domain.name}")
        
        print("MoE gating network with LoRA adapters training completed!")
        print(f"Training logs saved to: {log_dir}")

    def _initialize_model_with_adapters(self, domains: list[Domain]) -> None:
        """
        Initialize the model with all LoRA adapters loaded.
        """
        # Load the base model with the first domain's configuration
        first_domain = domains[0]
        self.current_model = load_catseg_model(first_domain.args)
        
        # Load all LoRA adapters
        for domain in domains:
            lora_path = domain.lora_path
            if lora_path and lora_path.exists():
                print(f"Loading LoRA adapter: {lora_path}")
                if hasattr(self.current_model, 'load_adapter'):
                    self.current_model.load_adapter(lora_path, domain.name)
                else:
                    # If not a PEFT model, wrap it
                    self.current_model = peft.PeftModel.from_pretrained(
                        self.current_model, lora_path, domain.name
                    )


class DomainOrchestrator:
    def __init__(
        self,
        domains: list[str],
        lora_db_path: Union[str, Path] = "loradb/",
        embedding_manager: EmbeddingManager = EmbeddingManager(),
    ) -> None:
        
        # TODO: Currently, to use catseg for experiments, we need to change the directory to the catseg directory
        # This can be fixed by refactoring the catseg repo
        parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        catseg_path = os.path.join(parent_dir, "catseg") # TODO: This is a hardcoded path, it should be a parameter
        
        print(f"Changing directory to '{catseg_path}' ...")

        try:
            os.chdir(catseg_path)
        except FileNotFoundError:
            print(f"Error: The specified path '{catseg_path}' does not exist.")
            exit(1)
        except PermissionError:
            print(f"Error: Insufficient permissions to access '{catseg_path}'.")
            exit(1)
        except Exception as e:
            print(f"Unexpected error while changing directory: {e}")
            exit(1)

        self.lora_db_path: Path = Path(lora_db_path)

        self.embedding_manager = embedding_manager

        self.current_model = None

        self.observer: DomainObserver = DomainObserver()

        print("Adding source domains ...")
        
        self._source_domains: Mapping[str, Domain] = self._add_domains(
            domains, split="train"
        )
        print("Source domains added. \n\n")

        print("Adding target domains ...")
        self._target_domains: Mapping[str, Domain] = self._add_domains(
            domains, split="val"
        )
        print("Target domains added. \n\n")

        self._setup_observer()


    def train_moe_gating_network(
        self,
        source_domains: list[str] = None,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        validation_split: float = 0.2,
        hidden_dim: int = 512
    ) -> None:
        """
        Train the MoE gating network using the specified source domains.
        """
        if source_domains is None:
            source_domains = list(self._source_domains.keys())
        
        # Get domain objects
        domains = [self._source_domains[name] for name in source_domains]
        
        # Train the gating network
        self.observer.train_gating_network(
            domains=domains,
            embedding_manager=self.embedding_manager,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_split=validation_split,
            hidden_dim=hidden_dim
        )


    def train_mixture_of_experts_with_adapters(
        self,
        source_domains: list[str],
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 8,
        validation_split: float = 0.2,
        hidden_dim: int = 512,
        log_dir: str = "./moe_training_logs"
    ) -> None:
        """
        Train the MoE gating network jointly with all LoRA adapters using segmentation loss.
        """
        
        # Get domain objects
        # domains = [self._source_domains[name] for name in source_domains]
        # print(f"source domains are {source_domains}")
        # print(f"source domains2 are {self._source_domains}")
        
        # Train the gating network with adapters
        self.observer.train_moe_with_adapters(
            source_domains=source_domains,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            validation_split=validation_split,
            hidden_dim=hidden_dim,
            log_dir=log_dir
        )

    def _benchmark_on_current_target_domain(self, name: str, target_domain: Domain) -> Any:
        print(
            f"Benchmarking {name} on the domain {target_domain.name} ...\n"
        )
        res = benchmark_catseg(self.current_model, target_domain.args)
        return res

    def _set_current_target_domain(
        self,
        target_domain: Domain,
    ) -> None:
        """
        Set the current target domain to the specified domain.
        """
        # We need to load all adapters each time the target domain changes because the same
        # config can't be used across datasets and PEFT does not allow us to change the base
        # model and keep the loaded adapters

        print(f"Setting current target domain to {target_domain.name}.\n")

        self.current_model = None  # This will ensure that the current PEFT model will be initialized using base model with new config
        self._load_adapters(target_domain)

    def _load_adapters(self, target_domain: Domain) -> None:
        """
        Load all adapters for the source domains.
        """
        for source_domain in self._source_domains.values():
            lora_path = source_domain.lora_path
            assert lora_path.exists(), lora_path
            print(f"Loading LoRA: '{lora_path}' ...")
            self._load_lora(target_domain, source_domain, lora_path)
            print(f"LoRA: '{lora_path}' loaded\n\n")

    def _load_lora(self, target_domain: Domain, source_domain: Domain, lora_path: Path) -> None:
        """
        Load the LoRA adapter for the specified domain.
        """
        # print(f"all the args are: {target_domain}, {source_domain}, {lora_path}")
        if self.current_model is None:
            # Wrap the model in PeftModel class the first time an adapter is loaded
            # The base model should be loaded with target domain config to avoid label space mismatch
            self.current_model = peft.PeftModel.from_pretrained(
                load_catseg_model(target_domain.args), lora_path, source_domain.name
            )
        else:
            self.current_model.load_adapter(lora_path, source_domain.name)

    def _add_domains(
        self,
        source_domain_names: list[str],
        split: Literal["train", "val"],
    ) -> Dict[str, Domain]:
        """
        Add the specified domains to the orchestrator.
        """

        source_domains = {}

        for source_domain_name in source_domain_names:
            args, evaluator, data_loader = get_domain_args(source_domain_name, split=split)
            source_domains.update(
                {
                    source_domain_name: self._add_domain(
                        domain_name=source_domain_name,
                        args=args,
                        evaluator=evaluator,
                        data_loader=data_loader,
                    )
                }
            )

        return source_domains


    def _add_domain(
        self,
        domain_name: str,
        args: Namespace,
        evaluator,
        data_loader,
        lora_path: Union[str, Path, None] = None,
    ) -> Domain:
        """Adds a Domain instance to the domains list."""

        train_dataset_path = Path(args.train_dataset_path)
        assert train_dataset_path.exists(), train_dataset_path

        if lora_path is None:
            lora_path = self.lora_db_path / domain_name

        statistics: Dict[str, npt.NDArray] = self.embedding_manager.calculate_statistics(
            domain_name, lora_path, train_dataset_path,
        )

        train_average_embedding: npt.NDArray = statistics[
            "train_average_embedding"
        ]

        domain = Domain(
            domain_name,
            args,
            train_dataset_path=train_dataset_path,
            lora_path=lora_path,
            train_average_embedding=train_average_embedding,
            evaluator=evaluator,
            data_loader=data_loader,
        )

        return domain
    
    def _batch_merge(
        self,
        target_domains: list[str],
        mode: Literal["uniform", "centroid"],
        remove_target_adapter: bool = False,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Merge the source domains and benchmark the merged adapter on the target domains.
        """

        results = {}
        weights = {}

        for current_target_domain_name in target_domains:

            current_target_domain = self._target_domains[current_target_domain_name]
            
            self._set_current_target_domain(
                current_target_domain,
            )

            weight_dict, merged_adpater_name = self._merge(
                current_target_domain,
                remove_target_adapter,
                mode,
                top_k=len(self._source_domains) - 1 if remove_target_adapter else len(self._source_domains)
            )

            weights.update({current_target_domain.name: weight_dict})

            result_dict = self._benchmark_on_current_target_domain(
                name=merged_adpater_name,
                target_domain=current_target_domain
            )

            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }  # Different datasets have different evaluation methods so this won't always work
            )

            # Delete the adapter so we can add another with the same name but different weights (remove unused adapters)
            print(f"Deleting adapter {merged_adpater_name}.")
            self.current_model.delete_adapter(merged_adpater_name)

            print("\n")

        return results, weights


    def _merge(
        self,
        target_domain: Domain,
        remove_target_adapter: bool,
        mode: Literal["uniform", "centroid"],
        target_embedding=None,
        softmax_temperature: Optional[int] = 0.05,
        top_k: int = 5,  # number of domains to merge
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: np.linalg.norm(v1 - v2),
        sort_descending: bool = True

    ) -> tuple[dict[str, float], str]:
        """
        Merge the source domains and benchmark the merged adapter on the target domain.
        """
    
        source_domains = None
        if remove_target_adapter:
            print(f"Removing {target_domain.name} from source domains!")
            source_domains = [
                domain
                for _, domain in self._source_domains.items() if domain.name != target_domain.name
            ]
        else:
            source_domains = [
                domain
                for _, domain in self._source_domains.items()
            ]

        if mode == "uniform":

            weights = [1 / len(source_domains) for _ in range(len(source_domains))]
            domain_weight_mapping = {domain.name: weight for domain, weight in zip(source_domains, weights)}

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.','_')}"
            merged_name += f"_{combination_type}_{target_domain.name}" # Create a unique name for merged adapter so that it does not override existing adapters

            self._merge_adapters(
                merge_domains=[domain.name for domain in source_domains],
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type
            )

        elif mode == "centroid":

            similarity_mapping = self.observer.calculate_similarity_to_domains(
                embedding=target_embedding,
                domains=source_domains,  
                similarity_measure=similarity_measure,
                sort_descending=sort_descending
            )

            k_closest_names = list(similarity_mapping.keys())[: top_k]
            k_closest_similarities = list(similarity_mapping.values())[: top_k]

            print(f"Similarities to {top_k} closest domains: ")
            for n, d in zip(k_closest_names, k_closest_similarities):
                print(f"{n}: {d}", end=", ")
            print("")

            weights = self._calculate_adapter_weights(k_closest_similarities, softmax_temperature)

            domain_weight_mapping = {
                k_closest_name: weight
                for k_closest_name, weight in zip(k_closest_names, weights)
            }

            merged_name = ""
            for n, w in domain_weight_mapping.items():
                merged_name += f"_{n}_{str(w).replace('.','_')}"
            merged_name += f"_{combination_type}_{target_domain.name}" # Create a unique name for merged adapter so that it does not override existing adapters

            self._merge_adapters(
                merge_domains=k_closest_names,
                weights=weights,
                merged_name=merged_name,
                combination_type=combination_type
            )

        print(f"Setting {merged_name} as the active adapter.\n")
        self.current_model.set_adapter(merged_name)

        return domain_weight_mapping, merged_name

    def _merge_adapters(
        self,
        merge_domains: list[str],
        weights: list[float],
        merged_name: str,
        combination_type: str,
    ) -> None:
        """
        Merge the specified adapters with the specified weights.
        """
        
        print(f"Merging domains with weights:")
        for n, w in zip(merge_domains, weights):
            print(f"{n}: {w}", end=", ")
        print("")

        self.current_model.add_weighted_adapter(
            merge_domains,
            weights,
            merged_name,
            combination_type=combination_type,
        )

    def _calculate_adapter_weights(self, similarities:list[float], temperature: float) -> list[float]:
        """
        Calculate the weights for the merged adapter based on the similarities to the source domains.
        """
        weights = softmax(similarities, temperature)
        return weights

    def _setup_observer(self):
        print("Adding domain prototypes to the observer.")
        for domain in self._source_domains.values():
            self.observer.add_domain_prototypes(
                domain=domain, average_embedding=domain.train_average_embedding
            )

    def _get_result_from_dict(self, result_dict: Mapping) -> float:
        res = result_dict["sem_seg"].get("IoU", None)
        if res is None:
            res = result_dict["sem_seg"].get("mIoU")
        return res


    def benchmark_zeroshot(self, target_domains: list[str]) -> dict[str, float]:
        results = {}

        for current_target_domain_name in target_domains:
            current_target_domain = self._target_domains[current_target_domain_name]

            args: Namespace = custom_domain_args(
                config_file=current_target_domain.args.config_file,
                output_path="output/benchmark_zeroshot/",
                num_gpus=1,
                model_path="models/model_final.pth",
            )

            self.current_model = load_catseg_model(
                args, model_path=args.model_path
            )

            result_dict = self._benchmark_on_current_target_domain(name="zeroshot", target_domain=current_target_domain)

            print(f"Zeroshot results for {current_target_domain.name}:")
            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }  # res can look different from dataset to dataset
            )

        return results
    
    def benchmark_oracle(self, target_domains: list[str]) -> dict[str, float]:
        results = {}

        for current_target_domain_name in target_domains:
            
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            self.current_model.set_adapter(current_target_domain.name)

            result_dict = self._benchmark_on_current_target_domain(
                name=current_target_domain.name,
                target_domain=current_target_domain
            )
            print(result_dict)

            result = self._get_result_from_dict(result_dict)

            results.update(
                {
                    current_target_domain.name: result
                }
            )

        return results

    def benchmark_uniform(
        self,
        target_domains: list[str],
        remove_target_adapter: bool,
    ) -> tuple[dict[str, float], dict[str, float]]:
        print(f"Starting uniform merge on domains {target_domains}")

        results, weights = self._batch_merge(
            target_domains=target_domains,
            mode="uniform",
            remove_target_adapter=remove_target_adapter,
        )

        print(f"Finished uniform merge on domains {target_domains}")

        return results, weights

    def benchmark_semla(
        self,
        target_domains: list[str],
        remove_target_adapter: bool = False,
        softmax_temperature: Optional[int] = 0.05,
        top_k: int = 5,  # number of domains to merge
        combination_type: str = "cat",
        similarity_measure: Callable[
            [npt.NDArray, npt.NDArray], np.float64
        ] = lambda v1, v2: 1 / np.linalg.norm(v1 - v2),
        sort_descending: bool = True
    ) -> tuple[dict[str, float], dict[str, float]]:
        
        from detectron2.evaluation import inference_context, SemSegEvaluator
        from contextlib import ExitStack

        results = {}
        weights = {}

        t0 = time.time()

        for current_target_domain_name in target_domains:
            
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            data_loader = current_target_domain.data_loader
            evaluator = current_target_domain.evaluator

            model = self.current_model

            # These lines are adopted from
            # https://github.com/facebookresearch/detectron2/blob/2a420edb307c9bdf640f036d3b196bed474b8593/detectron2/evaluation/evaluator.py#L103

            evaluator.reset()

            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                for _, inputs in enumerate(data_loader):

                    input_path = inputs[0]["file_name"]

                    print(f"Predicting image: {input_path}")

                    current_embedding = self.embedding_manager.embed_image(input_path)

                    weight_dict, merged_adpater_name = self._merge(
                        target_domain=current_target_domain,
                        remove_target_adapter=remove_target_adapter,
                        mode="centroid", 
                        target_embedding=current_embedding,
                        softmax_temperature=softmax_temperature,
                        top_k=top_k,
                        combination_type=combination_type,
                        similarity_measure=similarity_measure,
                        sort_descending=sort_descending,
                    )

                    for domain, weight in weight_dict.items():
                        weights.setdefault(domain, []).append(weight)

                    model = self.current_model

                    outputs = model(inputs)

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    if isinstance(evaluator, SemSegEvaluator):
                        _ = evaluator.process(inputs, outputs)
                    else:
                        _ = evaluator.process_image(inputs, outputs)

                    self.current_model.delete_adapter(merged_adpater_name)

            print(f"Benchmarking on domain '{current_target_domain.name}' ...")
            result_dict = evaluator.evaluate()
            result = self._get_result_from_dict(result_dict)
            print(f"Result for domain '{current_target_domain.name}': {result}\n")

            results.update({current_target_domain.name: result})

            if not isinstance(evaluator, SemSegEvaluator):
                evaluator._working_dir.cleanup()

        total = time.time() - t0
        print(f"Experiment took {total} seconds to complete!")

        return results, weights

    def benchmark_semla_moe(
        self,
        target_domains: list[str],
        remove_target_adapter: bool = False,
        top_k: int = 5,
        combination_type: str = "cat",
        log_dir: str = "./moe_weight_distribution"
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Benchmark using MoE gating network for adapter selection.
        """
        from detectron2.evaluation import inference_context, SemSegEvaluator
        from contextlib import ExitStack

        results = {}
        weights = {}

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Create a single log file for all images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"moe_weight_distribution_{timestamp}.csv")
        
        # Initialize CSV logging
        with open(log_file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'domain_rank', 'domain_name', 'weight', 'k_values'])
        
        t0 = time.time()

        for current_target_domain_name in target_domains:
            
            current_target_domain = self._target_domains[current_target_domain_name]

            self._set_current_target_domain(
                current_target_domain,
            )

            data_loader = current_target_domain.data_loader
            evaluator = current_target_domain.evaluator

            model = self.current_model

            evaluator.reset()

            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                # Open the log file for this target domain
                with open(log_file_path, 'a', newline='') as log_file:
                    for _, inputs in enumerate(data_loader):

                        input_path = inputs[0]["file_name"]
                        filename = os.path.basename(input_path)  # Extract just the filename

                        print(f"Predicting image: {input_path}")

                        current_embedding = self.embedding_manager.embed_image(input_path)

                        weight_dict, merged_adpater_name = self._merge_moe(
                            target_domain=current_target_domain,
                            remove_target_adapter=remove_target_adapter,
                            target_embedding=current_embedding,
                            top_k=top_k,
                            combination_type=combination_type,
                            filename=filename,
                            log_file_handle=log_file
                        )

                        for domain, weight in weight_dict.items():
                            weights.setdefault(domain, []).append(weight)

                        model = self.current_model

                        outputs = model(inputs)

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()

                        if isinstance(evaluator, SemSegEvaluator):
                            _ = evaluator.process(inputs, outputs)
                        else:
                            _ = evaluator.process_image(inputs, outputs)

                        self.current_model.delete_adapter(merged_adpater_name)

            print(f"Benchmarking on domain '{current_target_domain.name}' ...")
            result_dict = evaluator.evaluate()
            result = self._get_result_from_dict(result_dict)
            print(f"Result for domain '{current_target_domain.name}': {result}\n")

            results.update({current_target_domain.name: result})

            if not isinstance(evaluator, SemSegEvaluator):
                evaluator._working_dir.cleanup()

        total = time.time() - t0
        print(f"SemLA MoE Experiment took {total} seconds to complete!")
        print(f"Weight distribution logs saved to: {log_file_path}")

        return results, weights

    def _merge_moe(
        self,
        target_domain: Domain,
        remove_target_adapter: bool,
        target_embedding: npt.NDArray,
        top_k: int = 5,
        combination_type: str = "cat",
        filename: str = None,
        log_file_handle = None
    ) -> tuple[dict[str, float], str]:
        """
        Merge adapters using MoE gating network.
        """
        source_domains = None
        if remove_target_adapter:
            print(f"Removing {target_domain.name} from source domains!")
            source_domains = [
                domain
                for _, domain in self._source_domains.items() if domain.name != target_domain.name
            ]
        else:
            source_domains = [
                domain
                for _, domain in self._source_domains.items()
            ]

        # Use MoE gating network to get domain weights with logging
        domain_weight_mapping = self.observer.calculate_moe_similarity_to_domains(
            embedding=target_embedding,
            domains=source_domains,
            top_k=top_k,
            filename=filename,
            log_file_handle=log_file_handle
        )

        # Get the selected domains and their weights
        selected_domains = list(domain_weight_mapping.keys())
        weights = list(domain_weight_mapping.values())

        print(f"MoE selected domains: {selected_domains}")
        print(f"MoE weights: {weights}")

        merged_name = ""
        for n, w in domain_weight_mapping.items():
            merged_name += f"_{n}_{str(w).replace('.','_')}"
        merged_name += f"_{combination_type}_{target_domain.name}"

        self._merge_adapters(
            merge_domains=selected_domains,
            weights=weights,
            merged_name=merged_name,
            combination_type=combination_type
        )

        print(f"Setting {merged_name} as the active adapter.\n")
        self.current_model.set_adapter(merged_name)

        return domain_weight_mapping, merged_name
