#!/usr/bin/env python3
"""
Training script for MoE gating network.
This script trains the Mixture of Experts gating network using source domain data.
"""

import argparse
import yaml
import os
from pathlib import Path

from domain_orchestrator.domain_orchestrator import DomainOrchestrator

def load_domains_from_yaml(file_path: str) -> list[str]:
    """Load domains from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Train MoE gating network")
    
    # Required arguments
    parser.add_argument("--source_domains", type=str, required=True,
                        help="Path to YAML file containing source domains")
    parser.add_argument("--output_dir", type=str, default="./moe_results",
                        help="Directory to save training results")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--validation_split", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    
    # MoE specific parameters
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for the gating network")
    
    args = parser.parse_args()
    
    # Load source domains
    source_domains = load_domains_from_yaml(args.source_domains)
    print(f"Training MoE gating network on {len(source_domains)} source domains: {source_domains}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize domain orchestrator
    orchestrator = DomainOrchestrator(domains=source_domains)
    
    # Train the MoE gating network
    print("Starting MoE gating network training...")
    # orchestrator.train_moe_gating_network(
    #     source_domains=source_domains,
    #     epochs=args.epochs,
    #     learning_rate=args.learning_rate,
    #     batch_size=args.batch_size,
    #     validation_split=args.validation_split
    # )
    orchestrator.train_mixture_of_experts_with_adapters(
        source_domains=source_domains,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        validation_split=args.validation_split,
        hidden_dim=args.hidden_dim,
        log_dir=args.output_dir
    )
    
    print(f"MoE gating network training completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
