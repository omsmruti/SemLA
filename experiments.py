import argparse
import json
import os
import yaml
import scipy
import scipy.spatial
from typing import Dict, List, Callable, Any, Optional, Tuple
import itertools


from domain_orchestrator.domain_orchestrator import DomainOrchestrator

# Define distance measure mappings
NAME_MEASURE_MAPPING = {
    "euclidean": lambda u, v: 1. / scipy.spatial.distance.euclidean(u.squeeze(), v.squeeze()),
    "cosine": lambda u, v: scipy.spatial.distance.cosine(u.squeeze(), v.squeeze()),
}

def load_domains_from_yaml(file_path: str) -> List[str]:
    """Load domains from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def load_config_from_yaml(file_path: str) -> Dict[str, Any]:
    """Load configuration parameters from a YAML file."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_results(results: Dict, weights: Optional[Dict] = None, output_dir: str = "./results") -> None:
    """Save results and weights to JSON files."""
    
    # Change the current working directory to the root directory
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    print(f"Changing current working directory to {root_dir}")
    os.chdir(root_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    if weights is not None:
        with open(os.path.join(output_dir, "weights.json"), 'w') as f:
            json.dump(weights, f, indent=4)
    
    print(f"Results saved to {output_dir}")

def benchmark_zeroshot(source_domains: List[str], target_domains: List[str], 
                      output_dir: str) -> None:
    """Run zero-shot benchmark experiment."""
    orchestrator = DomainOrchestrator(source_domains)
    results = orchestrator.benchmark_zeroshot(target_domains)
    save_results(results, output_dir=output_dir)

def benchmark_oracle(source_domains: List[str], target_domains: List[str], 
                    output_dir: str) -> None:
    """Run oracle benchmark experiment."""
    orchestrator = DomainOrchestrator(domains=source_domains)
    results = orchestrator.benchmark_oracle(target_domains=target_domains)
    save_results(results, output_dir=output_dir)

def uniform_merge(source_domains: List[str], target_domains: List[str], 
                 remove_target_adapter: bool, output_dir: str) -> None:
    """Run uniform merge experiment."""
    orchestrator = DomainOrchestrator(domains=source_domains)
    results, weights = orchestrator.benchmark_uniform(
        target_domains=target_domains,
        remove_target_adapter=remove_target_adapter,
    )
    save_results(results, weights, output_dir=output_dir)

def semla_merge(source_domains: List[str], target_domains: List[str], 
                config: Dict[str, Any], remove_target_adapter: bool, 
                output_dir: str) -> None:
    """Run online merge experiment."""
    similarity_measure_name = config.get("similarity_measure_name", "euclidean")
    temperature = config.get("temperature", 0.05)
    top_k = config.get("top_k", 5)
    combination_type = config.get("combination_type", "cat")
    
    orchestrator = DomainOrchestrator(source_domains)
    results, weights = orchestrator.benchmark_semla(
        target_domains=target_domains,
        remove_target_adapter=remove_target_adapter,
        similarity_measure=NAME_MEASURE_MAPPING[similarity_measure_name],
        softmax_temperature=temperature,
        top_k=top_k,
        combination_type=combination_type
    )
    save_results(results, weights, output_dir=output_dir)

# TODO: maybe train the gating network elsewhere instead of in the benchmark step
def semla_moe_merge(source_domains: List[str], target_domains: List[str], 
                   config: Dict[str, Any], remove_target_adapter: bool, 
                   output_dir: str) -> None:
    """Run MoE-based merge experiment."""
    top_k = config.get("top_k", 5)
    combination_type = config.get("combination_type", "cat")
    
    orchestrator = DomainOrchestrator(source_domains)
    
    # Create log directory for MoE weight distribution
    moe_log_dir = os.path.join(output_dir, "moe_weight_logs")
    
    # Train gating network with LoRA adapters for each target domain separately
    results = {}
    weights = {}
    
    for target_domain in target_domains:
        print(f"Training MoE gating network for target domain: {target_domain}")
        
        # Remove target domain from source domains for training
        if remove_target_adapter:
            training_domains = [d for d in source_domains if d != target_domain]
        else:
            training_domains = source_domains
        
        # Train gating network with filtered domains
        print(f"Training gating network with domains: {training_domains}")
        orchestrator.train_moe_gating_network(
            source_domains=training_domains,
            epochs=config.get("epochs", 100),
            learning_rate=config.get("learning_rate", 0.001),
            batch_size=config.get("batch_size", 32),
            validation_split=config.get("validation_split", 0.2)
        )
        #orchestrator.train_mixture_of_experts_with_adapters(
        #    source_domains=training_domains,
        #    epochs=config.get("epochs", 100),
        #    learning_rate=config.get("learning_rate", 0.001),
        #    batch_size=config.get("batch_size", 8),  # Use smaller batch size
        #    validation_split=config.get("validation_split", 0.2),
        #    hidden_dim=config.get("hidden_dim", 512),
        #    log_dir=os.path.join(output_dir, f"moe_training_{target_domain}")
        #)
        
        # Run benchmark for this target domain
        target_results, target_weights = orchestrator.benchmark_semla_moe(
            target_domains=[target_domain],
            remove_target_adapter=remove_target_adapter,
            top_k=top_k,
            combination_type=combination_type,
            log_dir=moe_log_dir
        )
        
        results.update(target_results)
        weights.update(target_weights)
    
    save_results(results, weights, output_dir=output_dir)
    print(f"MoE weight distribution logs saved to: {moe_log_dir}")


def generate_weight_combinations(domains: List[str], weights_list: List[float]) -> List[Dict[str, float]]:
    """Generate all possible weight combinations that sum to 1."""
    combinations = []
    
    # Generate all possible combinations for first (length-1) domains
    for weights in itertools.product(weights_list, repeat=len(domains)-1):
        # Calculate the final weight as 1 - sum of selected weights
        final_weight = 1 - sum(weights)
        
        # Only include if final weight is non-negative
        if final_weight >= 0:
            # Create the complete weight list
            complete_weights = list(weights) + [final_weight]
            combination = dict(zip(domains, complete_weights))
            combinations.append(combination)
    
    print(f"Generated {len(combinations)} weight combinations")
    print(f"Weight combinations are ------------------> {combinations}")

    return combinations


def magic_weights_merge(source_domains: List[str], target_domains: List[str], 
                       weights_list: List[float],
                       remove_target_adapter: bool, output_dir: str) -> None:
    """Run magic weights experiment - try different weight combinations and select best per image."""
    
    orchestrator = DomainOrchestrator(domains=source_domains)
    
    results = {}
    best_weights_per_image = {}
    
    for target_domain in target_domains:
        print(f"Testing magic weights on target domain: {target_domain}")
        
        # Generate weight combinations excluding the target domain
        if remove_target_adapter:
            # Remove target domain from source domains when generating combinations
            filtered_source_domains = [domain for domain in source_domains if domain != target_domain]
            weight_combinations = generate_weight_combinations(filtered_source_domains, weights_list)
        else:
            # Use all source domains
            weight_combinations = generate_weight_combinations(source_domains, weights_list)
        
        print(f"Generated {len(weight_combinations)} weight combinations for target domain {target_domain}")
        
        # Get the best results and weights for this target domain
        target_results, target_weights = orchestrator.benchmark_magic_weights(
            target_domains=[target_domain],
            weight_combinations=weight_combinations,
            remove_target_adapter=remove_target_adapter
        )
        
        results.update(target_results)
        best_weights_per_image.update(target_weights)
    
    save_results(results, best_weights_per_image, output_dir=output_dir)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Domain adaptation experiments")
    
    # Required arguments
    parser.add_argument("--experiment", type=str, required=True, 
                        choices=["zeroshot", "oracle", "uniform", "semla", "semla_moe", "magic_weights"],
                        help="Type of experiment to run")
    
    # Optional arguments with defaults
    parser.add_argument("--source_domains", type=str, 
                        help="Path to YAML file containing source domains")
    parser.add_argument("--target_domains", type=str, 
                        help="Path to YAML file containing target domains")
    parser.add_argument("--semla_config", type=str, 
                        help="Path to YAML file containing configuration parameters")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save results")
    parser.add_argument("--remove_target_adapter", action="store_true", 
                        help="Whether to remove target adapter")
    
    return parser.parse_args()

def main():
    """Main function to run experiments based on command line arguments."""

    args = parse_args()
    
    # Load source domains
    source_domains = load_domains_from_yaml(args.source_domains) if args.source_domains else []
    
    # Load target domains
    target_domains = load_domains_from_yaml(args.target_domains) if args.target_domains else []
    
    # Load config if provided
    semla_config = load_config_from_yaml(args.semla_config) if args.semla_config else {}
    
    # Run the specified experiment
    if args.experiment == "zeroshot":
        benchmark_zeroshot(source_domains, target_domains, args.output_dir)
    elif args.experiment == "oracle":
        benchmark_oracle(source_domains, target_domains, args.output_dir)
    elif args.experiment == "uniform":
        uniform_merge(source_domains, target_domains, args.remove_target_adapter, args.output_dir)
    elif args.experiment == "semla":
        semla_merge(source_domains, target_domains, semla_config, args.remove_target_adapter, args.output_dir)
    elif args.experiment == "semla_moe":
        semla_moe_merge(source_domains, target_domains, semla_config, args.remove_target_adapter, args.output_dir)
    elif args.experiment == "magic_weights":
        weights_list = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.45, 
                        0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.95]
        magic_weights_merge(source_domains, target_domains, weights_list, args.remove_target_adapter, args.output_dir)

if __name__ == "__main__":
    main()
