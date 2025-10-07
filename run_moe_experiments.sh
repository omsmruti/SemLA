#!/bin/bash

# MoE SemLA Experiments
# This script runs the MoE-based SemLA experiments

echo "Running MoE-based SemLA experiments..."

# MoE SemLA Benchmark
echo "Running MoE SemLA benchmark..."
python experiments.py --experiment semla_moe \
    --source_domains config/source_domains.yaml \
    --target_domains config/target_domains.yaml \
    --semla_config config/semla_moe_config.yaml \
    --remove_target_adapter \
    --output_dir ./results/semla_moe

echo "MoE experiments completed!"
