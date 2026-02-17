#!/bin/bash
# Run script for Wavefunction Visualization

export RUST_LOG=info

if [ "$1" == "--benchmark" ]; then
    echo "Running quick runtime benchmark mode..."
    cargo run --release -- --benchmark
    exit 0
fi

if [ "$1" == "--benchmark-soak-30m" ]; then
    echo "Running timed soak benchmark mode (default 30 minutes)..."
    cargo run --release -- --benchmark-soak-30m
    exit 0
fi

if [ "$1" == "--benchmark-cap-pair-sweep" ]; then
    echo "Running coupled cap pair sweep benchmark mode..."
    cargo run --release -- --benchmark-cap-pair-sweep
    exit 0
fi

echo "Running application (native Metal renderer)..."
cargo run --release
