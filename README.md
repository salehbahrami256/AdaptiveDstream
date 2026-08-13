# Adaptive D-Stream v0

Minimal research prototype for adaptive multi-resolution density-based clustering of evolving data streams.

## Implemented

- one-pass online updates;
- exponential fading / time-decayed statistics;
- hierarchical adaptive grid;
- refinement from local variance + mean offset;
- equal redistribution of historical statistics after a split;
- sparse-cell pruning;
- dense / transitional / sparse states;
- connected-component clustering over face-adjacent dense cells;
- synthetic 2D drifting stream;
- visualization snapshots.

## v0 approximation

When a `d`-dimensional parent is split, it creates `2**d` children. Because historical raw observations are unavailable, the prototype distributes the parent decayed mass equally among the children and initializes each child's first/second moments using a uniform-within-child approximation.

This is intentionally simple. Later we can compare it against moment-based redistribution, a bounded reservoir, and selective-dimensional refinement.

## Install

```bash
pip install -e .
```

## Run

```bash
python examples/run_2d_drift.py
```

Snapshots are written to `outputs/` at t = 1000, 2000, and 3000.

## Deliberate limitations

- dense-cell adjacency is currently O(m^2);
- refinement splits every dimension, hence creates 2**d children;
- no contraction/merge mechanism yet;
- no ARI/NMI evaluation yet;
- no LSH / ANN indexing yet;
- no production-scale optimization.

The first goal is to make the 2D behavior correct and visually inspectable before optimizing.
