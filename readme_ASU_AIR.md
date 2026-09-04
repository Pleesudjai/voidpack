# ASU AIR SPARK 2026 Fall - Packaging Optimization

## Problem Statement

E-commerce orders are packed without any measurement of how well the items fill the box. The unused void space is paid for twice, once in packaging material and again in shipped air. Filling a box with items of mixed size and shape is a packing problem that belongs to the same class of problem that governs how aggregate particles fill a concrete mix.

## Project Description

We are building a pipeline that helps e-commerce companies optimize packaging by applying concrete particle packing optimization to box and void space.

## Breakthrough

We realized that the algorithm we use to optimize concrete particle packing in our research can be applied to a real-world problem from a different perspective. We paired it with a LiDAR program and a set of tools that generate the optimization space between particles.

## Challenges and Blockers

The LiDAR data we capture from the iPhone is not high enough quality. We have to smooth it and discretize more data before it is usable. Until the point cloud is clean, the optimizer cannot be fed a reliable particle geometry.

## Pipeline Stages

| Stage | Input | Output | Status |
|---|---|---|---|
| 1. Capture | Physical item | iPhone LiDAR point cloud | To Do |
| 2. Clean | Raw point cloud | Smoothed and discretized geometry | To Do |
| 3. Represent | Cleaned geometry | Particle set for the optimizer | To Do |
| 4. Optimize | Particle set plus box boundary | Packing arrangement | To Do |
| 5. Report | Packing arrangement | Void fraction and box size | To Do |

## Proposed Scope for Today

1. Fix the input: Smooth the iPhone LiDAR point cloud and discretize it into the particle representation the optimizer accepts.
2. Define the optimization space: Confirm what the tools generate between particles and how the box boundary is imposed.
3. Run the packing optimizer on one real order of items end to end.
4. Report the void fraction before and after optimization for that one order.

## Open Items to Confirm with Team

- Objective function: minimum box volume, minimum void fraction, or minimum shipping cost
- Capture app and raw point spacing specifications
- Packing routine selection from research code
- Demo deliverable requirements

## Current Status

- Sections 2, 3, and 4 submitted on AIR SPARK progress form (2026-09-03)
- Sections 1, 5, 6, and 7 are open for development
- No existing code or data files in workspace

## Next Steps

1. Team to confirm scope items in Section 6
2. Clarify open items in Section 7
3. Begin pipeline implementation starting with data processing stage