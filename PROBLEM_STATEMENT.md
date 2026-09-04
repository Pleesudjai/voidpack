# Problem Statement, ASU AIR SPARK 2026 Fall

**Team** SMC Labs, Mobasher Group, Arizona State University
**Date** 2026-09-03
**Status** Sections 2, 3 and 4 submitted on the AIR SPARK progress form 2026-09-03.
Sections 1, 5, 6 and 7 are open.

---

## 1. Problem statement

E-commerce orders are packed without any measurement of how well the items fill the
box. The unused void space is paid for twice, once in packaging material and again in
shipped air. Filling a box with items of mixed size and shape is a packing problem. It
belongs to the same class of problem that governs how aggregate particles fill a
concrete mix. We are building a pipeline that carries the particle packing optimization
developed in our concrete research into the packaging of an e-commerce order.

## 2. Project description (SUBMITTED 2026-09-03)

We are building a pipeline that helps e-commerce companies optimize packaging by
applying concrete particle packing optimization to box and void space.

*(22 words, within the 30-word limit.)*

## 3. Breakthrough (SUBMITTED 2026-09-03)

We realized that the algorithm we use to optimize concrete particle packing in our
research can be applied to a real-world problem from a different perspective. We paired
it with a LiDAR program and a set of tools that generate the optimization space between
particles.

## 4. Challenges and blockers (SUBMITTED 2026-09-03)

The LiDAR data we capture from the iPhone is not high enough quality. We have to smooth
it and discretize more data before it is usable. Until the point cloud is clean, the
optimizer cannot be fed a reliable particle geometry.

## 5. Pipeline, as currently defined

| Stage | Input | Output | Owner |
|---|---|---|---|
| 1. Capture | Physical item | iPhone LiDAR point cloud | TBD |
| 2. Clean | Raw point cloud | Smoothed and discretized geometry | TBD |
| 3. Represent | Cleaned geometry | Particle set for the optimizer | TBD |
| 4. Optimize | Particle set plus box boundary | Packing arrangement | TBD |
| 5. Report | Packing arrangement | Void fraction and box size | TBD |

## 6. Proposed scope for today

These four items follow directly from the blocker named in Section 4. They are a
proposal and need the team to confirm them before work starts.

1. Fix the input. Smooth the iPhone LiDAR point cloud and discretize it into the
   particle representation the optimizer accepts.
2. Define the optimization space. Confirm what the tools generate between particles and
   how the box boundary is imposed.
3. Run the packing optimizer on one real order of items end to end.
4. Report the void fraction before and after optimization for that one order.

## 7. Open items to confirm with the team

- What is the objective function, minimum box volume, minimum void fraction, or minimum
  shipping cost.
- Which capture app produces the point cloud, and what is the raw point spacing it
  delivers.
- Which packing routine from the research code is being ported, and does it run in the
  same language in this pipeline.
- What is the demo deliverable at the end of the hackathon.

---

*Sections 2, 3 and 4 were submitted on the AIR SPARK progress form on 2026-09-03 and
are locked. Do not edit them, the file and the form must stay identical. Sections 1, 5, 6
and 7 are drafted from that text and carry no result that has been measured.*
