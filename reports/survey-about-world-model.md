# World Models: A Comprehensive Survey

**Học viên:** Chu Minh Quân | **MSSV:** 2A202602709

## TL;DR
World models represent a transformative approach in artificial intelligence, enabling agents to simulate and predict environmental dynamics. This survey explores their foundational concepts, core algorithms, key systems, and current limitations, highlighting their potential to revolutionize fields like robotics, autonomous driving, and interactive entertainment.

## Background & Motivation
World models are generative neural network models that learn compressed representations of environments, allowing agents to simulate and predict how environments change over time in response to actions [1][2]. These models are divided into a Variational Autoencoder (VAE) and a Mixture Density Network (MDN) combined with a Recurrent Neural Network (RNN), which compress spatial and temporal representations of the environment and predict future states as a probability density function [1][3]. The controller model is kept simple and small to focus on the credit assignment problem, while the world model handles complexity [1][4][3].

## Core Algorithms and Methodologies
World models leverage advanced algorithms such as Variational Autoencoders (VAEs) and Mixture Density Networks (MDNs) combined with Recurrent Neural Networks (RNNs) to compress and predict environmental dynamics [1][3]. These models are trained using procedures that include offline geometric calibration and inconsistency-robust periodic 4DGS to ensure action-faithful and physically plausible video prediction [5]. The use of deep algorithm unrolling (DAU) for time-varying graph signal restoration further enhances the model's ability to handle complex spatiotemporal data [6].

## Key Systems and Applications
Several key systems have been developed to leverage world models, including DreamTrue for action-faithful robot world modeling [5], OuroWorld for dynamic 3D cinemagraph generation [7], and WorldGuide for procedural task execution [8]. These systems address challenges in action following, interaction coverage, and goal-directed video generation, demonstrating the versatility of world models in various applications [5][7][8].

## Empirical Benchmarks and Capabilities
Empirical benchmarks such as WorldTest and AutumnBench have been developed to evaluate world model capabilities, revealing that humans outperform current reasoning models in world-model learning [9]. WBench, an interactive world model benchmark, evaluates 43 models with 22 metrics across 5 dimensions, highlighting the need for fundamentally different internal representations for navigation and semantic interaction [10]. Additionally, studies have shown that vision-language models exhibit limitations in basic world modeling abilities, suggesting areas for improvement in 3D perception, temporal and motion understanding, cause-and-effect understanding, and transitive/compositional world modeling [11].

## Trends and Open Problems
World models are at the forefront of advancements in artificial intelligence, with ongoing research focusing on improving action-faithful prediction, dynamic scene generation, and procedural task execution [5][7][8]. However, significant challenges remain, including the uneven improvement in agent and object-interaction fidelity, and the need for more sophisticated internal representations to handle complex tasks [12][9]. Future research should aim to address these limitations and explore new applications of world models in various domains.

## References
[1] World Models. hf-search. https://huggingface.co/papers/1803.10122 (2018-03-27)
[2] World Models. web. https://worldmodels.github.io/ (2018-03-27)
[3] DriveDreamer: Towards Real-world-driven World Models for Autonomous Driving. hf-search. https://huggingface.co/papers/2309.09777 (2023-09-18)
[4] World Models for Math Story Problems. hf-search. https://huggingface.co/papers/2306.04347 (2023-06-07)
[5] DreamTrue: Action-Faithful Robot World Model with Counterfactual Post-Training. arxiv. https://arxiv.org/abs/2610.12468 (2026-10-08)
[6] Unrolled Time-Varying Graph Signal Restoration under Spatiotemporal Smoothness Priors. arxiv. https://arxiv.org/abs/2610.11359 (2026-10-08)
[7] What 30,000 Hours of Ego-centric Video Does Not Teach. arxiv. https://arxiv.org/abs/2610.12464 (2026-10-08)
[8] World Models. arxiv. https://arxiv.org/abs/2610.10932 (2026-10-07)
[9] Benchmarking World-Model Learning. web. https://arxiv.org/html/2510.19788v2 (N/A)
[10] WBench - Interactive World Model Benchmark. web. https://meituan-longcat.github.io/WBench/ (N/A)
[11] Do Vision-Language Models Have Internal World Models? Towards an Atomic Evaluation. web. https://p.rst.im/q/aclanthology.org/2025.findings-acl.1342.pdf (N/A)
[12] WorldDreamer: Towards General World Models for Video Generation via Predicting Masked Tokens. hf-search. https://huggingface.co/papers/2401.09985 (2024-01-18)
