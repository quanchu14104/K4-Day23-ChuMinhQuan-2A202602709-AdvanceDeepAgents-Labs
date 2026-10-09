# Survey of Reinforcement Learning for Large Language Model Reasoning

**Học viên:** Chu Minh Quân | **MSSV:** 2A202602709

## TL;DR

This survey explores the foundational concepts, core algorithms, and empirical benchmarks of reinforcement learning (RL) for enhancing large language model (LLM) reasoning capabilities. It examines key methodologies, training paradigms, and representative systems, highlighting advancements and current limitations in the field.

## Background & Motivation

Reinforcement learning has emerged as a critical methodology for advancing large language model (LLM) reasoning capabilities. Unlike traditional supervised learning, which relies on pre-labeled data, RL enables models to learn through interaction and feedback, making it particularly effective for complex reasoning tasks. The integration of RL with LLMs has shown promise in improving their ability to perform logical reasoning, problem-solving, and decision-making across various domains. This survey aims to provide a comprehensive overview of the current state of research in this area, highlighting key methodologies, training paradigms, and representative systems, while also identifying current limitations and future research directions.

## Core Methodologies and Training Paradigms

### Reinforcement Learning Frameworks

Several reinforcement learning frameworks have been developed to enhance LLM reasoning capabilities. Group Relative Policy Optimization (GRPO) [1] and its variants, such as Dr. GRPO [1], have been widely used to optimize policy learning in LLMs. These frameworks address the challenges of response-length bias and improve training efficiency by skipping uninformative prompts without reducing model performance [1].

### Training Paradigms

The training of LLMs using reinforcement learning involves several key paradigms. One notable approach is the use of verifiable rewards (RLVR) [2], which replaces learned reward models with programmatic verifiers (e.g., answer checkers or unit tests). This method ensures greater reliability and resistance to manipulation by making the system resistant to exploitation [2].

### Sampling Criteria and Exploration Strategies

Improving the sampling criteria and exploration strategies in reinforcement learning for LLM reasoning is crucial for enhancing model performance. Techniques such as Competence-Difficulty Alignment Sampling (CDAS) [3] and Generative Probabilities-based Exploration (ARM) [4] have been proposed to address low sample efficiency and entropy collapse, respectively. These methods improve both accuracy and efficiency in mathematical benchmarks and equilibrate confidence levels, leading to more robust and reliable reasoning capabilities in LLMs.

## Empirical Benchmarks and Representative Systems

### Key Benchmarks

Several empirical benchmarks have been established to evaluate the performance of LLMs trained with reinforcement learning. These benchmarks include mathematical reasoning tasks, coding competitions, and STEM fields, where LLMs have shown superior performance compared to those trained through conventional supervised learning on human demonstrations [5].

### Representative Systems

Several representative systems have been developed to showcase the advancements in reinforcement learning for LLM reasoning. DeepSeek-R1 [5] and DeepSeek-R1-Zero [5] are notable examples, demonstrating the effectiveness of RL in incentivizing reasoning in LLMs. These systems exhibit advanced reasoning patterns such as self-reflection, verification, and dynamic strategy adaptation, showcasing the potential of RL in enhancing LLM reasoning capabilities.

## Trends and Open Problems

### Trends

Recent trends in reinforcement learning for LLM reasoning include the development of more sophisticated reward systems, the integration of external search systems, and the use of modular frameworks to enhance reasoning capabilities. These trends highlight the ongoing efforts to improve the performance and reliability of LLMs in complex reasoning tasks.

### Open Problems

Despite the advancements, several open problems remain in the field of reinforcement learning for LLM reasoning. These include the need for better evaluation standards, the challenge of addressing issues like poor readability and language mixing, and the development of more efficient and scalable RL algorithms. Additionally, the integration of RL with other emerging technologies, such as neural reward models and process reward models, presents both opportunities and challenges for future research.

## References
[1] The State of Reinforcement Learning for LLM Reasoning. web. https://magazine.sebastianraschka.com/p/the-state-of-llm-reasoning-model-training (2025-04-19)
[2] A Survey of Reinforcement Learning for Large Reasoning Models. arxiv. https://arxiv.org/abs/2509.08827 (2025-09-08)
[3] Rethinking the Sampling Criteria in Reinforcement Learning for LLM Reasoning: A Competence-Difficulty Alignment Perspective. hf-search. https://huggingface.co/papers/2505.17652 (2025-05-23)
[4] Back to Basics: Revisiting Exploration in Reinforcement Learning for LLM Reasoning via Generative Probabilities. hf-search. https://huggingface.co/papers/2602.05281 (2026-02-05)
[5] DeepSeek-R1 incentivizes reasoning in LLMs through ... - Nature. web. https://www.nature.com/articles/s41586-025-09422-z (2025-09-17)
