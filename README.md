# Fine-Tuning DeepSeek-R1-Distill-Qwen-1.5B with LoRA 🚀

This project demonstrates an end-to-end process for fine-tuning the DeepSeek-R1-Distill-Qwen-1.5B language model using modern techniques to optimize performance and efficiency. 🤖✨

## Overview 🌟
The pipeline adapts a pre-trained large language model to a custom task using a curated dataset of 100 math question-and-answer pairs. By leveraging 8-bit quantization and Low-Rank Adaptation (LoRA), the fine-tuning process becomes both memory-efficient and computationally lighter while still delivering great performance. ⚡️

## Installation and Environment Setup 🛠️
The workflow starts by installing the necessary Python packages for data handling, model quantization, experiment tracking, and fine-tuning. Key libraries include Hugging Face's Transformers, Datasets, and Weights & Biases (wandb), which help streamline the training process and monitor performance in real time. 📦💻

## Model and Tokenizer Initialization 🤖
The DeepSeek-R1-Distill-Qwen-1.5B model is loaded along with its tokenizer using the Transformers library. This foundational step ensures the model is ready for further modifications and is set to run on the GPU for optimal performance. 🔥

## Dataset Preparation 📚
A custom dataset of 100 Q&A pairs (covering various arithmetic and algebra problems) is created and stored in JSONL format. The data is then loaded and split into training and evaluation sets, ensuring the model is properly assessed during fine-tuning. 📝✅

## Experiment Tracking 📊
Integration with Weights & Biases (wandb) allows for comprehensive experiment tracking. Important parameters—such as learning rate, model architecture details, dataset information, and training epochs—are logged. This real-time monitoring of training metrics simplifies hyperparameter tuning and performance analysis. 🔍📈

## Tokenization Strategy 🗣️
A thoughtful tokenization strategy is used to efficiently process the dataset. Questions and answers are combined into a single text sequence, tokenized with truncation and padding, ensuring consistent input lengths and proper label alignment for supervised learning. 🧩📏

## Memory Efficiency and LoRA Integration 💡
To address high memory requirements, 8-bit quantization is applied, significantly reducing the model’s memory footprint without compromising accuracy. Additionally, LoRA is used for parameter-efficient fine-tuning, updating only a small subset of model parameters and reducing computational demands. 🚀🔋

## Training Configuration 🎯
The fine-tuning process is managed via Hugging Face's Trainer API, which orchestrates training with parameters such as the number of epochs, batch size, gradient accumulation steps, and mixed precision (fp16). Evaluation steps are interleaved with training to continuously monitor performance on unseen data. 📆✅

## Conclusion 🎉
This project provides a clear roadmap for fine-tuning a large-scale language model using state-of-the-art techniques. Emphasizing efficiency through quantization and LoRA, the methodology is adaptable to various custom tasks, making it feasible to fine-tune advanced models even with limited computational resources. 🌐🚀

Feel free to explore, modify, and extend this workflow for your own fine-tuning projects. Happy coding! 😄👩‍💻👨‍💻
