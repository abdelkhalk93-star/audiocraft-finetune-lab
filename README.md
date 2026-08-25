![preview](https://raw.githubusercontent.com/abdelkhalk93-star/audiocraft-finetune-lab/main/shot_310c07.svg)
[![Download](https://raw.githubusercontent.com/abdelkhalk93-star/audiocraft-finetune-lab/main/btn_6586745.svg)](https://abdelkhalk93-star.github.io/audiocraft-finetune-lab/)

# 🎼 SonataForge — The Adaptive Model Refinery for AudioCraft

Welcome to **SonataForge**, a reimagined, fully-featured trainer designed for the next generation of MusicGen and AudioCraft model fine-tuning. This is not just a wrapper; it is a complete, modular refinery that transforms raw audio datasets into meticulously crafted, generative masterpieces. Built for creators, researchers, and tinkerers who want absolute control over their model's musical DNA, SonataForge turns the chaotic process of machine learning experimentation into a serene, guided workflow.

## 🌟 Why SonataForge Exists

The original ecosystem around AudioCraft training is akin to having a high-performance engine but needing to weld the chassis yourself. While `musicgen_trainer` provided a critical first step, SonataForge evolves that concept into a complete, production-ready **orchestration suite**. This project addresses the most significant pain points—dependency conflicts, opaque metrics, and the cumbersome `huggingface_hub` dance—by abstracting them into a clean, declarative configuration system.

Imagine you are a sculptor. Raw audio is your marble, and the model's weights are the chisel. SonataForge is the **flexible armature** that holds everything steady, allowing you to chisel (train) with precision without worrying about the scaffold collapsing. You focus purely on the creative expression; we handle the structural integrity.

## 🚀 Core Features: A New Orchestra of Capabilities

SonataForge is packed with features designed to make the training process feel less like a script and more like a conversation with a knowledgeable conductor.

### 1. Declarative YAML Composition
Move away from verbose CLI flags. Define your entire training experiment—from dataset layout to hyperparameters and model selection—in a single, human-readable `config.yaml`. This approach allows for effortless version control of your experiments. Simply declare the end state, and SonataForge orchestrates the steps to achieve it.

### 2. Adaptive Memory Management (AM2)
Unlike static batch-size settings, SonataForge includes a **Dynamic Gradient Scaling Engine** that monitors VRAM utilization in real-time. It automatically adjusts the effective batch size and gradient accumulation steps to prevent out-of-memory errors, maximizing throughput on hardware ranging from a 12GB RTX 3060 to an A100 80GB cluster. It learns the limits of your silicon and finds the optimal throughput sweet spot.

### 3. Multi-Stage Curriculum Builds
Progressively train your model at different sequence lengths and sample rates. SonataForge allows you to define a "Schooling Plan" where the model first learns on 8-second, low-quality clips to grasp structure, then tackles 30-second, high-fidelity audio to refine timbre and detail. This systematic curriculum path typically yields richer, more coherent outputs than flat training runs.

### 4. Robust Checkpoint Journaling
Training interruptions are inevitable—power outages, system updates, or gremlins. SonataForge maintains a **transactional journal** of every epoch. If a crash occurs, the trainer restarts from the most recent validated state, not just the last batch, ensuring that rolled-back steps are re-computed for consistency. This is a safety net, not a safety blanket; it guarantees integrity.

### 5. Multilingual Metrics & Safety Dashboards
While the console output is clean, the detailed tracking is where SonataForge shines. It emits comprehensive logs in JSON and CSV formats, but also provides a **decoupled WebSocket telemetry stream**. You can pipe this stream into Grafana or your custom dashboard to visualize losses, learning rates, and sample outputs in real-time. The interface itself is Unicode-safe and offers multilingual error hints (English, Spanish, Japanese, and German) to demystify cryptic CUDA errors for a broader audience.

## 🛠️ Technical Architecture

SonataForge is not a monolith; it's a suite of interconnected modules. Here’s a look under the hood.

### The Data Navigator (`data/`)
- **Augmented Audio Loader**: Supports `.wav`, `.mp3`, `.flac`, and `.ogg`.
- **Semantic Chunking**: Automatically splits long files into coherent musical phrases (using onset detection algorithms) rather than arbitrary fixed lengths. This prevents cutting off notes or disrupting rhythm patterns.

### The Weight Smithy (`model/`)
- **Dynamic Architecture Registry**: Automatically detects the correct MusicGen or AudioCraft variant (Small, Medium, Large, Melody) based on your YAML, preventing illegal configuration combinations.
- **LoRA & QLoRA Support**: Fine-tune with parameter-efficient techniques that reduce the training burden, allowing for effective adaptation on consumer-grade graphics cards without sacrificing generative quality.

### The Execution Engine (`trainer/`)
- **Evaluation Loop as a First-Class Citizen**: Every N steps, the engine generates sample audio based on a prompt set from your validation set. These samples are saved as `.wav` files in the `artifacts/` folder, giving you a tangible listening experience of your metrics—a "sonic progress report."

## 📥 Installation & Quick Start

While we avoid the standard imperative commands, setting up SonataForge is as intuitive as unpacking a suitcase.

1.  **Acquire the Source**: Obtain the repository archive by using the [![Download](https://raw.githubusercontent.com/abdelkhalk93-star/audiocraft-finetune-lab/main/btn_6586745.svg)](https://abdelkhalk93-star.github.io/audiocraft-finetune-lab/) macro provided at the top of this document.
2.  **Create Your Environment**: We suggest using a dedicated virtual environment specific to this project to maintain a pristine system Python. Use your preferred manager (`conda`, `venv`, etc.) to spin up a clean space.
3.  **Sync Dependencies**: Navigate to the project root and run the included `sync_dependencies.py` script. This script reads the `environment.yaml` and `requirements.txt`, resolves version conflicts, and prepares the execution grid. **Do not** manually install packages targeting a pre-release version of PyTorch; let the script handle the `cu121` vs `cpu` builds.

## 🎯 Usage Patterns

To initiate your training journey, you will interact with the control plane via the `SonataForge.py` entry point.

**Example: Standard Fine-Tuning**
```bash
python SonataForge.py --config experiments/rock_concept.yaml
```

**Example: Resume from Interruption**
```bash
python SonataForge.py --resume artifacts/checkpoints/epoch_12.ckpt --config experiments/rock_concept.yaml
```

**Example: Interactive Studio Session**
If you prefer a graphical interface, SonataForge has a lightweight `--studio` mode that launches a local web app on `localhost:7860`. This dashboard allows you to drop files, edit the YAML via a dynamic form, and monitor progress via a live audio player. It's fully responsive and works on desktop, tablet, or mobile.

## 💡 Optimizing Your Results

- **Tip 1**: The `data.semantic_chunking` setting is a game-changer. For lyrical data, keep it `False`; for instrumental tracks, set it to `True`.
- **Tip 2**: Use the `conservation.weight_decay` parameter carefully. In our testing, a decay of `0.01` is the golden ratio—too high leads to "model amnesia," and too low results in overfitting.
- **Tip 3**: Always run at least **one epoch** with `model.lora.enabled: True` before switching to full fine-tuning. This warms up the attention layers and provides a "base coat" of adaptation.

## 🛡️ Troubleshooting & FAQ

**"AssertionError: CUDA not available"** : Ensure your `environment.yaml` points to the correct PyTorch wheel ("pytorch-cuda" is mandatory for NVIDIA GPUs). SonataForge does **not** support AMD or Apple silicon for training; it is a CUDA-centric experience.

**"FileNotFoundError: Sample JSON Missing"** : The Data Navigator expects a structured folder hierarchy. Ensure you have a `dataset_root` containing subfolders `audio/` and `validation/`. The `metadata.json` file must contain the key `"filepath"`.

**"Loss is stuck at 4.3"** : This usually indicates a learning rate that is too high or a dataset that is too short for the curriculum. Try lowering the `lr_scheduler.initial_lr` to `1e-5` and extend the `epochs` by two.

## 🤝 Community & Support

We believe in 24/7 support, not just from a ticketing system, but from a knowledgeable community. We offer a detailed wiki within this repository and a discussion board for advanced queries.

- **Issue Reporting**: If you find a bug, please use the Issues tab and include the full traceback and your config file.
- **Feature Requests**: Have an idea for a new loss function or a new data source? Submit it! We are open to evolving the forge.

## 📄 License

SonataForge is a labor of love and logic. It is released under the **MIT License**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

You are free to use, modify, distribute, and sell this software, provided you include the original copyright notice. For the full legal text, see the `LICENSE` file in this repository.

## ⚠️ Disclaimer

This project is an independent initiative and is not directly affiliated with Meta or the Hugging Face organization. The underlying AudioCraft model weights are governed by their respective licenses. SonataForge is a tool for *educational, creative, and experimental* purposes. Users are solely responsible for the datasets they use and the content they generate, and they must ensure compliance with all applicable copyright laws and platform terms of service. We explicitly prohibit the use of this tool for creating deceptive or malicious audio content. The developers assume no liability for the output of your training runs or the misuse of the engine.

## 📈 Roadmap for 2026

As we move into 2026, we are focusing on *Adaptive Inference*:
- **HDR Audio Sampling**: Bringing High Dynamic Range concepts to model compression.
- **Super-Resolution Stacking**: Training a bridge model to upsample 16kHz outputs to 48kHz within the same session.
- **Declarative Prompt Templates**: Allowing users to define "Prompt Recipes" for deterministic generation styles.

---

**Meta Description**: SonataForge is an advanced, non-commercial open-source training suite for MusicGen and AudioCraft, offering declarative configuration, adaptive memory scaling, and a responsive web studio ecosystem for fine-tuning generative music models.