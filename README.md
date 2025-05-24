# 📚 VNU-RAG-QA

A Retrieval-Augmented Generation (RAG) based Question Answering system built with modern LLM frameworks and platforms.

---

## 🧱 Architecture Overview

### 1. System Construction Flow
![System Flow](https://github.com/user-attachments/assets/fa0ef5aa-2257-499a-90eb-5367bdcf912a)

### 2. Technologies Used in the System Structure
![Tech Stack](https://github.com/user-attachments/assets/abc85898-b989-434b-93ab-74fe31a8c0e1)

---

## 🚀 Getting Started

### 📥 Dataset
You can download the raw dataset from Google Drive:  
[📂 Click here to download the data](https://drive.google.com/file/d/1xYyzZJI5DLwaCO-MFLtYh07x4aPE7pV-/view?usp=drive_link)

---

### ⚙️ Prerequisites

- Install [Ollama](https://ollama.com/)
- Run **LLaMA 3.2** locally:

```bash
ollama run llama3.2
```

---

### 🛠️ Installation Steps

```bash
# Step 1: Move into the project folder
cd rag

# Step 2: Install Python dependencies
pip install -r requirements.txt
```

---

### 🧠 Build the Vector Database

If you haven't built the vector database yet, run:

```bash
python vector_db.py
```

---

### 💬 Run Chat Inference

To start generating answers based on your text files, run:

```bash
python chat.py
```

---

### ✅ Sample Result

![Chat Output](https://github.com/user-attachments/assets/2d3f787c-088f-4a78-ac1d-b79782c0fc59)

---

## 🧰 Frameworks & Platforms Used

### 🔧 Frameworks
- 🔥 [PyTorch](https://pytorch.org/) – deep learning framework
- 🤖 [Transformers](https://huggingface.co/docs/transformers) – pretrained language models
- 🔗 [LangChain](https://www.langchain.com/) – build LLM-powered applications

### ☁️ Platforms
- 🤗 [Hugging Face](https://huggingface.co/) – model hub and ecosystem
- 🧠 [Ollama](https://ollama.com/) – run LLMs locally with ease

<p align="center">
  <img src="https://cdn.worldvectorlogo.com/logos/pytorch-2.svg" alt="PyTorch" width="120" />
  <img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" alt="HuggingFace" width="100" />
  <img src="https://brandlogos.net/wp-content/uploads/2025/03/langchain-logo_brandlogos.net_9zgaw-768x768.png" alt="LangChain" width="100" />
  <img src="https://awakast.com/wp-content/uploads/2024/03/ollama-logo.png" alt="Ollama" width="100" />
</p>

