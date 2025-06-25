# 🧬 Genomic RAG

**Multimodal Metadata‑Driven PDF Retrieval**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)  [![Python](https://img.shields.io/badge/python-3.10%2B-green.svg)](https://www.python.org/)  [![Demo Video](https://img.shields.io/badge/demo_video-▶️-blue)](assets/demo.mp4)

**Genomic RAG** is a **multimodal** Retrieval‑Augmented Generation (RAG) system for **genomic classification** information in PDFs. It supports **text**, **table**, and **image** queries, delivering **lightning‑fast retrieval** via rich metadata indexing. Future releases will include a specialized financial‑documents module.

---

## 📋 Table of Contents

1. [Quick Start](#-quick-start)
2. [Running Locally](#-running-locally)
3. [Demo Video & GIF](#-demo-video--gif)
4. [Prerequisites](#-prerequisites)
5. [Contributing & Roadmap](#-contributing--roadmap)
6. [License](#-license)

---

## 🚀 Quick Start

1. **Clone the repo**

   ```bash
   git clone https://github.com/Luv-crypto/RAG_System.git
   cd RAG_System
   ```
2. **Setup virtual environment**

   ```bash
   python -m venv venv
   # macOS/Linux:
   source venv/bin/activate
   # Windows PowerShell:
   .\\venv\\Scripts\\Activate.ps1
   ```
3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
4. **Configure environment variables**

   ```bash
   cp .env.example .env
   # Edit .env and set:
   # SECRET_KEY, OPENAI_API_KEY, etc.
   ```
5. **Start the application**

   ```bash
   python main.py
   ```

---

## 🏃 Running Locally

By default, `main.py` launches the Flask server on `127.0.0.1:5000`:

```python
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
```

Open your browser at [http://localhost:5000](http://localhost:5000).

---

## 📄 Demo Video & GIF

### Demo Video

<video width="640" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Animated GIF

![App Demo](assets/demo.gif)

---

## 📦 Prerequisites

* **Python** ≥ 3.10
* **Git**

---

## 🤝 Contributing & Roadmap

* 🚧 **Current Focus**: Genomic metadata retrieval pipelines
* 🏦 **Next Up**: Financial‑documents specialized module

Contributions welcome! Please open issues or submit pull requests.

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.
