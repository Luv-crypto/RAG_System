# Genomic RAG: Multimodal Metadata‑Driven PDF Retrieval

**Genomic RAG** is a **multimodal** Retrieval‑Augmented Generation (RAG) system that provides rich, metadata‑driven access to genomic classification information embedded within PDF documents. It supports **text**, **table**, and **image** queries, enabling **lightning‑fast retrieval** of relevant scientific insights. Future releases will introduce a specialized financial‑documents version with tailored pipelines. Future releases will introduce a specialized financial‑documents version with tailored pipelines.

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Luv-crypto/RAG_System.git
cd RAG_System

# 2. Create & activate venv
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 3. Install deps
pip install --upgrade pip
pip install -r requirements.txt

# 4. Copy .env and set secrets
cp .env.example .env
# Edit .env: SECRET_KEY, OPENAI_API_KEY, etc.

# 5. Run the app
python main.py
```

# 4. Copy .env and set secrets

cp .env.example .env

# Edit .env: SECRET\_KEY, OPENAI\_API\_KEY, etc.

# 5. Run the app

python main.py

````

> In `main.py`, the app is launched with:
> ```python
> if __name__ == "__main__":
>     app.run(host="127.0.0.1", port=5000, debug=True)
> ```

---

## 📄 Demo Video

Check out the short demo:

<video width="640" controls>
  <source src="assets/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Or view the animated GIF:

![App Demo](assets/demo.gif)

---

## 📦 Prerequisites

- Python 3.10+
- Git

---

## 🤝 Contributing & Roadmap

- 🚧 **Current**: Genomic classification & metadata retrieval pipelines.
- 🏦 **Coming Soon**: Financial documents specialization module.

Feel free to open issues or submit pull requests for new features!

---

## 📄 License

MIT © Luv‑crypto

````
