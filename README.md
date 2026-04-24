# 🎬 VidMind · YouTube AI Analyst

> Turn any YouTube video into a conversation. Get AI-powered summaries, chat with video content, and download transcripts — all in one place.

---

## ✨ Features

- 📋 **Smart Summary** — Fact-based, concise summaries with no fluff
- 💬 **Chat with Video** — Ask questions and get context-aware answers from the video
- 🌐 **Multi-Language** — Auto-detects and translates transcripts from 15+ languages
- 📄 **Transcript Viewer** — Search, browse, and download the full transcript
- 📦 **Export Everything** — Download summary, chat history, or a full report

---

## 🚀 Demo

👉 **[Live App on Hugging Face Spaces](https://huggingface.co/spaces/RupStrange/vidmind)**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Llama 4 Scout via Groq |
| Embeddings | BAAI/bge-small-en-v1.5 (HuggingFace) |
| Vector Store | FAISS |
| RAG Pipeline | LangChain |
| Memory | ConversationSummaryBufferMemory |
| Transcripts | youtube-transcript-api |

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/RupStrange/vidmind-youtube-ai-chatbot.git
cd vidmind-youtube-ai-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up your API key
Create a `.streamlit/secrets.toml` file:
```toml
GROQ_API_KEY = "your_groq_api_key_here"
```
Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔑 Environment Variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Your Groq API key from [console.groq.com](https://console.groq.com) |

---

## 🎯 How It Works

1. 🔗 Paste a YouTube URL
2. 📥 App fetches the transcript (auto-detects language)
3. 🌐 Translates to English if needed
4. 🧠 Extracts facts and generates a summary
5. 🔍 Builds a RAG knowledge base from the transcript
6. 💬 You can now chat with the video content!

---

## 🌐 Supported Languages

English, Hindi, Bengali, Arabic, Chinese, French, German, Spanish, and many more via auto-detection.

---

## ⚠️ Known Limitations

- Some YouTube videos have transcripts disabled — these cannot be processed
- YouTube may block transcript fetching from certain cloud server IPs
- Very long videos may take longer to process

---

## 📄 License

MIT License — feel free to use, modify, and distribute.

---

## 🙌 Author

Made with ❤️ by [RupStrange](https://github.com/RupStrange)

⭐ Star this repo if you found it useful!
