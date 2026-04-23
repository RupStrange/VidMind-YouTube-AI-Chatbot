import os
import re
import streamlit as st
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.memory import ConversationSummaryBufferMemory
from urllib.parse import urlparse, parse_qs
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

st.set_page_config(
    page_title="VidMind · YouTube AI Analyst",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CACHED RESOURCES ───────────────────────────────────────────────────────────

@st.cache_resource
def get_model():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=st.secrets["GROQ_API_KEY"]
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

model = get_model()
parser = StrOutputParser()

# ── HELPERS ────────────────────────────────────────────────────────────────────

def extract_video_id(url: str):
    if not url:
        return None
    parsed = urlparse(url)
    if "youtube.com" in str(parsed.hostname or ""):
        return parse_qs(parsed.query).get("v", [None])[0]
    if parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")[:11] or None
    return None

def get_video_meta(video_id):
    try:
        import urllib.request, json
        url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
            return data.get("title", "YouTube Video"), data.get("author_name", "Unknown")
    except Exception:
        return "YouTube Video", "Unknown"

def transcripts_fetch(video_id):
    preferred = ["en", "hi", "bn", "ar", "zh", "fr", "de", "es"]
    if not video_id:
        return [], "unknown"
    try:
        for lang in preferred:
            try:
                tl = YouTubeTranscriptApi().list(video_id).find_transcript([lang]).fetch()
                return tl, lang
            except NoTranscriptFound:
                continue
        for transcript in YouTubeTranscriptApi().list(video_id):
            try:
                return transcript.fetch(), transcript.language_code
            except Exception:
                continue
        return [], "unknown"
    except TranscriptsDisabled:
        return [], "disabled"
    except Exception:
        return [], "error"

def translation(transcript_list, language_code):
    full_text = " ".join(snippet.text for snippet in transcript_list)
    if language_code == "en":
        return full_text
    chunks = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(full_text)
    prompt_translate = PromptTemplate(
        template="You are an expert translator. Translate from {language_code} to English.\nOutput ONLY the translated text.\ntext: {snippet}",
        input_variables=["language_code", "snippet"]
    )
    chain = prompt_translate | model | parser
    translated = []
    prog = st.progress(0, text="Translating transcript…")
    for i, chunk in enumerate(chunks):
        translated.append(chain.invoke({"language_code": language_code, "snippet": chunk}))
        prog.progress((i + 1) / len(chunks), text=f"Translating… {i+1}/{len(chunks)}")
    prog.empty()
    return " ".join(translated)

def preprocess(transcript_list, language_code):
    text = translation(transcript_list, language_code)
    return re.sub(r'\s+', ' ', text.replace("\n", " ")).strip()

def facts_extract(translated: str):
    docs = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=150).create_documents([translated])
    fact_prompt = PromptTemplate(
        input_variables=["text"],
        template="Extract only explicit facts as short bullet points. No interpretation.\n\n{text}\n\nFacts:"
    )
    facts, prog = [], st.progress(0, text="Extracting facts…")
    for i, doc in enumerate(docs):
        facts.append((fact_prompt | model | parser).invoke(doc.page_content))
        prog.progress((i + 1) / len(docs), text=f"Extracting facts… {i+1}/{len(docs)}")
    prog.empty()
    return "\n".join(facts)

def generate_summary(translated: str):
    facts = facts_extract(translated)
    final_prompt = PromptTemplate(
        input_variables=["facts"],
        template="Write a concise summary (max 500 words) using ONLY these facts. Chronological order, no new info.\n\n{facts}\n\nFinal Summary:"
    )
    return (RunnableLambda(lambda x: {"facts": x}) | final_prompt | model | parser).invoke(facts)

def build_rag(translated: str):
    docs = NLTKTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([translated])
    vs = FAISS.from_documents(documents=docs, embedding=get_embeddings())
    return vs.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5})

# ── SESSION STATE DEFAULTS ─────────────────────────────────────────────────────

defaults = {
    "processed": False,
    "translated": None,
    "summary": None,
    "retriever": None,
    "messages": [],
    "memory": None,
    "video_id": None,
    "video_title": "",
    "video_author": "",
    "lang_code": "en",
    "word_count": 0,
    "char_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🎬 VidMind")
    st.caption("🤖 YouTube AI Analyst · ⚡ Powered by Groq + LangChain")
    st.divider()

    if not st.session_state.processed:
        url_input = st.text_input(
            "🔗 YouTube URL",
            placeholder="https://youtube.com/watch?v=...",
            help="Paste any YouTube video link"
        )
        analyze_clicked = st.button("⚡ Analyze Video", type="primary", use_container_width=True)
    else:
        url_input = ""
        analyze_clicked = False
        vid = st.session_state.video_id
        if vid:
            st.image(f"https://img.youtube.com/vi/{vid}/mqdefault.jpg", use_container_width=True)
        st.markdown(f"**🎞️ {st.session_state.video_title}**")
        st.caption(f"📺 {st.session_state.video_author}")
        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("📝 Words", f"{st.session_state.word_count:,}")
        c2.metric("🌐 Language", st.session_state.lang_code.upper())
        st.metric("🔤 Characters", f"{st.session_state.char_count:,}")
        st.divider()
        if st.button("🔄 Analyze New Video", use_container_width=True):
            for k in defaults:
                st.session_state[k] = defaults[k]
            st.rerun()

    st.divider()
    with st.expander("ℹ️ How to use"):
        st.markdown("""
1. 🔗 Paste a YouTube URL above
2. ⚡ Click **Analyze Video**
3. 📋 Read the AI summary
4. 💬 Chat with the video content
5. ⬇️ Download transcript or summary
        """)

# ── PROCESS VIDEO ──────────────────────────────────────────────────────────────

if analyze_clicked and url_input:
    video_id = extract_video_id(url_input)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please check and try again.")
    else:
        st.session_state.video_id = video_id
        title, author = get_video_meta(video_id)
        st.session_state.video_title = title
        st.session_state.video_author = author

        with st.status("⚙️ Processing video…", expanded=True) as status:
            st.write("📥 Fetching transcript…")
            transcript_list, lang_code = transcripts_fe
