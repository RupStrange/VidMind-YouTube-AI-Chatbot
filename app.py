import os
import re
import streamlit as st
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationSummaryBufferMemory
from urllib.parse import urlparse, parse_qs
import warnings
warnings.filterwarnings("ignore")

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
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_text(full_text)
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
    docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).create_documents([translated])
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
            transcript_list, lang_code = transcripts_fetch(video_id)
            st.session_state.lang_code = lang_code

            if not transcript_list:
                msgs = {
                    "disabled": "🚫 Transcripts are disabled for this video.",
                    "error": "💥 An error occurred while fetching transcripts.",
                    "unknown": "🤷 No transcripts could be found.",
                }
                status.update(label="❌ Failed", state="error")
                st.error(msgs.get(lang_code, "❌ No transcripts found."))
            else:
                st.write(f"✅ Transcript fetched ({lang_code.upper()}) 🌐")
                st.write("🔄 Preprocessing & translating…")
                translated = preprocess(transcript_list, lang_code)
                st.session_state.translated = translated
                st.session_state.word_count = len(translated.split())
                st.session_state.char_count = len(translated)

                st.write("📝 Generating summary… ✨")
                st.session_state.summary = generate_summary(translated)

                st.write("🔍 Building knowledge base… 🧠")
                st.session_state.retriever = build_rag(translated)
                st.session_state.memory = ConversationSummaryBufferMemory(
                    llm=model, max_token_limit=1000, return_messages=True
                )
                st.session_state.messages = []
                st.session_state.processed = True
                status.update(label="✅ Ready to explore! 🚀", state="complete", expanded=False)
        st.rerun()

# ── MAIN CONTENT ───────────────────────────────────────────────────────────────

if not st.session_state.processed:
    st.title("🎬 Welcome to VidMind")
    st.markdown("**✨ Turn any YouTube video into a conversation.** 🔗 Paste a URL in the sidebar to get started.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### 🎯 Smart Summary\n✅ Get a concise, fact-based summary of any YouTube video — no fluff, just substance.")
    with c2:
        st.success("### 💬 Chat with Video\n🤖 Ask questions and get context-aware answers drawn directly from the video content.")
    with c3:
        st.warning("### 🌐 Multi-Language\n🗺️ Automatic transcript detection and translation across 15+ languages.")

    st.divider()
    st.markdown("#### 🚀 Works great with")
    eg1, eg2, eg3, eg4 = st.columns(4)
    eg1.markdown("🎓 Lectures & Tutorials")
    eg2.markdown("📰 News & Documentaries")
    eg3.markdown("💡 Tech Talks & Podcasts")
    eg4.markdown("🏋️ Fitness & How-To Guides")

else:
    hcol1, hcol2 = st.columns([3, 1])
    with hcol1:
        st.title(f"🎬 {st.session_state.video_title}")
        st.caption(f"by {st.session_state.video_author}  ·  {st.session_state.word_count:,} words  ·  {st.session_state.lang_code.upper()}")
    with hcol2:
        vid = st.session_state.video_id
        st.link_button("▶️ Watch on YouTube", f"https://youtube.com/watch?v={vid}", use_container_width=True)

    st.divider()

    tab_summary, tab_chat, tab_transcript = st.tabs(["📋 Summary", "💬 Chat", "📄 Transcript"])

    # ── SUMMARY TAB ────────────────────────────────────────────────────────────

    with tab_summary:
        st.subheader("📋 AI-Generated Summary ✨")
        st.info(st.session_state.summary)

        sc1, sc2, sc3 = st.columns([1, 1, 2])
        with sc1:
            st.download_button(
                "⬇️ Download Summary",
                data=st.session_state.summary,
                file_name="summary.txt",
                mime="text/plain",
                use_container_width=True
            )
        with sc2:
            if st.button("🔁 Regenerate ✨", use_container_width=True):
                with st.spinner("🔄 Regenerating summary…"):
                    st.session_state.summary = generate_summary(st.session_state.translated)
                st.rerun()

        st.divider()
        st.subheader("📊 Video Stats")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("📝 Words", f"{st.session_state.word_count:,}")
        m2.metric("🔤 Characters", f"{st.session_state.char_count:,}")
        m3.metric("🌐 Language", st.session_state.lang_code.upper())
        read_time = max(1, st.session_state.word_count // 200)
        m4.metric("⏱️ Est. Read Time", f"{read_time} min")

    # ── CHAT TAB ───────────────────────────────────────────────────────────────

    with tab_chat:
        if st.session_state.memory is None:
            st.warning("⚠️ Something went wrong — memory is not initialized. Please re-analyze the video.")
            st.stop()

        system_prompt = """You are VidMind, an expert analyst assistant for YouTube video transcripts.

Your behavior:
- Answer ONLY based on the provided video context
- If something wasn't in the video, say: "This wasn't covered in the video."
- If asked within the topic but not in context, you may elaborate slightly
- Never give one-word answers; always be thorough and helpful
- Remember prior conversation turns and refer to them when relevant
- Greet warmly if greeted; introduce yourself as VidMind

Context: {context}
"""

        def format_docs(docs):
            return "\n\n".join(d.page_content for d in docs)

        _memory = st.session_state.memory
        _retriever = st.session_state.retriever

        def history_load(_):
            if _memory:
                return _memory.load_memory_variables({})["history"]
            return []

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])

        rag_chain = (
            {
                "context": _retriever | format_docs,
                "question": RunnablePassthrough(),
                "history": history_load,
            }
            | prompt | model | parser
        )

        user_input = st.chat_input("💬 Ask anything about the video…")

        chat_container = st.container(height=500)
        with chat_container:
            if not st.session_state.messages:
                st.info("👋 Welcome! 🎬 Ask me anything about the video — I'm ready to help. 🤖✨")
            else:
                for msg in st.session_state.messages:
                    avatar = "🧑" if msg["role"] == "user" else "🤖"
                    with st.chat_message(msg["role"], avatar=avatar):
                        st.markdown(msg["content"])

        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(user_input)
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("🧠 Thinking…"):
                        response = rag_chain.invoke(user_input)
                        if _memory:
                            _memory.save_context(
                                {"input": user_input}, {"output": response}
                            )
                    st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

        if st.session_state.messages:
            st.divider()
            cc1, cc2 = st.columns([1, 5])
            with cc1:
                if st.button("🗑️ Clear Chat"):
                    st.session_state.messages = []
                    if st.session_state.memory:
                        st.session_state.memory.clear()
                    st.rerun()
            with cc2:
                chat_export = "\n\n".join(
                    f"{'🧑 You' if m['role']=='user' else '🤖 VidMind'}: {m['content']}"
                    for m in st.session_state.messages
                )
                st.download_button(
                    "⬇️ Export Chat 💾",
                    data=chat_export,
                    file_name="chat_export.txt",
                    mime="text/plain"
                )

    # ── TRANSCRIPT TAB ─────────────────────────────────────────────────────────

    with tab_transcript:
        st.subheader("📄 Processed Transcript 🌐")
        st.caption("🧹 Cleaned and translated transcript used for analysis.")

        search_term = st.text_input("🔍 Search transcript", placeholder="🔎 Type a keyword…")
        transcript = st.session_state.translated

        if search_term:
            count = transcript.lower().count(search_term.lower())
            st.caption(f"🎯 Found **{count}** occurrence(s) of '{search_term}'")
            parts = transcript.split(". ")
            hits = [p for p in parts if search_term.lower() in p.lower()]
            if hits:
                st.success("✅ **Matching sentences:**")
                for h in hits[:20]:
                    st.markdown(f"• {h.strip()}.")
            else:
                st.warning("🤷 No matching sentences found.")
            st.divider()

        with st.expander("📖 View full transcript", expanded=False):
            st.text_area(
                label="Full transcript",
                value=transcript,
                height=400,
                label_visibility="collapsed"
            )

        st.download_button(
            "⬇️ Download Full Transcript 📄",
            data=transcript,
            file_name="transcript.txt",
            mime="text/plain",
        )

        st.divider()
        st.subheader("📦 Export Everything 🚀")
        full_export = f"""VidMind Export
==============
Video: {st.session_state.video_title}
Channel: {st.session_state.video_author}
Language: {st.session_state.lang_code.upper()}
Word Count: {st.session_state.word_count:,}

SUMMARY
-------
{st.session_state.summary}

FULL TRANSCRIPT
---------------
{st.session_state.translated}
"""
        st.download_button(
            "📦 Download Full Report (Summary + Transcript)",
            data=full_export,
            file_name="vidmind_report.txt",
            mime="text/plain",
            use_container_width=True,
            type="primary"
        )
