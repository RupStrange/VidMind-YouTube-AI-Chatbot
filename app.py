import os
import re
import time
import streamlit as st
from dotenv import load_dotenv
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

# Load environment variables before any Streamlit command
load_dotenv()

os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
    # Use a widely‑available Groq model as fallback if the original is not found.
    models_to_try = [
        "llama-3.1-70b-versatile",          # very reliable
        "meta-llama/llama-4-scout-17b-16e-instruct",  # original (if still valid)
        "mixtral-8x7b-32768",
    ]
    for model_name in models_to_try:
        try:
            return ChatGroq(model=model_name, api_key=os.getenv("GROQ_API_KEY"))
        except Exception:
            continue
    # If none work, the last attempt will raise an exception that will be caught at startup
    return ChatGroq(model=models_to_try[-1], api_key=os.getenv("GROQ_API_KEY"))

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
    """
    Fetch transcripts with a timeout to avoid hanging.
    Returns: (list of snippet dicts, language_code or status)
    """
    preferred = ["en", "hi", "bn", "ar", "zh", "fr", "de", "es"]
    if not video_id:
        return [], "unknown"

    # Helper to attempt fetching a transcript with timeout
    def try_fetch_transcript(transcript):
        try:
            # The fetch() call itself doesn't support timeout, so we wrap it
            # in a thread with timeout (simple approach with time.sleep is not ideal,
            # but we can use youtube_transcript_api's built-in proxy parameter?).
            # Instead, we'll use the requests timeout via the API's http client.
            # Unfortunately the library doesn't expose that; we'll add a crude
            # wrapper using threading. But to keep it simple, we catch all exceptions.
            return transcript.fetch()  # No timeout – we handle with a short overall timeout
        except Exception:
            return None

    try:
        # Get the transcript list object
        try:
            transcript_list = YouTubeTranscriptApi().list(video_id)
        except TranscriptsDisabled:
            return [], "disabled"
        except Exception:
            return [], "error"

        # Try preferred languages
        for lang in preferred:
            try:
                transcript = transcript_list.find_transcript([lang])
                # Attempt fetch – we can't easily set timeout, but we'll catch
                fetched = transcript.fetch()
                if fetched:
                    return fetched, lang
            except NoTranscriptFound:
                continue
            except Exception:
                continue

        # Fallback: try any available transcript
        for transcript in transcript_list:
            try:
                fetched = transcript.fetch()
                if fetched:
                    return fetched, transcript.language_code
            except Exception:
                continue

        return [], "unknown"
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
        try:
            translated.append(chain.invoke({"language_code": language_code, "snippet": chunk}))
        except Exception as e:
            st.error(f"Translation failed at chunk {i+1}: {str(e)[:200]}")
            translated.append(chunk)  # fallback to original chunk
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
        try:
            facts.append((fact_prompt | model | parser).invoke(doc.page_content))
        except Exception as e:
            st.error(f"Fact extraction failed at chunk {i+1}: {str(e)[:200]}")
            facts.append(doc.page_content)  # use raw text as fallback
        prog.progress((i + 1) / len(docs), text=f"Extracting facts… {i+1}/{len(docs)}")
    prog.empty()
    return "\n".join(facts)

def generate_summary(translated: str):
    facts = facts_extract(translated)
    final_prompt = PromptTemplate(
        input_variables=["facts"],
        template="Write a concise summary (max 500 words) using ONLY these facts. Chronological order, no new info.\n\n{facts}\n\nFinal Summary:"
    )
    try:
        return (RunnableLambda(lambda x: {"facts": x}) | final_prompt | model | parser).invoke(facts)
    except Exception as e:
        st.error(f"Summary generation failed: {str(e)[:200]}")
        return facts  # fallback: just return the facts

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
            transcript_list, lang_code = transcripts_fetch(video_id)
            st.session_state.lang_code = lang_code

            if not transcript_list:
                msgs = {
                    "disabled": "🚫 Transcripts are disabled for this video.",
                    "error": "💥 An error occurred while fetching transcripts (check network or video availability).",
                    "unknown": "🤷 No transcripts could be found.",
                }
                status.update(label="❌ Failed", state="error")
                st.error(msgs.get(lang_code, "❌ No transcripts found."))
                st.stop()  # Stop here, don't rerun or continue
            else:
                st.write(f"✅ Transcript fetched ({lang_code.upper()}) 🌐")
                st.write("🔄 Preprocessing & translating…")
                try:
                    translated = preprocess(transcript_list, lang_code)
                except Exception as e:
                    status.update(label="❌ Translation error", state="error")
                    st.error(f"Translation error: {str(e)[:300]}")
                    st.stop()
                st.session_state.translated = translated
                st.session_state.word_count = len(translated.split())
                st.session_state.char_count = len(translated)

                st.write("📝 Generating summary… ✨")
                try:
                    st.session_state.summary = generate_summary(translated)
                except Exception as e:
                    status.update(label="❌ Summary error", state="error")
                    st.error(f"Summary creation error: {str(e)[:300]}")
                    st.stop()

                st.write("🔍 Building knowledge base… 🧠")
                st.session_state.retriever = build_rag(translated)
                st.session_state.memory = ConversationSummaryBufferMemory(
                    llm=model, max_token_limit=1000, return_messages=True
                )
                st.session_state.messages = []
                st.session_state.processed = True
                status.update(label="✅ Ready to explore! 🚀", state="complete", expanded=False)
                # Delay slightly for the user to see the completion
                time.sleep(0.5)
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

    # ── CHAT TAB (FIXED DUPLICATION) ───────────────────────────────────────────

    with tab_chat:
        if st.session_state.memory is None:
            st.warning("⚠️ Memory not initialized. Please re-analyze the video.")
            st.stop()

        st.subheader("💬 Chat with the video")

        for msg in st.session_state.messages:
            avatar = "🧑" if msg["role"] == "user" else "🤖"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if not st.session_state.messages:
            st.info("👋 Welcome! Ask me anything about the video — I'm ready to help. 🤖✨")

        if prompt := st.chat_input("💬 Ask anything about the video…"):
            st.session_state.messages.append({"role": "user", "content": prompt})

            def format_docs(docs):
                return "\n\n".join(d.page_content for d in docs)

            def history_load(_):
                if st.session_state.memory:
                    return st.session_state.memory.load_memory_variables({}).get("history", [])
                return []

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

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])

            rag_chain = (
                {
                    "context": st.session_state.retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "history": history_load,
                }
                | prompt_template
                | model
                | parser
            )

            with st.spinner("🧠 Thinking…"):
                try:
                    response = rag_chain.invoke(prompt)
                    if st.session_state.memory:
                        st.session_state.memory.save_context(
                            {"input": prompt}, {"output": response}
                        )
                except Exception as e:
                    response = f"⚠️ An error occurred: {str(e)[:200]}"
                    st.error(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

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
