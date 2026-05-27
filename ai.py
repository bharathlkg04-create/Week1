import os
import re
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from urllib.parse import urlparse, parse_qs

from flask import Flask, render_template, request, Response, stream_with_context

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

app = Flask(__name__)


# ════════════════════════════════════════════════════════
#  SHARED HELPERS
# ════════════════════════════════════════════════════════

def get_llm(openai_api_key: str, model: str, temperature: float = 0.3):
    return ChatOpenAI(
        model=model,
        openai_api_key="",
        temperature=temperature,
        max_tokens=4096,
    )


def lcel_summarise(text: str, prompt_str: str, openai_api_key: str, model: str) -> str:
    """Generic LCEL summarise — prompt_str must contain {text}."""
    llm = get_llm("#enter_openai_api_key_here#", model)
    prompt = PromptTemplate(input_variables=["text"], template=prompt_str)
    chain  = prompt | llm | StrOutputParser()
    return chain.invoke({"text": text})


def extract_sources(text: str) -> list:
    sources, seen = [], set()
    for url in re.findall(r'https?://[^\s\]\)"\']+', text):
        url = url.rstrip(".,;)")
        if url not in seen and len(url) > 15:
            seen.add(url)
            domain = url.split("/")[2].replace("www.", "")
            sources.append({"url": url, "domain": domain})
    return sources[:8]


def sse(name: str, payload: dict) -> str:
    return f"event: {name}\ndata: {json.dumps(payload)}\n\n"


# ════════════════════════════════════════════════════════
#  FEATURE 1 — AI RESEARCH SUMMARISER (existing)
# ════════════════════════════════════════════════════════

def build_research_agent(openai_api_key, tavily_api_key, n_results, model):
    os.environ["TAVILY_API_KEY"] = "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"
    llm  = get_llm(openai_api_key, model, temperature=0.2)
    tool = TavilySearchResults(
        max_results=n_results, search_depth="advanced",
        include_answer=True, include_raw_content=False,
    )
    system = (
        "You are an expert research assistant. Search for the most recent and relevant "
        "information using the search tool. Run multiple searches if needed. "
        "Return ALL raw search results — do NOT summarise yourself."
    )
    return create_react_agent(llm, [tool], prompt=system)


def run_agent(agent, query: str) -> str:
    result   = agent.invoke({"messages": [("human", query)]})
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and isinstance(msg.content, str) and msg.content.strip():
            return msg.content
    return ""


STYLE_PROMPTS = {
    "structured": (
        "You are an expert analyst. Write a structured research report with sections:\n"
        "## Overview\n## Key Findings\n## Important Details\n## Conclusion\n\n"
        "Research data:\n{text}\n\nReport:"
    ),
    "bullets": (
        "You are an expert analyst. Summarise the research as concise bullet points grouped by theme.\n\n"
        "Research data:\n{text}\n\nSummary:"
    ),
    "brief": (
        "You are an expert analyst. Write a 3-paragraph executive brief (max 200 words): "
        "situation, key insights, implications.\n\nResearch data:\n{text}\n\nBrief:"
    ),
}


# ════════════════════════════════════════════════════════
#  FEATURE 2 — NEWS SCRAPER & SUMMARISER
# ════════════════════════════════════════════════════════

def scrape_news(topic: str, openai_api_key: str, tavily_api_key: str,
                n_results: int, model: str) -> dict:
    os.environ["TAVILY_API_KEY"] = "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"
    tool = TavilySearchResults(
        max_results=n_results, search_depth="advanced",
        include_answer=True, include_raw_content=False,
    )
    query   = f"latest news {topic} 2025"
    results = tool.invoke(query)

    articles = []
    for r in (results if isinstance(results, list) else []):
        articles.append({
            "title":   r.get("title", "Untitled"),
            "url":     r.get("url", ""),
            "content": r.get("content", ""),
            "domain":  urlparse(r.get("url", "")).netloc.replace("www.", ""),
        })

    combined = "\n\n".join(
        f"Title: {a['title']}\nSource: {a['domain']}\nContent: {a['content']}"
        for a in articles
    )

    prompt = (
        f"You are a news analyst. Summarise these latest news articles about '{topic}'.\n"
        "Format:\n## Headlines\n(bullet list of article titles)\n\n"
        "## What Happened\n(clear summary of events)\n\n"
        "## Why It Matters\n(impact and implications)\n\n"
        f"Articles:\n{{text}}\n\nSummary:"
    )
    summary = lcel_summarise(combined, prompt, openai_api_key, model)
    return {"articles": articles, "summary": summary}


# ════════════════════════════════════════════════════════
#  FEATURE 3 — YOUTUBE VIDEO SUMMARISER
# ════════════════════════════════════════════════════════

def extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from any YouTube URL format."""
    patterns = [
        r'(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})',
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def get_youtube_transcript(video_id: str) -> str:
    """Fetch transcript using youtube-transcript-api."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join(t["text"] for t in transcript)
    except Exception as e:
        return f"ERROR: {e}"


def summarise_youtube(url: str, openai_api_key: str, model: str, style: str) -> dict:
    video_id = extract_video_id(url)
    if not video_id:
        return {"error": "Invalid YouTube URL. Please paste a valid YouTube link."}

    transcript = get_youtube_transcript(video_id)
    if transcript.startswith("ERROR:"):
        return {"error": f"Could not fetch transcript: {transcript[7:]}. "
                         "Make sure the video has subtitles/captions enabled."}

    style_map = {
        "structured": (
            "Write a structured video summary with:\n"
            "## Video Overview\n## Main Topics Covered\n## Key Takeaways\n## Conclusion\n"
        ),
        "bullets": "Write a concise bullet-point summary of the video grouped by topic.\n",
        "brief":   "Write a 3-paragraph brief summary of the video (max 200 words).\n",
    }

    prompt = (
        f"You are an expert at summarising video content.\n"
        f"{style_map.get(style, style_map['structured'])}\n"
        f"Video transcript:\n{{text}}\n\nSummary:"
    )

    # Truncate transcript to ~12000 chars to stay within token limits
    summary = lcel_summarise(transcript[:12000], prompt, openai_api_key, model)
    return {
        "video_id":   video_id,
        "transcript": transcript[:500] + "..." if len(transcript) > 500 else transcript,
        "summary":    summary,
        "thumb":      f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg",
    }


# ════════════════════════════════════════════════════════
#  FEATURE 4 — EMAIL SENDER
# ════════════════════════════════════════════════════════

def generate_email_body(topic: str, content: str, tone: str,
                        openai_api_key: str, model: str) -> str:
    tone_map = {
        "professional": "Write a formal, professional email.",
        "friendly":     "Write a warm, friendly email.",
        "brief":        "Write a very short, direct email (max 5 sentences).",
        "newsletter":   "Write an engaging newsletter-style email with sections and bullet points.",
    }
    prompt = (
        f"{tone_map.get(tone, tone_map['professional'])}\n"
        f"Topic/Subject: {topic}\n"
        f"Key content to include: {content}\n\n"
        "Write only the email body (no subject line). Use markdown for formatting if appropriate.\n\n"
        "Email body:"
    )
    llm   = get_llm(openai_api_key, model)
    chain = PromptTemplate(input_variables=["text"], template="{text}") | llm | StrOutputParser()
    return chain.invoke({"text": prompt})


def send_email(smtp_host: str, smtp_port: int, sender: str, password: str,
               recipient: str, subject: str, body: str) -> dict:
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = sender
        msg["To"]      = recipient
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            server.login(sender, password)
            server.sendmail(sender, recipient, msg.as_string())
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ════════════════════════════════════════════════════════
#  FLASK ROUTES
# ════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")


# ── Research ──────────────────────────────────────────
@app.route("/research", methods=["POST"])
def research():
    data        = request.get_json()
    topic       = data.get("topic", "").strip()
    openai_key  = data.get("openai_key", "").strip()
    tavily_key  = data.get("tavily_key", "").strip()
    model       = data.get("model", "gpt-4o")
    max_results = int(data.get("max_results", 5))
    style       = data.get("style", "structured")

    if not all([topic, openai_key, tavily_key]):
        return {"error": "Missing fields."}, 400

    def generate():
        try:
            yield sse("status", {"msg": f"Searching the web for: {topic}"})
            agent      = build_research_agent(openai_key, tavily_key, max_results, model)
            raw_output = run_agent(agent, f"Research this topic thoroughly: {topic}")
            yield sse("status", {"msg": "Generating summary..."})
            summary = lcel_summarise(raw_output, STYLE_PROMPTS.get(style, STYLE_PROMPTS["structured"]), openai_key, model)
            sources = extract_sources(raw_output)
            yield sse("done", {"summary": summary, "sources": sources,
                               "topic": topic, "model": model, "style": style})
        except Exception as e:
            yield sse("error", {"msg": str(e)})

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── News ──────────────────────────────────────────────
@app.route("/news", methods=["POST"])
def news():
    data        = request.get_json()
    topic       = data.get("topic", "").strip()
    openai_key  = data.get("openai_key", "").strip()
    tavily_key  = data.get("tavily_key", "").strip()
    model       = data.get("model", "gpt-4o")
    max_results = int(data.get("max_results", 6))

    if not all([topic, openai_key, tavily_key]):
        return {"error": "Missing fields."}, 400

    def generate():
        try:
            yield sse("status", {"msg": f"Scraping latest news for: {topic}"})
            result = scrape_news(topic, openai_key, tavily_key, max_results, model)
            yield sse("status", {"msg": "Summarising articles..."})
            yield sse("done", {**result, "topic": topic})
        except Exception as e:
            yield sse("error", {"msg": str(e)})

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── YouTube ───────────────────────────────────────────
@app.route("/youtube", methods=["POST"])
def youtube():
    data       = request.get_json()
    url        = data.get("url", "").strip()
    openai_key = data.get("openai_key", "").strip()
    model      = data.get("model", "gpt-4o")
    style      = data.get("style", "structured")

    if not all([url, openai_key]):
        return {"error": "Missing fields."}, 400

    def generate():
        try:
            yield sse("status", {"msg": "Fetching YouTube transcript..."})
            result = summarise_youtube(url, openai_key, model, style)
            if "error" in result:
                yield sse("error", {"msg": result["error"]})
            else:
                yield sse("status", {"msg": "Summarising video..."})
                yield sse("done", result)
        except Exception as e:
            yield sse("error", {"msg": str(e)})

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ── Email ─────────────────────────────────────────────
@app.route("/email/generate", methods=["POST"])
def email_generate():
    data       = request.get_json()
    topic      = data.get("topic", "").strip()
    content    = data.get("content", "").strip()
    tone       = data.get("tone", "professional")
    openai_key = data.get("openai_key", "").strip()
    model      = data.get("model", "gpt-4o")

    if not all([topic, openai_key]):
        return {"error": "Missing fields."}, 400

    try:
        body = generate_email_body(topic, content, tone, openai_key, model)
        return {"body": body}
    except Exception as e:
        return {"error": str(e)}, 500


@app.route("/email/send", methods=["POST"])
def email_send():
    data      = request.get_json()
    result    = send_email(
        smtp_host = data.get("smtp_host", "smtp.gmail.com"),
        smtp_port = int(data.get("smtp_port", 465)),
        sender    = data.get("sender", "").strip(),
        password  = data.get("password", "").strip(),
        recipient = data.get("recipient", "").strip(),
        subject   = data.get("subject", "").strip(),
        body      = data.get("body", "").strip(),
    )
    return result


if __name__ == "__main__":
    app.run(debug=True, port=5000)