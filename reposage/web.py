"""Phase 5 — FastAPI service + a minimal web UI.

  uvicorn reposage.web:app --port 8000

GET  /              -> the single-page UI
POST /api/ask       -> {question, agentic} -> {answer, sources, path}
GET  /api/health    -> liveness + whether an LLM key is configured
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from .config import config

app = FastAPI(title="RepoSage", description="Chat with a codebase — grounded, cited answers.")

_retriever = None


def get_retriever():
    """Lazy singleton — loads the embedding models on first request (not at import).
    On a fresh deploy, auto-index a repo if the collection is empty
    (set REPOSAGE_INDEX_REPO), reusing THIS client so Qdrant's local-mode lock is
    never double-held."""
    global _retriever
    if _retriever is None:
        import os
        from .retrieve import Retriever
        _retriever = Retriever()
        repo = os.getenv("REPOSAGE_INDEX_REPO")
        if repo:
            c = _retriever.client
            empty = (not c.collection_exists(config.collection)
                     or c.get_collection(config.collection).points_count == 0)
            if empty:
                from .index import index_chunks
                from .ingest import ingest_repo
                index_chunks(ingest_repo(repo), client=c)
    return _retriever


class AskRequest(BaseModel):
    question: str
    agentic: bool = False


@app.get("/api/health")
def health():
    return {"ok": True, "provider": config.llm_provider, "llm_ready": config.llm_ready}


@app.post("/api/ask")
def ask(req: AskRequest):
    r = get_retriever()
    hits = r.search(req.question, top_k=8)
    answer_text, path = "", "retrieval-only"

    if config.llm_ready:
        if req.agentic:
            from .agentic import agentic_answer
            answer_text, path = agentic_answer(req.question, r)
        else:
            from .generate import answer
            answer_text, path = answer(req.question, hits), "baseline"

    sources = [
        {"file": h.file_path, "lines": f"{h.start_line}-{h.end_line}",
         "symbol": h.symbol, "score": round(h.score, 3)}
        for h in hits
    ]
    return {"answer": answer_text, "sources": sources, "path": path, "llm_ready": config.llm_ready}


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


INDEX_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RepoSage — chat with a codebase</title>
<style>
  :root{--ink:#0f172a;--muted:#64748b;--border:#e2e8f0;--primary:#6366f1;--violet:#8b5cf6;--bg:#f3f5fb}
  *{box-sizing:border-box} body{margin:0;font-family:Inter,-apple-system,'Segoe UI',sans-serif;background:var(--bg);color:var(--ink)}
  .wrap{max-width:860px;margin:0 auto;padding:28px 20px 64px}
  header{display:flex;align-items:center;gap:12px;margin-bottom:6px}
  .mark{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;color:#fff;font-size:22px;
        background:linear-gradient(135deg,var(--primary),var(--violet));box-shadow:0 6px 16px rgba(99,102,241,.35)}
  h1{font-size:22px;margin:0;letter-spacing:-.3px} .sub{color:var(--muted);font-size:14px;margin:2px 0 22px}
  .card{background:#fff;border:1px solid var(--border);border-radius:16px;box-shadow:0 1px 2px rgba(15,23,42,.05),0 8px 24px rgba(15,23,42,.06);padding:18px}
  textarea{width:100%;border:1px solid var(--border);border-radius:10px;padding:12px;font-size:15px;font-family:inherit;resize:vertical;min-height:60px}
  textarea:focus{outline:none;border-color:var(--primary);box-shadow:0 0 0 3px rgba(99,102,241,.15)}
  .row{display:flex;align-items:center;justify-content:space-between;margin-top:12px;gap:12px}
  label.tog{display:flex;align-items:center;gap:7px;color:var(--muted);font-size:13.5px;cursor:pointer}
  button{cursor:pointer;border:0;border-radius:10px;color:#fff;font-weight:700;font-size:15px;padding:11px 22px;
         background:linear-gradient(135deg,var(--primary),var(--violet));box-shadow:0 6px 16px rgba(99,102,241,.3)}
  button:disabled{opacity:.55;cursor:default}
  .answer{white-space:pre-wrap;line-height:1.6;font-size:15px;margin-top:18px}
  .badge{display:inline-block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;
         color:#4f46e5;background:#eef2ff;border-radius:999px;padding:3px 10px;margin-bottom:10px}
  .srcs{margin-top:18px;border-top:1px solid var(--border);padding-top:14px}
  .srcs h3{font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:var(--muted);margin:0 0 8px}
  .src{font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:#334155;padding:5px 0;border-bottom:1px solid #f1f5f9}
  .src b{color:var(--ink)} .muted{color:var(--muted)}
  .hint{color:var(--muted);font-size:12.5px;margin-top:14px}
</style></head><body>
<div class="wrap">
  <header><div class="mark">🧙</div><div><h1>RepoSage</h1></div></header>
  <p class="sub">Ask about the indexed codebase. Answers are grounded in the actual source, with file:line citations.</p>
  <div class="card">
    <textarea id="q" placeholder="e.g. How is authentication implemented? How are atomic transfers done?"></textarea>
    <div class="row">
      <label class="tog"><input type="checkbox" id="agentic"> agentic mode (self-RAG loop for hard questions)</label>
      <button id="go" onclick="ask()">Ask</button>
    </div>
    <div id="out"></div>
  </div>
  <p class="hint">Hybrid retrieval (dense + BM25, RRF) → grounded generation. First query warms up the embedding model.</p>
</div>
<script>
async function ask(){
  const q=document.getElementById('q').value.trim(); if(!q)return;
  const agentic=document.getElementById('agentic').checked;
  const btn=document.getElementById('go'), out=document.getElementById('out');
  btn.disabled=true; out.innerHTML='<p class="muted" style="margin-top:18px">Thinking…</p>';
  try{
    const res=await fetch('/api/ask',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({question:q,agentic})});
    const d=await res.json();
    let html='';
    if(d.answer){html+='<span class="badge">routed: '+d.path+'</span><div class="answer">'+esc(d.answer)+'</div>';}
    else{html+='<p class="muted" style="margin-top:18px">No LLM key set — showing retrieved sources only.</p>';}
    if(d.sources&&d.sources.length){
      html+='<div class="srcs"><h3>Sources</h3>';
      for(const s of d.sources){html+='<div class="src"><b>'+esc(s.file)+'</b>:'+s.lines+(s.symbol?' <span class="muted">('+esc(s.symbol)+')</span>':'')+' <span class="muted">· '+s.score+'</span></div>';}
      html+='</div>';
    }
    out.innerHTML=html;
  }catch(e){out.innerHTML='<p class="muted" style="margin-top:18px">Error: '+esc(String(e))+'</p>';}
  btn.disabled=false;
}
function esc(s){return s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
document.getElementById('q').addEventListener('keydown',e=>{if(e.key==='Enter'&&(e.ctrlKey||e.metaKey))ask();});
</script>
</body></html>"""
