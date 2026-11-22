"""
GlassBox audit dashboard (Streamlit).

    streamlit run glassbox/src/dashboard/app.py

Pages
    Ask             type a question (or pick a QA-set question); the pipeline runs and its
                    trace is shown and saved.
    Stored traces   table of saved traces (filter by verdict / type / abstention); pick
                    one to audit it. Reads trace files only, no recomputation.

Trace view: one panel per pipeline stage (question redaction, query analysis,
retrieval, answer, verification) plus timings and the raw trace JSON.

Privacy: the dashboard shows what the trace holds, i.e. redacted text and
placeholders. The original question typed on the Ask page is shown back only
in this local session and is never written to a trace. Raw-baseline traces
(no privacy by design) are labelled as such.
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from glassbox.src.dashboard.traces import (ABSTAIN_EXPLANATION, LABEL_COLOUR, TRACE_ROOT,  # noqa: E402
                                           VERDICT_COLOUR, chunk_anchor, load_rows, load_trace,
                                           source_label, trace_sets)

st.set_page_config(page_title="GlassBox audit", page_icon="🔍", layout="wide")
TRACES = ROOT / TRACE_ROOT


def badge(text: str, colour: str) -> str:
    return f":{colour}-badge[{text}]"


# ---------------------------------------------------------------- trace view

def show_question(trace, original=None):
    q, cfg = trace["question"], trace["config"]
    st.subheader("1 · Question redaction")
    if cfg.get("baseline_mode"):
        st.warning("Raw baseline: no question redaction and an un-redacted index (no privacy, for comparison only).")
    c1, c2 = st.columns(2)
    if original is not None:
        c1.markdown("**As typed** (this session only, never stored)")
        c1.code(original, language=None, wrap_lines=True)
    c2.markdown("**As processed by every later stage**")
    c2.code(q["text"], language=None, wrap_lines=True)
    if q["redacted"]:
        st.markdown("Removed: " + " ".join(badge(r["type"], "violet") for r in q["redacted"]))
    else:
        st.caption("No personal identifiers found in the question.")
    gen = trace["generation"]
    decision = gen.get("guard_decision")
    if decision == "abstain":
        st.error(f"**Abstain guard: refused** ({gen['abstain_reason']}). {ABSTAIN_EXPLANATION.get(gen['abstain_reason'], '')}")
    elif decision == "disabled":
        st.warning("Abstain guard **disabled** for this configuration (shows what the model does on its own).")
    else:
        st.success("Abstain guard: passed. The question does not identify a person and its conditions match records.")


def show_analysis(trace):
    a = trace["query_analysis"]
    st.subheader("2 · Query analysis")
    filters = a.get("filters") or {}
    chips = [badge(f"{k}: {v}", "blue") for k, v in filters.items()]
    st.markdown("Filters: " + (" ".join(chips) if chips else "_none_"))
    c1, c2, c3 = st.columns(3)
    c1.metric("Scope", a.get("scope", "?"))
    c2.metric("Chunks retrieved (k)", trace["retrieval"]["k"])
    c3.metric("Fell back to unfiltered", "yes" if a.get("fallback") else "no")
    st.markdown(f"**Ranking query:** `{a.get('ranking_query', '')}`")
    if a.get("ask"):
        st.markdown(f"**Two-hop split:** ask = `{a['ask']}` · condition = `{a.get('condition', '')}`")
    if a.get("extracted"):
        with st.expander("Phrases turned into filters"):
            st.json(a["extracted"])


def show_retrieval(trace):
    r = trace["retrieval"]
    labels = source_label(trace)
    cited = {cid for c in trace["generation"].get("claims") or [] for cid in c["citations"]}
    st.subheader("3 · Retrieval")
    st.caption(f"{r['k']} chunks, {r['context_chars']:,} characters shown to the model · index "
               f"`{trace['config']['index']}` · hybrid {trace['config']['fusion_weights']} (dense:BM25)")
    for c in r["chunks"]:
        label = labels.get(c["chunk_id"], "")
        st.markdown(f'<div id="{chunk_anchor(c["chunk_id"])}"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            head = [f"**#{c['rank']}**", f"`{label}`" if label else "", f"**{c['section']}**", f"`{c['doc_id']}`",
                    badge(f"selected by {c['selected_by']}", "gray")]
            if c["chunk_id"] in cited:
                head.append(badge("cited", "green"))
            if c.get("low_information"):
                head.append(badge("low information", "orange"))
            st.markdown(" ".join(h for h in head if h))
            st.caption(f"fused {c['score']:.4f} · dense rank {c['dense_rank']} ({c['dense_score']}) · "
                       f"BM25 rank {c['bm25_rank']} ({c['bm25_score']})")
            st.text(c["text"])


def show_answer(trace):
    gen, ver = trace["generation"], trace["verification"]
    st.subheader("4 · Answer and 5 · Verification")
    st.markdown("Answer verdict: " + badge(ver["verdict"], VERDICT_COLOUR.get(ver["verdict"], "gray")))
    if gen["abstained"]:
        st.info(f"**{gen['answer']}**  \nReason: `{gen['abstain_reason']}`")
    elif not gen.get("ok"):
        st.error("The model's reply could not be parsed (after one retry). See the raw output below.")
    else:
        st.markdown(f"> {gen['answer']}")
    labels = source_label(trace)
    verdicts = {v["claim"]: v for v in ver.get("claims", [])}
    for i, c in enumerate(gen.get("claims") or [], 1):
        v = verdicts.get(c["text"], {})
        lab = v.get("label")
        links = " ".join(f"[{labels.get(cid, cid)}](#{chunk_anchor(cid)})" for cid in c["citations"]) or "_no citation_"
        st.markdown(f"**Claim {i}.** {c['text']}  \nCites {links} "
                    + (badge(lab, LABEL_COLOUR.get(lab, "gray")) if lab else ""))
        if v.get("probabilities"):
            p = v["probabilities"]
            st.caption(f"NLI: entailment {p['entailment']:.3f} · neutral {p['neutral']:.3f} · "
                       f"contradiction {p['contradiction']:.3f} · premise {', '.join(labels.get(x, x) for x in v['premise_chunks'])}"
                       + (f" · checked as: “{v['checked_as']}”" if v.get("checked_as") != v.get("claim") else ""))
        if c.get("invalid_citations"):
            st.caption(f"Dropped invalid citations: {', '.join(c['invalid_citations'])}")
    if gen.get("model_called"):
        with st.expander(f"Raw model output ({gen.get('attempts')} attempt(s), "
                         f"{gen.get('prompt_tokens')} prompt + {gen.get('generation_tokens')} generated tokens)"):
            for out in gen.get("raw_outputs") or []:
                st.code(out, language="json", wrap_lines=True)
            if gen.get("errors"):
                st.caption("Parse errors: " + "; ".join(gen["errors"]))


def show_trace(trace, original=None):
    cfg = trace["config"]
    st.caption(f"Trace `{trace['trace_id']}` · {trace['timestamp']} · config **{cfg['name']}** · "
               f"{cfg['generator_model']} (prompt {cfg['prompt_version']}) · verifier {cfg['verifier_model']}")
    if "qa" in trace:
        s = trace["qa"]["scores"]
        st.caption(f"QA question `{trace['qa']['qid']}` ({trace['qa']['type']}, {trace['qa']['split']}) · "
                   f"grounded accuracy {s.get('grounded_accuracy')} · gold notes {', '.join(trace['qa']['gold_doc_ids'])}")
    t = trace["timings_ms"]
    cols = st.columns(len(t))
    for col, (stage, ms) in zip(cols, t.items()):
        col.metric(stage.replace("_", " "), f"{ms / 1000:.2f} s" if ms >= 1000 else f"{ms:.0f} ms")
    show_question(trace, original)
    st.divider()
    show_analysis(trace)
    st.divider()
    left, right = st.columns([1, 1], gap="large")
    with left:
        show_answer(trace)
    with right:
        show_retrieval(trace)
    with st.expander("Full trace JSON"):
        st.json(trace, expanded=False)


# ---------------------------------------------------------------- pages

@st.cache_resource(show_spinner="Loading the retriever, the 7B generator and the NLI verifier…")
def get_pipeline(dataset: str, config: str):
    from glassbox.src.pipeline import GlassBox
    return GlassBox(dataset, config, trace_dir=TRACES / dataset / config)


@st.cache_data
def qa_questions(dataset: str):
    from glassbox.src.qa.build_qa_set import load_qa_set
    from glassbox.src.retrieval.index import DATA_ROOT
    return load_qa_set(ROOT / DATA_ROOT / "raw" / dataset / "qa_eval_set.jsonl")


def page_ask(dataset: str, config: str):
    st.header("Ask a question")
    mode = st.radio("Question source", ["Type a question", "Pick from the QA set"], horizontal=True)
    q, qa_item = None, None
    if mode == "Type a question":
        q = st.text_input("Question", placeholder="Which medications were prescribed to patients who presented with fatigue?")
    else:
        qs = qa_questions(dataset)
        qtype = st.selectbox("Type", ["clinical_fact", "aggregate", "identity"])
        pool = [x for x in qs if x["type"] == qtype]
        qa_item = st.selectbox("Question", pool, format_func=lambda x: f"{x['qid']} · {x['question']}")
        q = qa_item["question"] if qa_item else None
    if st.button("Run pipeline", type="primary", disabled=not q):
        gb = get_pipeline(dataset, config)
        with st.spinner("Running: redaction → retrieval → generation → verification"):
            if qa_item:
                from glassbox.run import run_qa
                trace = run_qa(gb, qa_item)
            else:
                trace = gb.answer(q)
        st.session_state["last"] = (trace, q)
    if "last" in st.session_state:
        trace, original = st.session_state["last"]
        if trace["config"]["name"] == config:
            st.divider()
            show_trace(trace, original)


def page_traces(dataset: str, config: str):
    st.header("Stored traces")
    trace_set = f"{dataset}/{config}"
    rows = load_rows(trace_set, TRACES)
    if not rows:
        st.info(f"No traces in `{TRACE_ROOT / trace_set}`. Run `python -m glassbox.run --qa-set --config {config}`.")
        return
    c1, c2, c3 = st.columns(3)
    verdicts = c1.multiselect("Verdict", sorted({r["verdict"] for r in rows}))
    types = c2.multiselect("Question type", sorted({r["type"] for r in rows}))
    splits = c3.multiselect("Split", sorted({r["split"] for r in rows if r["split"]}))
    shown = [r for r in rows if (not verdicts or r["verdict"] in verdicts) and (not types or r["type"] in types)
             and (not splits or r["split"] in splits)]
    answerable = [r for r in shown if r["type"] not in ("identity", "ad hoc") and r["grounded_accuracy"] is not None]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Traces", len(shown))
    m2.metric("Verified", sum(r["verdict"] == "VERIFIED" for r in shown))
    m3.metric("Abstained", sum(r["verdict"] == "ABSTAINED" for r in shown))
    if answerable:
        m4.metric("Grounded accuracy (answerable)", f"{sum(r['grounded_accuracy'] for r in answerable) / len(answerable):.3f}")
    event = st.dataframe([{k: v for k, v in r.items() if k != "path"} for r in shown], hide_index=True,
                         width="stretch", on_select="rerun", selection_mode="single-row")
    sel = event.selection.rows if event and event.selection else []
    if sel:
        st.divider()
        show_trace(load_trace(Path(shown[sel[0]]["path"])))
    else:
        st.caption("Select a row to audit that trace.")


def main():
    st.sidebar.title("🔍 GlassBox")
    st.sidebar.caption("Privacy-preserving, verifiable question answering over clinical notes")
    page = st.sidebar.radio("Page", ["Ask", "Stored traces"])
    dataset = st.sidebar.selectbox("Dataset", ["realistic", "synthetic"])
    config = st.sidebar.selectbox("Configuration", ["redacted", "raw", "redacted_noguard"],
                                  help="redacted = the GlassBox system · raw = no-privacy baseline · "
                                       "redacted_noguard = abstain guard off")
    st.sidebar.caption("Sets on disk: " + (", ".join(trace_sets(TRACES)) or "none"))
    (page_ask if page == "Ask" else page_traces)(dataset, config)


if __name__ == "__main__":   # streamlit runs the script as __main__; tests import the functions
    main()
