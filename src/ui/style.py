"""Visual system for the console: a model risk control ledger, not a SaaS card kit.

Palette (named): ink #0B1F33, paper #F5F7F9, rule #D3DAE1, regulator blue #1F4E8C, and the rating colours
green #2E7D32, amber #B7791F, red #B42318 (darkened for contrast on white). Type: IBM Plex Sans for text,
IBM Plex Mono for IDs, hashes and figures. The one bold element is the overall rating band.
"""
from __future__ import annotations

import html

import altair as alt
import streamlit as st

INK, PAPER, RULE, MUTED, BLUE = "#0B1F33", "#F5F7F9", "#D3DAE1", "#5B6B7C", "#1F4E8C"
RAG = {"green": "#2E7D32", "amber": "#B7791F", "red": "#B42318"}
SERIES = [BLUE, "#7A8FA6", "#0F766E", "#9A6700", "#6B4C9A"]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
/* Text elements only: a broad selector would also replace Streamlit's icon font and print icon names. */
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp input, .stApp textarea,
.stApp button p, .stApp td, .stApp th {{
  font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif;
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
code, .mono {{ font-family: 'IBM Plex Mono', ui-monospace, Consolas, monospace; font-size: 0.86em; }}
.block-container {{ padding-top: 2.2rem; max-width: 1240px; }}
h1 {{ font-weight: 700; letter-spacing: -0.02em; font-size: 2.1rem; color: {INK}; margin-bottom: 0.1rem; }}
h2, h3 {{ font-weight: 600; letter-spacing: -0.01em; color: {INK}; }}
p, li {{ max-width: 80ch; }}
section[data-testid="stSidebar"] {{ background: {INK}; }}
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] small, section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
  color: #E6ECF2 !important; }}
section[data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: #13283F; border-color: #2B4560; }}
section[data-testid="stSidebar"] [data-baseweb="select"] * {{ color: #E6ECF2 !important; }}
section[data-testid="stSidebar"] hr {{ border-color: #2B4560; }}
/* Sidebar text is light, so sidebar buttons need a dark face to stay readable. */
section[data-testid="stSidebar"] .stButton button {{ background: #13283F; border-color: #2B4560; }}
section[data-testid="stSidebar"] .stButton button:hover {{ border-color: #E6ECF2; }}
.stTabs [data-baseweb="tab-list"] {{ border-bottom: 1px solid {RULE}; gap: 1.6rem; }}
.stTabs [data-baseweb="tab"] {{ padding-left: 0; padding-right: 0; font-weight: 500; }}
.stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {{ border-radius: 3px; font-weight: 600; }}
*:focus-visible {{ outline: 2px solid {BLUE} !important; outline-offset: 2px; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}

.eyebrow {{ color: {MUTED}; font-size: 0.9rem; margin: 0 0 1.4rem 0; }}
.kpis {{ display: flex; flex-wrap: wrap; border-top: 2px solid {INK}; border-bottom: 1px solid {RULE}; margin: 0.4rem 0 1.6rem; }}
.kpi {{ flex: 1 1 140px; padding: 0.8rem 1.2rem 0.9rem 0; margin-right: 1.2rem; border-right: 1px solid {RULE}; }}
.kpi:last-child {{ border-right: none; }}
.kpi b {{ display: block; font: 600 2rem/1.1 'IBM Plex Mono', monospace; color: {INK}; }}
.kpi span {{ color: {MUTED}; font-size: 0.86rem; }}

.ledger-wrap {{ overflow-x: auto; margin-bottom: 0.4rem; }}
table.ledger {{ width: 100%; min-width: 980px; border-collapse: collapse; font-size: 0.9rem; }}
table.ledger td.model span {{ white-space: nowrap; }}
table.ledger th {{ text-align: left; font-weight: 500; color: {MUTED}; padding: 0.4rem 0.5rem; border-bottom: 1px solid {INK}; white-space: nowrap; }}
table.ledger th .n {{ font-family: 'IBM Plex Mono', monospace; color: {BLUE}; margin-right: 0.35rem; }}
table.ledger td {{ padding: 0.55rem 0.5rem; border-bottom: 1px solid {RULE}; vertical-align: middle; }}
table.ledger td.model b {{ display: block; color: {INK}; }}
table.ledger td.model span {{ font-family: 'IBM Plex Mono', monospace; color: {MUTED}; font-size: 0.78rem; }}
.stamp {{ display: inline-block; padding: 0.1rem 0.45rem; border: 1.5px solid; border-radius: 2px; font-size: 0.78rem; font-weight: 600; white-space: nowrap; }}
.stamp.done {{ color: {RAG['green']}; border-color: {RAG['green']}; }}
.stamp.approved {{ color: #fff; background: {RAG['green']}; border-color: {RAG['green']}; }}
.stamp.rejected {{ color: {RAG['red']}; border-color: {RAG['red']}; }}
.stamp.waiting {{ color: #fff; background: {RAG['amber']}; border-color: {RAG['amber']}; }}
.stamp.next {{ color: {BLUE}; border-color: {BLUE}; border-style: dashed; }}
.stamp.error {{ color: #fff; background: {RAG['red']}; border-color: {RAG['red']}; }}
.stamp.none {{ color: {RULE}; border-color: transparent; }}

.band {{ display: flex; align-items: stretch; margin: 0.6rem 0 1.4rem; color: #fff; }}
.band .rating {{ padding: 1.1rem 1.6rem; min-width: 200px; }}
.band .rating small {{ display: block; opacity: 0.85; font-size: 0.82rem; }}
.band .rating b {{ font-size: 2.3rem; letter-spacing: 0.04em; }}
.band .counts {{ display: flex; flex: 1; background: {INK}; }}
.band .counts div {{ flex: 1; padding: 1.1rem 1.2rem; border-left: 1px solid rgba(255,255,255,0.15); }}
.band .counts b {{ display: block; font: 600 1.7rem/1.1 'IBM Plex Mono', monospace; }}
.band .counts span {{ font-size: 0.82rem; opacity: 0.85; }}
.band .counts i {{ display: inline-block; width: 0.6rem; height: 0.6rem; margin-right: 0.35rem; border-radius: 50%; }}

.stepper {{ display: flex; flex-wrap: wrap; gap: 0; border-top: 1px solid {RULE}; border-bottom: 1px solid {RULE}; margin: 0.5rem 0 1.4rem; }}
.stepper div {{ flex: 1 1 110px; padding: 0.6rem 0.6rem 0.65rem; border-right: 1px solid {RULE}; font-size: 0.86rem; color: {MUTED}; }}
.stepper div:last-child {{ border-right: none; }}
.stepper div .n {{ font-family: 'IBM Plex Mono', monospace; display: block; font-size: 0.75rem; }}
.stepper div.done {{ color: {INK}; }}
.stepper div.done .n {{ color: {RAG['green']}; }}
.stepper div.current {{ color: {INK}; font-weight: 600; box-shadow: inset 0 -3px 0 {BLUE}; }}
.stepper div.current.waiting {{ box-shadow: inset 0 -3px 0 {RAG['amber']}; }}
.stepper div.rejected .n {{ color: {RAG['red']}; }}
.note {{ border-left: 3px solid {BLUE}; padding: 0.3rem 0 0.3rem 0.9rem; color: {MUTED}; margin: 0.6rem 0 1rem; max-width: 80ch; }}
</style>
"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def esc(value) -> str:
    return html.escape(str(value))


def kpi_strip(items: list[tuple[str, str]]) -> None:
    cells = "".join(f"<div class='kpi'><b>{esc(v)}</b><span>{esc(k)}</span></div>" for k, v in items)
    st.markdown(f"<div class='kpis'>{cells}</div>", unsafe_allow_html=True)


def stamp(kind: str, text: str) -> str:
    return f"<span class='stamp {kind}'>{esc(text)}</span>"


def rating_band(overall: str, counts: dict[str, int]) -> None:
    colour = RAG.get(overall, MUTED)
    parts = "".join(f"<div><b>{counts.get(r, 0)}</b><span><i style='background:{RAG[r]}'></i>{r.title()} "
                    f"checks</span></div>" for r in ("green", "amber", "red"))
    st.markdown(f"<div class='band' role='status' aria-label='Overall rating {esc(overall)}'>"
                f"<div class='rating' style='background:{colour}'><small>Overall rating</small>"
                f"<b>{esc(overall.title())}</b></div><div class='counts'>{parts}</div></div>",
                unsafe_allow_html=True)


def note(text: str) -> None:
    st.markdown(f"<div class='note'>{esc(text)}</div>", unsafe_allow_html=True)


@alt.theme.register("ledger", enable=True)
def ledger_theme() -> alt.theme.ThemeConfig:
    font = "IBM Plex Sans, system-ui, sans-serif"
    return {"config": {
        "font": font, "background": "transparent", "view": {"stroke": "transparent"},
        "title": {"font": font, "fontSize": 14, "fontWeight": 600, "color": INK, "anchor": "start"},
        "axis": {"labelColor": MUTED, "titleColor": MUTED, "gridColor": "#E3E8ED", "domainColor": RULE,
                 "tickColor": RULE, "labelFont": font, "titleFont": font, "titleFontWeight": 500},
        "legend": {"labelColor": MUTED, "titleColor": MUTED, "labelFont": font, "titleFont": font},
        "range": {"category": SERIES}, "line": {"color": BLUE}, "bar": {"color": BLUE},
    }}
