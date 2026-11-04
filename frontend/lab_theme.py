"""Local-only visual layer for the synthetic data laboratory."""
import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


ASSETS = Path(__file__).resolve().parent / "assets"


def render_lab_theme():
    st.html("<style>" + (ASSETS / "lab.css").read_text(encoding="utf-8") + "</style>")
    st.html('''
        <div class="sdf-liquid" aria-hidden="true">
            <div class="sdf-fluid sdf-fluid-purple"></div>
            <div class="sdf-fluid sdf-fluid-green"></div>
            <div class="sdf-fluid sdf-fluid-violet"></div>
            <div class="sdf-bubble b1"></div><div class="sdf-bubble b2"></div>
            <div class="sdf-bubble b3"></div><div class="sdf-bubble b4"></div>
            <div class="sdf-bubble b5"></div><div class="sdf-bubble b6"></div>
            <div class="sdf-bubble b7"></div><div class="sdf-bubble b8"></div>
        </div>
    ''')


def render_lab_header():
    flask_svg = base64.b64encode((ASSETS / "flask.svg").read_bytes()).decode("ascii")
    st.html('''
        <header class="sdf-hero">
            <div class="sdf-eyebrow"><span class="sdf-status-dot"></span> SYNTHETIC DATA LAB <span class="sdf-edition">EXPERIMENT / 001</span></div>
            <h1>Synthetic Data<br><span>Factory.</span></h1>
            <p>Real patterns. New possibilities.<br>Generate synthetic data, then put it to the test.</p>
            <div class="sdf-tags"><span>01 / Statistical similarity</span><span>02 / ML utility</span><span>03 / Copying signals</span></div>
            <div class="sdf-flask" aria-hidden="true">
                <img src="data:image/svg+xml;base64,{flask_svg}" alt="" width="210" height="240">
                <span>IDEAS IN REACTION</span>
            </div>
        </header>
    '''.replace("{flask_svg}", flask_svg))
    # This tiny local component owns animation controls and pointer listeners.
    # It never reads dataset contents or sends requests.
    components.html((ASSETS / "lab_controls.html").read_text(encoding="utf-8"), height=64)
