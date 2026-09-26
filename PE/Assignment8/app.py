"""
Streamlit Web Application
AI Content Creation & Analysis System

Minimal, light-themed interface.
"""

import streamlit as st
import pandas as pd
import json
import matplotlib.pyplot as plt

from prompts import (
    STORY_PROMPT,
    POEM_PROMPT,
    SOCIAL_MEDIA_PROMPT,
    PODCAST_PROMPT,
    TEXT_ANALYSIS_PROMPT,
    EXPERIMENTATION_PROMPT,
    build_full_prompt
)
from llm_handler import LLMHandler

# Page Configuration
st.set_page_config(
    page_title="AI Content Creation & Analysis System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Light Theme CSS
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #FAFAFA;
        color: #18181B;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Headers & Subheadings */
    h1, h2, h3, h4 {
        color: #09090B !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
    }

    /* Minimal Header Styling */
    .app-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: #09090B;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #71717A;
        margin-bottom: 1.5rem;
    }

    /* Container & Cards */
    .content-card {
        background: #FFFFFF;
        border: 1px solid #E4E4E7;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Minimal Code Block Styling */
    .stCodeBlock {
        border: 1px solid #E4E4E7 !important;
        border-radius: 6px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F4F4F5;
        border-right: 1px solid #E4E4E7;
    }

    /* Metric Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #09090B !important;
    }

    /* Button Styling */
    .stButton>button {
        background-color: #18181B !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        transition: opacity 0.2s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #E4E4E7;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 500;
        color: #71717A;
        border-radius: 4px;
    }
    .stTabs [aria-selected="true"] {
        color: #09090B !important;
        background-color: #F4F4F5 !important;
    }
</style>
""", unsafe_allow_html=True)

# Minimal Page Header
st.markdown('<div class="app-title">AI Content Creation & Analysis System</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Structured Prompt Engineering, Automated Generation, Podcast Planning, NLP Analysis, and Parameter Evaluation</div>', unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("### Model Configuration")
    
    provider_choice = st.selectbox(
        "LLM Provider",
        ["Auto-Detect", "Google Gemini", "OpenAI", "Offline Mode"],
        index=0
    )
    
    provider_map = {
        "Auto-Detect": "auto",
        "Google Gemini": "gemini",
        "OpenAI": "openai",
        "Offline Mode": "mock"
    }
    selected_provider = provider_map[provider_choice]

    api_key_input = st.text_input(
        "API Key (Optional)",
        type="password",
        help="Leave empty to use environment variables or offline fallback."
    )

    st.markdown("---")
    st.markdown("### Default Parameters")
    default_temp = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05, help="Controls output randomness.")
    default_top_p = st.slider("Top-P", 0.1, 1.0, 0.95, 0.05, help="Nucleus sampling probability threshold.")

# Initialize LLM Handler
handler = LLMHandler(provider=selected_provider, api_key=api_key_input if api_key_input else None)

# Main Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Prompt Design",
    "Content Generation",
    "Podcast Planning",
    "Text Analysis",
    "Parameter Experimentation"
])

# ---------------------------------------------------------------------------
# TAB 1: PROMPT DESIGN
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("System Prompt Design Architecture")
    st.caption("Demonstrating structured system prompts engineered with explicit role, context, constraints, and format specifications.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("##### Prompt Architecture Components")
        st.markdown("""
        * **Role:** Defines the specific persona assigned to the model.
        * **Context:** Provides domain background and audience scope.
        * **Constraints:** Enforces strict boundary rules, lengths, and tone.
        * **Output Format:** Dictates structural layout or JSON schema.
        * **User Input:** Injects dynamic topic payload at runtime.
        """)
        
        selected_task_demo = st.selectbox(
            "Select Template",
            ["Story Generation", "Poem Generation", "Social Media Campaign", "Podcast Planning", "Text Analysis (JSON Schema)"]
        )

        template_map = {
            "Story Generation": STORY_PROMPT,
            "Poem Generation": POEM_PROMPT,
            "Social Media Campaign": SOCIAL_MEDIA_PROMPT,
            "Podcast Planning": PODCAST_PROMPT,
            "Text Analysis (JSON Schema)": TEXT_ANALYSIS_PROMPT
        }
        cfg = template_map[selected_task_demo]

    with col2:
        st.markdown("##### Compiled System Prompt")
        sample_topic = "Artificial Intelligence in Renewable Energy"
        full_compiled_prompt = build_full_prompt(cfg, sample_topic)
        st.code(full_compiled_prompt, language="markdown")

    st.markdown("---")
    st.markdown("##### Structured Breakdown")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"**Role:**\n{cfg['role']}")
        st.markdown(f"**Context:**\n{cfg['context']}")
    with c2:
        st.markdown(f"**Constraints:**\n{cfg['constraints']}")
    with c3:
        st.markdown(f"**Output Format:**\n```\n{cfg['output_format']}\n```")

# ---------------------------------------------------------------------------
# TAB 2: CONTENT GENERATION
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Content Generation")
    st.caption("Generate structured narrative stories, formal poems, or social media campaigns.")

    topic_input = st.text_input("Topic", "Future of Space Exploration and Mars Colonization")
    
    col_type, col_temp = st.columns([2, 1])
    with col_type:
        content_type = st.radio("Content Format", ["Story", "Poem", "Social Media Post"], horizontal=True)
    with col_temp:
        gen_temp = st.slider("Generation Temperature", 0.1, 1.2, default_temp, key="gen_temp")

    if st.button("Generate Content", type="primary"):
        with st.spinner("Generating response..."):
            if content_type == "Story":
                p_cfg = STORY_PROMPT
            elif content_type == "Poem":
                p_cfg = POEM_PROMPT
            else:
                p_cfg = SOCIAL_MEDIA_PROMPT
                
            compiled_prompt = build_full_prompt(p_cfg, topic_input)
            response_text, meta = handler.generate(compiled_prompt, temperature=gen_temp, top_p=default_top_p)

            st.markdown("##### Output")
            st.markdown(response_text)
            st.caption(f"Engine: {meta['provider']} | Model: {meta['model']} | Latency: {meta['latency_seconds']}s")

# ---------------------------------------------------------------------------
# TAB 3: PODCAST PLANNING
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Podcast Planning & Interview Design")
    st.caption("Generate episode titles, description summaries, target guest specifications, and interview questions.")

    podcast_topic = st.text_input("Podcast Topic", "Ethical AI and Prompt Engineering", key="pod_topic")

    if st.button("Generate Podcast Plan", type="primary"):
        with st.spinner("Generating podcast plan..."):
            pod_prompt = build_full_prompt(PODCAST_PROMPT, podcast_topic)
            pod_res, pod_meta = handler.generate(pod_prompt, temperature=default_temp, top_p=default_top_p)

            st.markdown("##### Episode Plan")
            st.markdown(pod_res)
            st.caption(f"Engine: {pod_meta['provider']} | Latency: {pod_meta['latency_seconds']}s")

# ---------------------------------------------------------------------------
# TAB 4: TEXT ANALYSIS
# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Text Analysis & Keyword Extraction")
    st.caption("Evaluate textual sentiment scores (-1.0 to +1.0) and extract relevant keywords.")

    sample_default_text = (
        "Prompt engineering has completely transformed how developers build software with artificial intelligence. "
        "Although mastering structured prompt design requires careful context formulation, the immediate gains in developer productivity, "
        "output quality, and user satisfaction make it an invaluable skill."
    )

    user_analysis_text = st.text_area("Input Text", sample_default_text, height=130)

    if st.button("Analyze Text", type="primary"):
        with st.spinner("Analyzing text..."):
            analysis_prompt = build_full_prompt(TEXT_ANALYSIS_PROMPT, user_analysis_text)
            res_raw, meta = handler.generate(analysis_prompt, temperature=0.2, top_p=0.8)

            try:
                clean_json_str = res_raw.strip()
                if "```json" in clean_json_str:
                    clean_json_str = clean_json_str.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_json_str:
                    clean_json_str = clean_json_str.split("```")[1].split("```")[0].strip()

                parsed = json.loads(clean_json_str)

                sentiment = parsed.get("sentiment", {})
                score = sentiment.get("score", 0.0)
                label = sentiment.get("label", "Neutral")
                explanation = sentiment.get("explanation", "")
                keywords = parsed.get("keywords", [])

                col_s1, col_s2, col_s3 = st.columns(3)
                with col_s1:
                    st.metric("Sentiment Label", label)
                with col_s2:
                    st.metric("Sentiment Score", f"{score:+.2f}")
                with col_s3:
                    st.metric("Keywords Extracted", len(keywords))

                st.markdown(f"**Rationale:** {explanation}")

                if keywords:
                    st.markdown("##### Keyword Relevance Distribution")
                    kw_df = pd.DataFrame(keywords)
                    kw_df["keyword_short"] = kw_df["keyword"].apply(lambda x: x[:30] + "..." if len(x) > 30 else x)
                    
                    fig, ax = plt.subplots(figsize=(7, 2.8))
                    fig.patch.set_facecolor('#FFFFFF')
                    ax.set_facecolor('#FFFFFF')
                    
                    bars = ax.barh(kw_df["keyword_short"], kw_df["relevance"], color='#2563EB', height=0.55)
                    ax.set_xlabel("Relevance (0.0 to 1.0)", fontsize=9, color='#71717A')
                    ax.set_xlim(0, 1.0)
                    ax.invert_yaxis()
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_color('#E4E4E7')
                    ax.spines['bottom'].set_color('#E4E4E7')
                    ax.tick_params(colors='#71717A', labelsize=9)
                    
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.2f}', ha='left', va='center', fontsize=8.5, color='#09090B')
                    st.pyplot(fig)

            except Exception:
                st.code(res_raw, language="json")

# ---------------------------------------------------------------------------
# TAB 5: PARAMETER EXPERIMENTATION
# ---------------------------------------------------------------------------
with tab5:
    st.subheader("LLM Parameter Evaluation")
    st.caption("Compare how altering Temperature and Top-P impacts output determinism versus creativity.")

    exp_topic = st.text_input("Prompt", "Describe Artificial Intelligence in 3 sentences.", key="exp_topic")

    cp1, cp2, cp3 = st.columns(3)
    with cp1:
        st.markdown("##### Profile 1: Low Temperature")
        t1 = st.slider("Temp 1", 0.0, 1.5, 0.1, 0.1, key="t1")
        p1 = st.slider("Top-P 1", 0.1, 1.0, 0.5, 0.1, key="p1")

    with cp2:
        st.markdown("##### Profile 2: Medium Temperature")
        t2 = st.slider("Temp 2", 0.0, 1.5, 0.7, 0.1, key="t2")
        p2 = st.slider("Top-P 2", 0.1, 1.0, 0.9, 0.1, key="p2")

    with cp3:
        st.markdown("##### Profile 3: High Temperature")
        t3 = st.slider("Temp 3", 0.0, 1.5, 1.2, 0.1, key="t3")
        p3 = st.slider("Top-P 3", 0.1, 1.0, 0.98, 0.1, key="p3")

    if st.button("Compare Parameters", type="primary"):
        with st.spinner("Generating profile responses..."):
            exp_compiled_prompt = build_full_prompt(EXPERIMENTATION_PROMPT, exp_topic)

            r1, m1 = handler.generate(exp_compiled_prompt, temperature=t1, top_p=p1)
            r2, m2 = handler.generate(exp_compiled_prompt, temperature=t2, top_p=p2)
            r3, m3 = handler.generate(exp_compiled_prompt, temperature=t3, top_p=p3)

            col_out1, col_out2, col_out3 = st.columns(3)
            with col_out1:
                st.markdown(f"**Low Temp** (`{t1}`, Top-P `{p1}`)")
                st.text_area("Response 1", r1, height=140, key="r1_view", disabled=True)
            with col_out2:
                st.markdown(f"**Medium Temp** (`{t2}`, Top-P `{p2}`)")
                st.text_area("Response 2", r2, height=140, key="r2_view", disabled=True)
            with col_out3:
                st.markdown(f"**High Temp** (`{t3}`, Top-P `{p3}`)")
                st.text_area("Response 3", r3, height=140, key="r3_view", disabled=True)
