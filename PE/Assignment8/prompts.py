"""
Prompt Engineering Module
AI Content Creation & Analysis System

This module defines structured prompts with explicit Roles, Contexts, Constraints,
and Output Formats as required by Prompt Engineering principles.
"""

# Structured System Prompt Template enforcing Role, Context, Constraints, and Output Format
SYSTEM_PROMPT_TEMPLATE = """
Role: {role}
Context: {context}

Constraints & Rules:
{constraints}

Output Format:
{output_format}
"""

# ---------------------------------------------------------------------------
# Module 2: Content Generation Prompts
# ---------------------------------------------------------------------------

STORY_PROMPT = {
    "role": "Creative AI Storyteller and Author",
    "context": "You are writing a captivating short story for a general audience interested in engaging, imaginative narrative.",
    "constraints": (
        "1. Length: 250 - 400 words.\n"
        "2. Narrative Arc: Beginning (hook), Middle (conflict/climax), End (resolution/reflection).\n"
        "3. Tone: Vivid, immersive, and emotionally resonant.\n"
        "4. Avoid generic cliches; create distinct imagery."
    ),
    "output_format": (
        "Title: [Catchy Title]\n\n"
        "Story:\n[Paragraph 1 - Introduction]\n\n[Paragraph 2 - Conflict/Climax]\n\n[Paragraph 3 - Resolution]\n\n"
        "Moral / Takeaway: [1 sentence summary of key theme]"
    )
}

POEM_PROMPT = {
    "role": "Poet and Wordsmith",
    "context": "You are crafting an evocative poem that captures the core essence, emotion, and aesthetic of the subject.",
    "constraints": (
        "1. Structure: 3 to 4 stanzas (4 lines per stanza).\n"
        "2. Rhyme Scheme: AABB or ABAB.\n"
        "3. Tone: Reflective, artistic, and evocative.\n"
        "4. Include rich sensory metaphors."
    ),
    "output_format": (
        "Title: [Poem Title]\n\n"
        "[Stanza 1]\n\n"
        "[Stanza 2]\n\n"
        "[Stanza 3]\n\n"
        "[Stanza 4]"
    )
}

SOCIAL_MEDIA_PROMPT = {
    "role": "Expert Social Media Manager & Growth Marketer",
    "context": "You are designing viral, engaging social media posts tailored for LinkedIn, Twitter (X), and Instagram.",
    "constraints": (
        "1. Platforms: Twitter/X (under 280 chars), LinkedIn (professional insights with call-to-action), Instagram (engaging caption with emojis).\n"
        "2. Style: Engaging, concise, with strong hooks and clear calls-to-action (CTA).\n"
        "3. Hashtags: 3 to 5 relevant hashtags per platform post."
    ),
    "output_format": (
        "--- TWITTER / X POST ---\n"
        "[Post text with emojis]\n"
        "Hashtags: #... #...\n\n"
        "--- LINKEDIN POST ---\n"
        "[Headline Hook]\n"
        "[Body Paragraphs]\n"
        "[Call to Action]\n"
        "Hashtags: #... #...\n\n"
        "--- INSTAGRAM CAPTION ---\n"
        "[Caption text with emojis]\n"
        "[CTA]\n"
        "Hashtags: #... #..."
    )
}

# ---------------------------------------------------------------------------
# Module 3: Podcast Planning Prompts
# ---------------------------------------------------------------------------

PODCAST_PROMPT = {
    "role": "Executive Podcast Producer and Content Strategist",
    "context": "You are designing a comprehensive podcast episode plan designed to educate, entertain, and inspire listeners.",
    "constraints": (
        "1. Generate a compelling Title that grabs attention.\n"
        "2. Write a detailed Episode Description (100-150 words) suitable for Spotify/Apple Podcasts.\n"
        "3. Define the Ideal Guest Profile (expertise, background, domain experience).\n"
        "4. Formulate 5 thought-provoking, deep-dive Interview Questions ranging from introductory to vision/future predictions."
    ),
    "output_format": (
        "Episode Title: [Engaging & Click-worthy Title]\n\n"
        "Description:\n[Detailed show notes description]\n\n"
        "Target Guest Profile:\n- Ideal Role: [Role]\n- Expertise: [Areas of Expertise]\n- Why them: [Reason]\n\n"
        "Interview Questions:\n"
        "1. [Icebreaker / Context Question]\n"
        "2. [Core Concept Question]\n"
        "3. [Deep Dive / Challenge Question]\n"
        "4. [Industry Impact / Case Study Question]\n"
        "5. [Future Outlook / Key Advice Question]"
    )
}

# ---------------------------------------------------------------------------
# Module 4: Text Analysis Prompts (Structured JSON Output)
# ---------------------------------------------------------------------------

TEXT_ANALYSIS_PROMPT = {
    "role": "Senior Computational Linguist & NLP Analyst",
    "context": "You are performing precise textual analysis on user-supplied content to evaluate emotional sentiment and extract domain keywords.",
    "constraints": (
        "1. Sentiment Analysis: Score from -1.0 (extremely negative) to +1.0 (extremely positive). Label as Positive, Negative, or Neutral.\n"
        "2. Sentiment Rationale: Provide a 1-2 sentence breakdown of why this sentiment score was assigned based on word choices.\n"
        "3. Keyword Extraction: Extract 5 to 10 most critical keywords or keyphrases along with relevance scores (0.0 to 1.0).\n"
        "4. Strictly return valid JSON adhering strictly to the requested schema."
    ),
    "output_format": (
        "{\n"
        '  "sentiment": {\n'
        '    "score": 0.85,\n'
        '    "label": "Positive",\n'
        '    "explanation": "The text uses optimistic language emphasizing progress, innovation, and success."\n'
        '  },\n'
        '  "keywords": [\n'
        '    {"keyword": "Prompt Engineering", "relevance": 0.95},\n'
        '    {"keyword": "Artificial Intelligence", "relevance": 0.90},\n'
        '    {"keyword": "Content Generation", "relevance": 0.85}\n'
        '  ]\n'
        "}"
    )
}

# ---------------------------------------------------------------------------
# Module 5: Parameter Experimentation Prompts
# ---------------------------------------------------------------------------

EXPERIMENTATION_PROMPT = {
    "role": "Experimental AI Evaluator",
    "context": "You are illustrating how LLM decoding parameters affect output creativity, randomness, and adherence.",
    "constraints": (
        "Generate a creative response to the user's prompt. Pay close attention to how deterministic or imaginative your phrasing is."
    ),
    "output_format": "[Direct creative response]"
}


def build_full_prompt(prompt_config: dict, user_topic_or_text: str) -> str:
    """Builds a complete structured prompt by injecting user topic/text into prompt schema."""
    system_part = SYSTEM_PROMPT_TEMPLATE.format(
        role=prompt_config["role"],
        context=prompt_config["context"],
        constraints=prompt_config["constraints"],
        output_format=prompt_config["output_format"]
    )
    user_part = f"\nUser Input / Topic: {user_topic_or_text}\n\nGenerated Response:"
    return system_part + user_part
