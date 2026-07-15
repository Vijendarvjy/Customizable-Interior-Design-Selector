"""
Customizable Interior Design Selector
--------------------------------------
Lets a user pick a room type + an element (door / window / TV unit / woodwork),
choose style, material, color, and extra details, then generates a photoreal
concept image using Hugging Face's FLUX.1-schnell model.

Run:
    streamlit run app.py

Requires a Hugging Face token with Inference API access, set as:
    - an environment variable HF_TOKEN, or
    - st.secrets["HF_TOKEN"] in .streamlit/secrets.toml, or
    - pasted into the sidebar at runtime
"""

import os
import io
import time
from datetime import datetime

import streamlit as st
from huggingface_hub import InferenceClient

# --------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Customizable Interior Design Studio",
    page_icon="🛋️",
    layout="wide",
)

MODEL_ID = "black-forest-labs/FLUX.1-schnell"

# --------------------------------------------------------------------------
# Domain data: rooms, elements, styles, materials
# --------------------------------------------------------------------------
ROOMS = {
    "Bedroom": {
        "icon": "🛏️",
        "elements": ["Door", "Window", "TV Unit / Wall", "Woodwork (Wardrobe)"],
    },
    "Children's Bedroom": {
        "icon": "🧸",
        "elements": ["Door", "Window", "TV Unit / Wall", "Woodwork (Wardrobe/Study Unit)"],
    },
    "Hall / Entryway": {
        "icon": "🚪",
        "elements": ["Door", "Window", "Woodwork (Shoe Rack/Console)"],
    },
    "Living Room": {
        "icon": "🛋️",
        "elements": ["Door", "Window", "TV Unit / Wall", "Woodwork (Cabinet/Paneling)"],
    },
    "Kitchen": {
        "icon": "🍳",
        "elements": ["Door", "Window", "Woodwork (Cabinets/Island)"],
    },
}

STYLES = [
    "Modern Minimalist",
    "Contemporary",
    "Scandinavian",
    "Industrial",
    "Traditional Indian",
    "Luxury Classic",
    "Mid-Century Modern",
    "Rustic Farmhouse",
]

WOOD_MATERIALS = [
    "Teak Wood", "Walnut Veneer", "Oak", "Laminate Finish",
    "MDF with PU Paint", "Engineered Wood", "Rosewood", "Ply with Laminate",
]

DOOR_STYLES = [
    "Flush Door", "Panel Door", "Glass Sliding Door", "French Door",
    "Barn Sliding Door", "Louvered Door", "Frosted Glass Door",
]

WINDOW_STYLES = [
    "Casement Window", "Sliding Window", "Bay Window",
    "Fixed Glass Window", "French Window", "Louvered Window",
]

TV_UNIT_STYLES = [
    "Floating Wall-Mounted Unit", "Wooden Panel Wall with Backlit LED",
    "TV Console with Storage", "Entertainment Unit with Fluted Panels",
    "Media Wall with Fireplace Niche",
]

COLOR_PALETTES = [
    "Warm Wood Tones with White", "Charcoal Grey and Walnut",
    "Pastel Blue and Beige", "Black and Gold Accents",
    "Earthy Terracotta and Cream", "Monochrome Grey",
    "Sage Green and Natural Wood", "Navy Blue and Brass",
]

ELEMENT_FOCUS_MAP = {
    "Door": DOOR_STYLES,
    "Window": WINDOW_STYLES,
    "TV Unit / Wall": TV_UNIT_STYLES,
}


def get_element_style_options(element_name: str):
    """Return the relevant style-option list for a given element (falls back to wood finishes)."""
    for key, options in ELEMENT_FOCUS_MAP.items():
        if key in element_name:
            return options
    return WOOD_MATERIALS  # woodwork / cabinet elements


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "gallery" not in st.session_state:
    st.session_state.gallery = []  # list of dicts: {room, element, prompt, image_bytes, ts}

def _load_default_token() -> str:
    """Look for a token in secrets.toml first, then fall back to an env var."""
    try:
        if hasattr(st, "secrets") and "HF_TOKEN" in st.secrets:
            return st.secrets["HF_TOKEN"]
    except Exception:
        pass  # no secrets.toml present — that's fine, just fall through
    return os.environ.get("HF_TOKEN", "")


if "hf_token" not in st.session_state:
    st.session_state.hf_token = _load_default_token()


# --------------------------------------------------------------------------
# Sidebar — configuration
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Setup")

    token_source = "secrets.toml / env" if st.session_state.hf_token else "not set"
    token_input = st.text_input(
        "Hugging Face API Token",
        value=st.session_state.hf_token,
        type="password",
        help=(
            f"Currently loaded from: {token_source}. "
            "Needs Inference API access — get one at huggingface.co/settings/tokens"
        ),
    )
    if token_input:
        st.session_state.hf_token = token_input

    st.caption(f"Model: `{MODEL_ID}`")

    st.divider()
    st.header("🖼️ Project Gallery")
    if st.session_state.gallery:
        for i, item in enumerate(reversed(st.session_state.gallery)):
            with st.expander(f"{item['room']} · {item['element']}"):
                st.image(item["image_bytes"], use_container_width=True)
        if st.button("Clear Gallery", use_container_width=True):
            st.session_state.gallery = []
            st.rerun()
    else:
        st.caption("No designs generated yet.")


# --------------------------------------------------------------------------
# Prompt builder
# --------------------------------------------------------------------------
def build_prompt(room, element, style, focus_style, material, colors, details):
    prompt = (
        f"A professional interior design photograph of a {style.lower()} {room.lower()}, "
        f"featuring a {focus_style.lower()} for the {element.lower()}, "
        f"crafted from {material.lower()}, "
        f"color palette of {colors.lower()}. "
        f"Photorealistic, architectural digest style, soft natural lighting, "
        f"high detail, 4k, wide angle shot, no people, no text, no watermark."
    )
    if details:
        prompt += f" Additional details: {details}."
    return prompt


# --------------------------------------------------------------------------
# Image generation
# --------------------------------------------------------------------------
def generate_image(prompt: str):
    if not st.session_state.hf_token:
        st.error("Please enter your Hugging Face API token in the sidebar first.")
        return None
    try:
        client = InferenceClient(model=MODEL_ID, token=st.session_state.hf_token)
        image = client.text_to_image(prompt)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        st.error(f"Generation failed: {e}")
        return None


# --------------------------------------------------------------------------
# Main layout
# --------------------------------------------------------------------------
st.title("🛋️ Customizable Interior Design Studio")
st.write(
    "Design doors, windows, TV units, and woodwork for every room in your project — "
    "generate photoreal concepts instantly with AI."
)

tab_design, tab_project = st.tabs(["🎨 Design a Single Element", "🏠 Full Project View"])

# ---------------- Tab 1: Single element designer ----------------
with tab_design:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Choose Room & Element")
        room = st.selectbox(
            "Room Type",
            list(ROOMS.keys()),
            format_func=lambda r: f"{ROOMS[r]['icon']} {r}",
        )
        element = st.selectbox("Element to Design", ROOMS[room]["elements"])

        st.subheader("2. Choose Style")
        style = st.selectbox("Overall Design Style", STYLES)

        focus_options = get_element_style_options(element)
        focus_label = "Door/Window/Unit Style" if any(
            k in element for k in ["Door", "Window", "TV"]
        ) else "Wood Finish"
        focus_style = st.selectbox(focus_label, focus_options)

        st.subheader("3. Material & Color")
        material = st.selectbox("Primary Material", WOOD_MATERIALS)
        colors = st.selectbox("Color Palette", COLOR_PALETTES)

        details = st.text_area(
            "Extra details (optional)",
            placeholder="e.g. brass handles, backlit panel, frosted glass insert, curved edges...",
        )

        generate_btn = st.button("✨ Generate Design", type="primary", use_container_width=True)

    with col2:
        st.subheader("Preview")
        prompt_preview = build_prompt(room, element, style, focus_style, material, colors, details)
        with st.expander("Generated prompt (editable)", expanded=False):
            edited_prompt = st.text_area("Prompt sent to model", value=prompt_preview, height=140)

        if generate_btn:
            final_prompt = edited_prompt if edited_prompt else prompt_preview
            with st.spinner("Generating your design... this can take 10-30 seconds"):
                image_bytes = generate_image(final_prompt)
            if image_bytes:
                st.image(image_bytes, use_container_width=True, caption=f"{room} · {element} · {style}")
                st.download_button(
                    "⬇️ Download Image",
                    data=image_bytes,
                    file_name=f"{room}_{element}_{int(time.time())}.png".replace(" ", "_").replace("/", "-"),
                    mime="image/png",
                    use_container_width=True,
                )
                st.session_state.gallery.append({
                    "room": room,
                    "element": element,
                    "prompt": final_prompt,
                    "image_bytes": image_bytes,
                    "ts": datetime.now().isoformat(),
                })

# ---------------- Tab 2: Full project view ----------------
with tab_project:
    st.subheader("All Generated Designs for This Project")
    if not st.session_state.gallery:
        st.info("Generate designs in the first tab — they'll be collected here, grouped by room.")
    else:
        rooms_present = sorted(set(item["room"] for item in st.session_state.gallery))
        for r in rooms_present:
            st.markdown(f"### {ROOMS.get(r, {}).get('icon', '')} {r}")
            items = [i for i in st.session_state.gallery if i["room"] == r]
            cols = st.columns(min(len(items), 4) or 1)
            for idx, item in enumerate(items):
                with cols[idx % len(cols)]:
                    st.image(item["image_bytes"], caption=item["element"], use_container_width=True)
            st.divider()

st.caption(
    "Powered by Hugging Face FLUX.1-schnell · Built with Streamlit · "
    "Tip: keep prompts specific (material + style + color) for the most consistent results."
)
