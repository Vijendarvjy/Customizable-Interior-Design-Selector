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
import re
import time
from datetime import datetime

import streamlit as st
import pandas as pd
from huggingface_hub import InferenceClient

try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

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
# Force a clean white theme (independent of .streamlit/config.toml, so it
# stays white even if a viewer's browser/OS prefers dark mode)
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp {
            background-color: #FFFFFF;
        }
        [data-testid="stSidebar"] {
            background-color: #F7F5F2;
        }
        [data-testid="stHeader"] {
            background-color: #FFFFFF;
        }
        h1, h2, h3, h4, p, span, label, .stMarkdown {
            color: #262626;
        }
        img {
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Domain data: rooms, elements, styles, materials
# --------------------------------------------------------------------------
# SAMPLE_ROOMS: a worked example (from a previously uploaded 3BHK plan) shown
# until the user uploads their own floor plan and confirms rooms below.
SAMPLE_ROOMS = {
    "C. Bedroom (11'-6\" x 11'-10.5\")": {
        "icon": "🛏️",
        "dimension": "11 ft 6 in x 11 ft 10.5 in",
        "door_ref": "D3", "window_ref": "W2",
        "elements": ["Door", "Window", "TV Unit / Wall", "Woodwork (Wardrobe)"],
    },
    "G. Bedroom (11'-6\" x 11'-7.5\")": {
        "icon": "🛏️",
        "dimension": "11 ft 6 in x 11 ft 7.5 in",
        "door_ref": "D3", "window_ref": "W4",
        "elements": ["Door", "Window", "TV Unit / Wall", "Woodwork (Wardrobe)"],
    },
    "M. Bedroom (14'-0\" x 10'-0\")": {
        "icon": "🛏️",
        "dimension": "14 ft x 10 ft",
        "door_ref": "D1", "window_ref": "W1 / W4",
        "elements": ["Door", "Window", "TV Unit / Wall", "Woodwork (Wardrobe)"],
    },
    "Kitchen (9'-6\" x 8'-1.5\")": {
        "icon": "🍳",
        "dimension": "9 ft 6 in x 8 ft 1.5 in",
        "door_ref": "D1", "window_ref": "KW1",
        "elements": ["Door", "Window", "Woodwork (Cabinets/Island)"],
    },
    "Utility (4'-6\" x 8'-1.5\")": {
        "icon": "🧺",
        "dimension": "4 ft 6 in x 8 ft 1.5 in",
        "door_ref": "D2", "window_ref": "—",
        "elements": ["Door", "Woodwork (Storage Unit)"],
    },
    "Living/Dining (12'-1.5\" x 20'-9\")": {
        "icon": "🛋️",
        "dimension": "12 ft 1.5 in x 20 ft 9 in",
        "door_ref": "M.D-1 (main)", "window_ref": "3'-6\" wide balcony opening",
        "elements": ["Door", "Window", "TV Unit / Wall", "Woodwork (Cabinet/Paneling)"],
    },
    "Toilet - C.Bedroom (4'-8\" x 8'-3\")": {
        "icon": "🚿",
        "dimension": "4 ft 8 in x 8 ft 3 in",
        "door_ref": "D3", "window_ref": "—",
        "elements": ["Door", "Woodwork (Vanity Unit)"],
    },
    "Toilet - G.Bedroom (4'-9\" x 8'-0\")": {
        "icon": "🚿",
        "dimension": "4 ft 9 in x 8 ft 0 in",
        "door_ref": "D3", "window_ref": "—",
        "elements": ["Door", "Woodwork (Vanity Unit)"],
    },
    "Toilet - M.Bedroom (8'-8\" x 5'-0\")": {
        "icon": "🚿",
        "dimension": "8 ft 8 in x 5 ft 0 in",
        "door_ref": "D2", "window_ref": "—",
        "elements": ["Door", "Woodwork (Vanity Unit)"],
    },
}

ROOM_KEYWORD_DEFAULTS = [
    # (keywords to match in OCR/room text, icon, default element list)
    (["toilet", "wash", "bath"], "🚿", ["Door", "Woodwork (Vanity Unit)"]),
    (["kitchen"], "🍳", ["Door", "Window", "Woodwork (Cabinets/Island)"]),
    (["utility"], "🧺", ["Door", "Woodwork (Storage Unit)"]),
    (["living", "dining", "hall"], "🛋️", ["Door", "Window", "TV Unit / Wall", "Woodwork (Cabinet/Paneling)"]),
    (["bed"], "🛏️", ["Door", "Window", "TV Unit / Wall", "Woodwork (Wardrobe)"]),
    (["balcony"], "🌿", ["Door", "Window"]),
]


def guess_room_defaults(label_text: str):
    """Given a room label (e.g. 'M.Bed room'), guess an icon + sensible element list."""
    lower = (label_text or "").lower()
    for keywords, icon, elements in ROOM_KEYWORD_DEFAULTS:
        if any(k in lower for k in keywords):
            return icon, elements
    return "🏠", ["Door", "Window", "Woodwork (Cabinet/Paneling)"]


# Loose pattern: matches noisy OCR dimension-like tokens, e.g. 11'-6"x11'-10½",
# 9'-6°x8"-1'/4", 14'-O°x10-0". Intentionally permissive — the user reviews
# and corrects the extracted text in an editable table afterward.
DIMENSION_PATTERN = re.compile(
    r"[\dOo][\w'\"°%½¼¾\-/]{2,}\s*[xX×]\s*[\dOo][\w'\"°%½¼¾\-/]{1,}"
)


def extract_candidate_dimensions(image):
    """
    OCR the uploaded floor plan and return a list of candidate rows:
    {room_guess, dimension_raw, line_text}. Upscaling + a sparse-text page
    segmentation mode noticeably improves recognition on small, decorative
    floor-plan labels.
    """
    if not OCR_AVAILABLE:
        return []

    scale = 3
    big = image.convert("RGB").resize(
        (image.width * scale, image.height * scale), Image.LANCZOS
    )
    try:
        raw_text = pytesseract.image_to_string(big, config="--psm 11")
    except pytesseract.TesseractNotFoundError:
        # Binary missing at runtime (e.g. packages.txt not yet applied on this
        # deploy) — degrade to manual entry instead of crashing the app.
        st.session_state["ocr_runtime_unavailable"] = True
        return []
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]

    all_keywords = [kw for grp in ROOM_KEYWORD_DEFAULTS for kw in grp[0]]
    candidates = []
    for i, line in enumerate(lines):
        match = DIMENSION_PATTERN.search(line)
        if not match:
            continue
        room_guess = line.replace(match.group(0), "").strip(" -:")
        if not room_guess:
            for prev in reversed(lines[max(0, i - 2): i]):
                if any(k in prev.lower() for k in all_keywords):
                    room_guess = prev
                    break
        candidates.append({
            "room_guess": room_guess or "",
            "dimension_raw": match.group(0),
            "line_text": line,
        })
    return candidates


ROOMS = SAMPLE_ROOMS

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
if "custom_rooms" not in st.session_state:
    st.session_state.custom_rooms = None  # populated once user confirms rooms from an uploaded plan

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
def build_prompt(room, element, style, focus_style, material, colors, details, dimension=None, door_ref=None, window_ref=None):
    prompt = (
        f"A professional interior design photograph of a {style.lower()} room "
        f"measuring exactly {dimension}, " if dimension else
        f"A professional interior design photograph of a {style.lower()} room, "
    )
    prompt += (
        f"featuring a {focus_style.lower()} for the {element.lower()}, "
        f"crafted from {material.lower()}, "
        f"color palette of {colors.lower()}. "
        f"Furniture and fixtures proportioned realistically to fit the stated room size. "
        f"Photorealistic, architectural digest style, soft natural lighting, "
        f"high detail, 4k, wide angle shot, no people, no text, no watermark."
    )
    if door_ref and door_ref != "—":
        prompt += f" Door position corresponds to marker {door_ref} on the floor plan."
    if window_ref and window_ref != "—":
        prompt += f" Window/opening corresponds to marker {window_ref} on the floor plan."
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

tab_plan, tab_design, tab_project = st.tabs(
    ["📤 Upload Floor Plan", "🎨 Design a Single Element", "🏠 Full Project View"]
)

# ---------------- Tab 0: Upload plan & confirm room dimensions ----------------
with tab_plan:
    st.subheader("1. Upload your floor plan")
    st.write(
        "Upload a floor plan image and this will scan it for room labels and dimensions "
        "(e.g. \"11'-6\\\" x 11'-10½\\\"\"). OCR on decorative/rotated floor-plan text is "
        "imperfect, so review and correct the table below before generating designs — "
        "it's pre-filled with best-effort guesses, not final answers."
    )

    if not OCR_AVAILABLE:
        st.warning(
            "OCR isn't available in this environment (missing `pytesseract` / `tesseract-ocr`). "
            "You can still upload a plan for reference and fill in the table manually below."
        )
    elif st.session_state.get("ocr_runtime_unavailable"):
        st.warning(
            "The `tesseract-ocr` binary isn't installed on this deployment yet, so automatic "
            "text scanning is unavailable right now — you can still fill in the table manually. "
            "If you're on Streamlit Cloud: make sure `packages.txt` (containing `tesseract-ocr`) "
            "sits in the same root folder as `app.py` and `requirements.txt`, then go to "
            "**Manage app → Reboot app** — apt packages are only installed on a fresh boot, "
            "not on every push."
        )

    uploaded_plan = st.file_uploader("Floor plan image", type=["png", "jpg", "jpeg"])

    if uploaded_plan is not None:
        plan_image = Image.open(uploaded_plan) if OCR_AVAILABLE else None
        col_img, col_info = st.columns([1, 1])
        with col_img:
            st.image(uploaded_plan, caption="Uploaded floor plan", use_container_width=True)

        if "plan_candidates" not in st.session_state or st.session_state.get("plan_file_name") != uploaded_plan.name:
            st.session_state.plan_file_name = uploaded_plan.name
            if OCR_AVAILABLE:
                with st.spinner("Scanning plan for room labels and dimensions..."):
                    st.session_state.plan_candidates = extract_candidate_dimensions(plan_image)
            else:
                st.session_state.plan_candidates = []

        candidates = st.session_state.plan_candidates
        with col_info:
            if candidates:
                st.success(f"Found {len(candidates)} possible dimension label(s). Review below.")
            elif OCR_AVAILABLE:
                st.info("No dimension-like text detected automatically — add rows manually below.")

        st.subheader("2. Confirm rooms & dimensions")
        if candidates:
            default_rows = [
                {
                    "Room Name": c["room_guess"] or f"Room {i+1}",
                    "Dimension": c["dimension_raw"],
                    "Door Ref": "",
                    "Window Ref": "",
                    "Elements (comma-separated)": ", ".join(guess_room_defaults(c["room_guess"])[1]),
                }
                for i, c in enumerate(candidates)
            ]
        else:
            default_rows = [
                {"Room Name": "", "Dimension": "", "Door Ref": "", "Window Ref": "",
                 "Elements (comma-separated)": "Door, Window, TV Unit / Wall, Woodwork (Wardrobe)"}
            ]

        rooms_df = st.data_editor(
            pd.DataFrame(default_rows),
            num_rows="dynamic",
            use_container_width=True,
            key="rooms_editor",
        )

        if st.button("✅ Build Design Rooms from This Table", type="primary"):
            new_rooms = {}
            for _, row in rooms_df.iterrows():
                name = str(row.get("Room Name", "")).strip()
                dimension = str(row.get("Dimension", "")).strip()
                if not name or not dimension:
                    continue
                icon, default_elements = guess_room_defaults(name)
                elements_raw = str(row.get("Elements (comma-separated)", "")).strip()
                elements = [e.strip() for e in elements_raw.split(",") if e.strip()] or default_elements
                key = f"{name} ({dimension})"
                new_rooms[key] = {
                    "icon": icon,
                    "dimension": dimension,
                    "door_ref": str(row.get("Door Ref", "")).strip() or "—",
                    "window_ref": str(row.get("Window Ref", "")).strip() or "—",
                    "elements": elements,
                }
            if new_rooms:
                st.session_state.custom_rooms = new_rooms
                st.success(f"{len(new_rooms)} room(s) ready — switch to the Design tab to generate.")
            else:
                st.error("No valid rows found — each room needs at least a Room Name and Dimension.")

    if st.session_state.custom_rooms:
        st.divider()
        st.caption(
            f"✅ Currently using **{len(st.session_state.custom_rooms)} room(s)** from your uploaded plan "
            "in the Design tab. Upload a new plan and rebuild the table above to replace them."
        )
        if st.button("Reset to sample plan"):
            st.session_state.custom_rooms = None
            st.rerun()

active_rooms = st.session_state.custom_rooms or SAMPLE_ROOMS
if not st.session_state.custom_rooms:
    st.sidebar.info("Using the sample plan. Upload your own in the '📤 Upload Floor Plan' tab.")

# ---------------- Tab 1: Single element designer ----------------
with tab_design:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Choose Room & Element")
        room = st.selectbox(
            "Room Type",
            list(active_rooms.keys()),
            format_func=lambda r: f"{active_rooms[r]['icon']} {r}",
        )
        room_info = active_rooms[room]
        st.caption(
            f"📐 Dimensions: **{room_info['dimension']}** · "
            f"Door ref: **{room_info['door_ref']}** · "
            f"Window ref: **{room_info['window_ref']}**"
        )
        element = st.selectbox("Element to Design", room_info["elements"])

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
        prompt_preview = build_prompt(
            room, element, style, focus_style, material, colors, details,
            dimension=room_info["dimension"],
            door_ref=room_info["door_ref"],
            window_ref=room_info["window_ref"],
        )
        with st.expander("Generated prompt (editable)", expanded=False):
            edited_prompt = st.text_area("Prompt sent to model", value=prompt_preview, height=140)

        if generate_btn:
            final_prompt = edited_prompt if edited_prompt else prompt_preview
            with st.spinner("Generating your design... this can take 10-30 seconds"):
                image_bytes = generate_image(final_prompt)
            if image_bytes:
                st.session_state.last_image = image_bytes
                st.session_state.last_caption = f"{room} · {element} · {style}"
                st.session_state.last_filename = (
                    f"{room}_{element}_{int(time.time())}.png".replace(" ", "_").replace("/", "-")
                )
                st.session_state.gallery.append({
                    "room": room,
                    "element": element,
                    "prompt": final_prompt,
                    "image_bytes": image_bytes,
                    "ts": datetime.now().isoformat(),
                })

        if st.session_state.get("last_image"):
            st.markdown("#### 🖼️ Your Generated Design")
            st.image(
                st.session_state.last_image,
                use_container_width=True,
                caption=st.session_state.last_caption,
            )
            st.download_button(
                "⬇️ Download Image",
                data=st.session_state.last_image,
                file_name=st.session_state.last_filename,
                mime="image/png",
                use_container_width=True,
            )

# ---------------- Tab 2: Full project view ----------------
with tab_project:
    st.subheader("All Generated Designs for This Project")
    if not st.session_state.gallery:
        st.info("Generate designs in the first tab — they'll be collected here, grouped by room.")
    else:
        rooms_present = sorted(set(item["room"] for item in st.session_state.gallery))
        for r in rooms_present:
            st.markdown(f"### {active_rooms.get(r, {}).get('icon', '')} {r}")
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
