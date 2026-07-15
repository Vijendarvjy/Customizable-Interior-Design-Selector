# Customizable Interior Design Studio

Streamlit app for a real-estate / interior design project that lets clients customize
**doors, windows, TV units, and woodwork** across every room type:

- Bedroom
- Children's Bedroom
- Hall / Entryway
- Living Room
- Kitchen

For each room, pick the element to design, an overall style (Modern, Scandinavian,
Traditional Indian, Luxury Classic, etc.), a material/wood finish, and a color palette.
The app builds a detailed prompt and generates a photoreal concept image using
Hugging Face's **FLUX.1-schnell** model. All generated designs collect into a
"Full Project View" tab, grouped by room — useful for presenting a complete
apartment/unit concept to a client.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a Hugging Face token with Inference API access:
   https://huggingface.co/settings/tokens

3. Provide the token one of three ways:
   - Paste it into the sidebar at runtime, or
   - Set an environment variable: `export HF_TOKEN=hf_...`, or
   - Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill it in

4. Run locally:
   ```bash
   streamlit run app.py
   ```

## Deploying to Streamlit Cloud

- Push this folder to a GitHub repo (do **not** commit a real `secrets.toml`)
- In Streamlit Cloud → App settings → Secrets, paste:
  ```toml
  HF_TOKEN = "hf_your_real_token_here"
  ```
- Make sure `requirements.txt` (already included here) lists `huggingface_hub` —
  this was the missing dependency that previously caused deployment failures.

## Extending

- Add more rooms/elements by editing the `ROOMS` dict at the top of `app.py`.
- Add more style/material/color options by editing `STYLES`, `WOOD_MATERIALS`,
  `DOOR_STYLES`, `WINDOW_STYLES`, `TV_UNIT_STYLES`, `COLOR_PALETTES`.
- Swap models by changing `MODEL_ID` (any HF Inference-API-compatible
  text-to-image model will work).

## Security note

Never commit real API tokens to GitHub or paste them in chat — treat any token
that's been shared elsewhere as compromised and regenerate it from the
Hugging Face tokens page.
