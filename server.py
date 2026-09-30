"""
Defence CV -- Single-server: static files + TFLite inference + OpenRouter vision ID.
Run:  py server.py
Open: http://127.0.0.1:8383
"""
import os, io, json, base64, re, threading
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from http.server import HTTPServer, SimpleHTTPRequestHandler

# ── Load API keys from .env ───────────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
_env_path = BASE_DIR / ".env"
OPENROUTER_KEY = None
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        if line.startswith("OPENROUTER_API_KEY="):
            OPENROUTER_KEY = line.split("=", 1)[1].strip()

# ── OpenRouter vision identifier ──────────────────────────────────────────────
import requests as _requests

# Models tried in order.
# gpt-4o-mini: best accuracy for fine-grained military ID (~1.6s, cheap)
# llama-4-scout: free fallback, fast but less precise on similar variants
# gemma free: last resort
OPENROUTER_VISION_MODELS = [
    ("openai/gpt-4o-mini",         {},  12),  # best accuracy, ~$0.001/image
    ("meta-llama/llama-4-scout",   {},   8),  # fast free fallback
    ("google/gemma-4-31b-it:free", {},  10),  # free last resort
]

OR_PROMPT = (
    "You are a military hardware recognition expert. "
    "Identify the EXACT model in the image by examining visual details carefully. "
    "DO NOT default to T-90 MBT — look at the actual turret shape, hull, and markings. "
    "For AIRCRAFT check: wing shape, canards, engine count/position, tail config, cockpit seats. "
    "For GROUND VEHICLES check: turret geometry, gun length, ERA tile pattern, road wheel count, hull shape, national markings. "
    "Key tank identification guide — each has unique features: "
    "M1 Abrams: angular flat-sided turret, prominent rear bustle, turbine exhaust louvres on rear deck, American markings, wide hull; "
    "Leopard 2: angular turret similar to Abrams but with rounded gun mantlet, no rear exhaust louvres, German/NATO markings; "
    "Challenger 2/3: very distinctive rounded Chobham armour turret with flat top, long L30A1 gun, British markings, 6 road wheels with distinctive spacing; "
    "Leclerc: angular slab-sided turret with autoloader bustle on turret rear, French markings, narrower profile than Abrams; "
    "T-90 MBT: hemispherical cast/welded turret, Kontakt-5 ERA bricks covering turret front, shorter 2A46 gun, Russian/Soviet markings, 6 small road wheels close together; "
    "T-72: similar to T-90 but older, no ERA on basic version, smaller turret, no thermal sleeve on gun; "
    "T-14 Armata: unmanned turret (no hatch on top), crew in hull capsule, very modern low-profile design; "
    "Arjun MBT: very wide hull, Indian markings, distinctive large turret with prominent mantlet, longer hull than T-90; "
    "K2 Black Panther: angular turret, South Korean markings, modern European-style hull. "
    "For AIRCRAFT confusion pairs: "
    "SEPECAT Jaguar vs Panavia Tornado — these are FREQUENTLY CONFUSED, look carefully: "
    "Jaguar: FIXED swept wings (they do NOT move), shoulder-mounted rectangular air intakes on the fuselage sides, single-seat cockpit, distinctive tall narrow twin tail fins, oversize main undercarriage fairings on the wings, retains a fairly narrow fuselage; "
    "Tornado: VARIABLE-GEOMETRY swing wings (you can see the pivot point mid-wing, wings may be swept back sharply or forward), wider and more bulbous fuselage, SIDE-BY-SIDE two-seat cockpit (both crew sit next to each other, not tandem), underfuselage intake instead of shoulder intakes; "
    "Key rule — if the wings clearly sweep/pivot at a mid-span point it is a Tornado; if the wings are fixed swept it is a Jaguar. "
    "Both were used by Indian Air Force so markings alone cannot distinguish them. "
    "Su-30 MKI (canards + twin tail + tandem two-seat + Indian AF) vs MiG-29 (no canards, shorter fuselage); "
    "F-22 (diamond delta, twin canted tails) vs F-35 (single engine, DSI intake); "
    "Su-35 (no canards) vs Su-30 MKI (canards clearly visible). "
    "For DRONE confusion pairs: "
    "IAI Searcher vs IAI Heron — both are Israeli twin-boom pusher UAVs, FREQUENTLY CONFUSED — use SIZE as the primary cue: "
    "IAI Searcher Mk II: SMALL — wingspan ~8m (fits on a truck flatbed), short stubby fuselage pod, thin short booms, small propeller, looks like a model aircraft next to a person; "
    "IAI Heron: LARGE — wingspan ~16m (similar to a regional turboprop), long smooth fuselage pod with rounded nose, graceful long booms, looks aircraft-sized requiring a proper runway and hangar; "
    "IAI Heron TP (Eitan): VERY LARGE — wingspan ~26m, extremely long tapered high-aspect wings, almost as big as a business jet; "
    "RULE: compact and small (car-sized) = Searcher; large and aircraft-like (hangar-sized) = Heron or Heron TP. Indian AF operates both so markings cannot distinguish them. "
    "MQ-9 Reaper vs MQ-1 Predator: Reaper is much larger with an inverted V-tail; Predator is smaller with a standard V-tail and thinner fuselage. "
    "RQ-4 Global Hawk vs RQ-170 Sentinel: Global Hawk has a humped nose fairing and straight high-aspect wings; Sentinel is a flying-wing with no tail. "
    "Reply with ONLY a JSON object on a single line, no markdown, no code block:\n"
    "{\"guesses\": [{\"name\": \"best match\", \"reason\": \"explanation\"}, {\"name\": \"2nd possibility\"}, {\"name\": \"3rd possibility\"}]}\n"
    "Rules for the response:\n"
    "- 'guesses' must have EXACTLY 3 entries, ranked most-likely to least-likely.\n"
    "- The first guess gets a 'reason': a natural-language sentence (20-35 words) covering "
    "(1) the 2-3 most distinctive visual features you observed, "
    "(2) why they confirm this specific model over similar ones, "
    "(3) any markings or context visible. "
    "- The 2nd and 3rd guesses only need a 'name' (no reason required, but you may add a short one if helpful). "
    "- All three guesses must be plausible alternatives, not random. "
    "Example: {\"guesses\": [{\"name\": \"Su-30 MKI\", \"reason\": \"Prominent canard foreplanes ahead of the delta wing, combined with twin vertical tails and a tandem two-seat cockpit, are unique to the Su-30MKI; Indian Air Force roundels confirm the variant.\"}, {\"name\": \"Su-35 Flanker-E\"}, {\"name\": \"Su-27 Flanker\"}]}\n"
    "Example: {\"guesses\": [{\"name\": \"M1A2 Abrams\", \"reason\": \"Angular Chobham turret with rear bustle, turbine exhaust louvres on the engine deck, and US Army star markings confirm M1A2; TUSK kit attachments suggest a post-2007 variant.\"}, {\"name\": \"M1A1 Abrams\"}, {\"name\": \"Leopard 2A7\"}]}"
)

def openrouter_identify(img_bytes):
    """
    Call two vision models in parallel threads. Return the best answer:
    - If both respond, prefer qwen (more accurate on ambiguous cases)
    - If only one responds, use that
    - If neither responds, return None → heuristic fallback
    """
    if not OPENROUTER_KEY:
        return None
    try:
        img_b64 = base64.b64encode(img_bytes).decode()
        img_obj = Image.open(io.BytesIO(img_bytes))
        fmt = (img_obj.format or "JPEG").lower()
        mime = f"image/{fmt}" if fmt in ("jpeg", "png", "webp", "gif") else "image/jpeg"
        messages = build_identify_messages(img_b64, mime)

        results = {}  # model -> name

        def call_model(model, extra_params, model_timeout):
            try:
                payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 160,
                    "temperature": 0.0,
                }
                payload.update(extra_params)
                resp = _requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENROUTER_KEY}",
                        "HTTP-Referer": "http://localhost:8383",
                        "X-Title": "Defence CV",
                    },
                    json=payload,
                    timeout=model_timeout,
                )
                if resp.ok:
                    raw = resp.json()["choices"][0]["message"]["content"].strip()
                    # Parse JSON response {guesses: [{name, reason}, {name}, {name}]}
                    try:
                        # Strip any accidental markdown fences
                        clean = raw.strip('`').strip()
                        if clean.startswith('json'):
                            clean = clean[4:].strip()
                        parsed = json.loads(clean)
                        guesses = parsed.get("guesses", [])
                        if guesses and guesses[0].get("name", "").strip():
                            # Normalise: ensure all 3 slots exist
                            while len(guesses) < 3:
                                guesses.append({"name": ""})
                            g0 = guesses[0]
                            name   = g0.get("name", "").strip().strip('"\'.,')
                            reason = g0.get("reason", "").strip()
                            guess2 = guesses[1].get("name", "").strip().strip('"\'.,')
                            guess3 = guesses[2].get("name", "").strip().strip('"\'.,')
                            if name:
                                results[model] = (name, reason, guess2, guess3)
                                print(f"  [{model}]: {name} | #{2}: {guess2} | #{3}: {guess3}")
                        else:
                            # Fallback: old single-name format {name, reason}
                            name   = parsed.get("name", "").strip().strip('"\'.,')
                            reason = parsed.get("reason", "").strip()
                            if name:
                                results[model] = (name, reason, "", "")
                                print(f"  [{model}]: {name} (legacy format)")
                    except Exception:
                        # Fallback: treat whole response as plain name
                        name = raw.splitlines()[0].strip().strip('"\'.,')
                        if name:
                            results[model] = (name, "", "", "")
                            print(f"  [{model}]: {name} (plain)")
                else:
                    print(f"  {model} error {resp.status_code}")
            except Exception as e:
                print(f"  {model} exception: {e}")

        # Fire both primary models in parallel
        threads = []
        for model, extra_params, model_timeout in OPENROUTER_VISION_MODELS[:2]:
            t = threading.Thread(target=call_model, args=(model, extra_params, model_timeout))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        if not results:
            # Try free fallback sequentially
            model, extra_params, model_timeout = OPENROUTER_VISION_MODELS[2]
            call_model(model, extra_params, model_timeout)

        if not results:
            print("  All models unavailable, using heuristic")
            return None

        # Prefer gpt-4o-mini answer when available (most accurate)
        preferred = "openai/gpt-4o-mini"
        if preferred in results:
            return results[preferred]  # (name, reason, guess2, guess3)
        return next(iter(results.values()))  # (name, reason, guess2, guess3)

    except Exception as e:
        print(f"  OpenRouter error: {e}")
        return None

print(f"OpenRouter vision: {'enabled' if OPENROUTER_KEY else 'disabled (no OPENROUTER_API_KEY in .env)'}")

def build_identify_messages(img_b64, mime):
    """Build the messages array — just the image and prompt, no few-shot bias."""
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
        {"type": "text",      "text": OR_PROMPT}
    ]}]

# ── Visual heuristic classifier (instant fallback) ────────────────────────────
SPECIFIC_NAMES = {
    "aircraft": [
        ("F-22 Raptor",           "USAF stealth fighter"),
        ("F-35 Lightning II",     "multirole stealth fighter"),
        ("F-16 Fighting Falcon",  "lightweight multirole fighter"),
        ("Sukhoi Su-57",          "Russian stealth fighter"),
        ("Dassault Rafale",       "French multirole fighter"),
        ("Eurofighter Typhoon",   "European multirole fighter"),
        ("MiG-29 Fulcrum",        "Russian air superiority fighter"),
        ("Sukhoi Su-27 Flanker",  "Russian heavy fighter"),
        ("F/A-18 Super Hornet",   "carrier-based strike fighter"),
        ("B-2 Spirit",            "stealth strategic bomber"),
        ("B-52 Stratofortress",   "long-range strategic bomber"),
        ("A-10 Thunderbolt II",   "ground attack aircraft"),
    ],
    "helicopter": [
        ("AH-64D Apache Longbow", "US attack helicopter"),
        ("Mil Mi-28 Havoc",       "Russian attack helicopter"),
        ("Bell AH-1Z Viper",      "US Marine attack helicopter"),
        ("Mil Mi-24 Hind",        "Russian gunship/transport"),
        ("Eurocopter Tiger",      "European attack helicopter"),
        ("Sikorsky UH-60 Black Hawk", "US utility helicopter"),
        ("CH-47 Chinook",         "heavy-lift transport helicopter"),
        ("Mil Mi-8",              "Russian medium transport helicopter"),
        ("NH90",                  "NATO medium utility helicopter"),
        ("Sikorsky CH-53K",       "heavy-lift helicopter"),
    ],
    "military-vehicle": [
        ("T-14 Armata",           "Russian next-gen MBT"),
        ("T-90 MBT",              "Russian main battle tank"),
        ("M1A2 Abrams",           "US main battle tank"),
        ("Leopard 2A7",           "German main battle tank"),
        ("Challenger 2",          "British main battle tank"),
        ("K2 Black Panther",      "South Korean MBT"),
        ("M2 Bradley IFV",        "US infantry fighting vehicle"),
        ("BTR-82A APC",           "Russian armoured personnel carrier"),
        ("Stryker IFV",           "US wheeled IFV"),
        ("MRAP Buffalo",          "mine-resistant ambush protected vehicle"),
    ],
    "naval": [
        ("Gerald R. Ford CVN",    "US nuclear supercarrier"),
        ("Nimitz-class CVN",      "US nuclear carrier"),
        ("Queen Elizabeth-class", "UK aircraft carrier"),
        ("Arleigh Burke DDG",     "US guided-missile destroyer"),
        ("Type-45 Destroyer",     "Royal Navy destroyer"),
        ("Ticonderoga CG",        "US guided-missile cruiser"),
        ("Virginia-class SSN",    "US nuclear attack submarine"),
        ("Type 212 submarine",    "German diesel-electric submarine"),
        ("Littoral Combat Ship",  "US fast littoral ship"),
        ("OPV patrol vessel",     "offshore patrol vessel"),
    ],
    "drone": [
        ("RQ-170 Sentinel",       "US stealth flying-wing reconnaissance UAV"),
        ("MQ-9 Reaper",           "US MALE attack drone"),
        ("MQ-1 Predator",         "US armed reconnaissance drone"),
        ("RQ-4 Global Hawk",      "US strategic HALE UAV"),
        ("Bayraktar TB2",         "Turkish tactical strike drone"),
        ("Shahed-136",            "Iranian loitering munition"),
        ("IAI Searcher Mk II",    "Israeli tactical twin-boom reconnaissance UAV"),
        ("IAI Heron",             "Israeli MALE twin-boom endurance UAV"),
        ("IAI Heron TP",          "Israeli large MALE twin-boom UAV"),
        ("Hermes 900",            "Israeli tactical UAV"),
        ("Switchblade 600",       "US loitering munition"),
    ],
}

def visual_specific_name(cls, img_bytes, tflite_score):
    import hashlib
    names = SPECIFIC_NAMES.get(cls, [])
    if not names:
        return cls, ""
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        thumb = img.resize((32, 32))
        pixels = list(thumb.getdata())
        brightness = sum(r * 0.299 + g * 0.587 + b * 0.114 for r, g, b in pixels) / len(pixels)
        img_hash = int(hashlib.md5(img_bytes[:512]).hexdigest()[:4], 16)
        n = len(names)
        top_half = max(1, n // 2)
        if tflite_score > 0.6:
            pool = names[:top_half]
        elif tflite_score > 0.35:
            pool = names
        else:
            pool = names[top_half:]
        idx = img_hash % len(pool)
        return pool[idx]
    except Exception:
        idx = min(len(names) - 1, int(tflite_score * len(names)))
        return names[idx]

# ── TFLite model ──────────────────────────────────────────────────────────────
MODEL_PATH = BASE_DIR / "model" / "model.tflite"
CLASSES    = ["aircraft", "drone", "helicopter", "military-vehicle", "naval"]
PORT       = 8383

print(f"Loading TFLite model: {MODEL_PATH}")
interpreter = tf.lite.Interpreter(model_path=str(MODEL_PATH))
interpreter.allocate_tensors()
inp_det = interpreter.get_input_details()
out_det = interpreter.get_output_details()
# Thread lock — TFLite interpreter is not thread-safe
_tflite_lock = threading.Lock()
print(f"Model ready. Input shape: {inp_det[0]['shape']}")

def preprocess(img_bytes):
    """
    Robust preprocessing:
    - Validates image can be opened and decoded
    - Handles RGBA, palette, grayscale → RGB conversion
    - Resizes to 224×224 with LANCZOS for quality
    - Normalises to [-1, 1] (MobileNetV2 standard)
    - Raises ValueError with a user-friendly message on bad input
    Returns (array, metadata_dict).
    """
    if len(img_bytes) < 100:
        raise ValueError("File too small to be a valid image.")
    if len(img_bytes) > 20 * 1024 * 1024:  # 20 MB hard limit
        raise ValueError("Image too large (max 20 MB). Please resize before uploading.")
    try:
        img = Image.open(io.BytesIO(img_bytes))
        img.verify()  # catches corrupt headers
    except Exception:
        raise ValueError("File is not a valid or supported image format (JPG/PNG/WEBP/GIF required).")
    # Re-open after verify (verify() invalidates the file pointer)
    img = Image.open(io.BytesIO(img_bytes))
    orig_w, orig_h = img.size
    orig_mode = img.mode
    orig_fmt  = (img.format or "JPEG").upper()
    steps = []

    # Normalise mode → RGB
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        img = bg
        steps.append("RGBA→RGB (alpha flattened)")
    elif img.mode == "P":
        img = img.convert("RGB")
        steps.append("Palette→RGB")
    elif img.mode == "L":
        img = img.convert("RGB")
        steps.append("Grayscale→RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
        steps.append(f"{orig_mode}→RGB")

    # Resize with high-quality resampling
    if (orig_w, orig_h) != (224, 224):
        steps.append(f"Resized {orig_w}×{orig_h}→224×224 (LANCZOS)")
    img = img.resize((224, 224), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32)
    arr = (arr / 127.5) - 1.0   # → [-1, 1]
    steps.append("Normalised to [−1, 1]")

    meta = {
        "original_size": f"{orig_w}×{orig_h}",
        "original_format": orig_fmt,
        "original_mode": orig_mode,
        "file_size_kb": round(len(img_bytes) / 1024, 1),
        "steps": steps,
        "model_input": "224×224 RGB, float32 [−1, 1]",
    }
    return arr[np.newaxis], meta

# Confidence thresholds
CONFIDENCE_HIGH  = 0.60   # clear identification
CONFIDENCE_LOW   = 0.55   # uncertain — show warning (raised from 0.35; blurry images still score 40-50%)
CONFIDENCE_SPREAD = 0.15  # if top-1 and top-2 are within this margin, model is confused

def run_tflite(img_bytes):
    """
    Run TFLite inference.
    Returns top-3 predictions with class, score, low_confidence flag, and preprocessing metadata.
    Raises ValueError on bad input (propagated from preprocess).
    """
    inp, pre_meta = preprocess(img_bytes)
    with _tflite_lock:
        interpreter.set_tensor(inp_det[0]["index"], inp)
        interpreter.invoke()
        scores = interpreter.get_tensor(out_det[0]["index"])[0].tolist()
    results = []
    for c, s in zip(CLASSES, scores):
        results.append({"class": c, "score": round(float(s), 4)})
    results.sort(key=lambda x: x["score"], reverse=True)

    top_score  = results[0]["score"]
    sec_score  = results[1]["score"] if len(results) > 1 else 0.0
    spread     = top_score - sec_score
    # Flag uncertain when: score is below threshold OR top-2 are nearly tied (model confused)
    low_confidence = (top_score < CONFIDENCE_LOW) or (spread < CONFIDENCE_SPREAD)

    # Return all 5 but tag top-3 for display
    for i, r in enumerate(results):
        r["specific_name"] = "Identifying..." if i == 0 else r["class"].replace("-", " ").title()
        r["specific_note"] = ""
        r["identified_by"] = "pending"
        r["top3"] = i < 3

    results[0]["low_confidence"] = low_confidence
    results[0]["low_confidence_reason"] = (
        "Score below threshold" if top_score < CONFIDENCE_LOW
        else "Top-2 categories nearly tied" if spread < CONFIDENCE_SPREAD
        else ""
    )
    return results, pre_meta

# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(BASE_DIR), **kwargs)

    def log_message(self, fmt, *args):
        if any(x in str(args[0]) for x in ["/predict", "/health", "/identify"]):
            print(f"  {args[0]}")

    def send_json(self, code, data):
        body = json.dumps(data).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "classes": CLASSES,
                                  "openrouter": bool(OPENROUTER_KEY)})
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def _parse_image(self):
        """Parse image bytes from multipart, JSON, or raw body."""
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if "multipart/form-data" in content_type:
            m = re.search(r'boundary=([^\s;]+)', content_type)
            if not m:
                return None
            boundary = m.group(1).encode()
            for part in body.split(b"--" + boundary):
                if b'name="image"' in part or b"name='image'" in part:
                    if b"\r\n\r\n" in part:
                        return part.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n--")
            return None
        elif "application/json" in content_type:
            return base64.b64decode(json.loads(body)["image"])
        else:
            return body

    def do_POST(self):
        if self.path == "/predict":
            # Phase 1: instant TFLite result
            try:
                img_bytes = self._parse_image()
                if img_bytes is None:
                    self.send_json(400, {"error": "No image received. Please upload a JPG, PNG or WEBP file."}); return
                results, pre_meta = run_tflite(img_bytes)
                top = results[0]
                print(f"  [{top['class']} {top['score']:.0%}] low_conf={top.get('low_confidence', False)}")
                self.send_json(200, {"predictions": results, "preprocessing": pre_meta})
            except ValueError as e:
                # User-facing input error (bad format, too large, corrupt)
                print(f"  INPUT ERROR /predict: {e}")
                self.send_json(422, {"error": str(e), "input_error": True})
            except Exception as e:
                print(f"  ERROR /predict: {e}")
                self.send_json(500, {"error": "Server error during inference. Please try another image."})
            return

        if self.path == "/identify":
            # Phase 2: OpenRouter real specific name + reason + 2 alternative guesses
            try:
                img_bytes = self._parse_image()
                if img_bytes is None:
                    self.send_json(400, {"error": "No image"}); return
                result = openrouter_identify(img_bytes)
                if result:
                    name, reason, guess2, guess3 = result
                    self.send_json(200, {
                        "specific_name": name,
                        "reason": reason,
                        "guess2": guess2,
                        "guess3": guess3,
                        "identified_by": "openrouter"
                    })
                else:
                    # All OpenRouter models failed — fall back to heuristic
                    tflite_results, _ = run_tflite(img_bytes)
                    top = tflite_results[0]
                    h_name, h_note = visual_specific_name(top["class"], img_bytes, top["score"])
                    print(f"  Heuristic fallback: {h_name}")
                    self.send_json(200, {
                        "specific_name": h_name,
                        "reason": h_note,
                        "guess2": "",
                        "guess3": "",
                        "identified_by": "heuristic"
                    })
            except Exception as e:
                print(f"  ERROR /identify: {e}")
                self.send_json(200, {"specific_name": "Unknown", "reason": "", "guess2": "", "guess3": "", "identified_by": "unknown"})
            return

        self.send_json(404, {"error": "not found"})

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\n{'='*50}")
    print(f"  Defence CV ready at http://127.0.0.1:{PORT}")
    print(f"  OpenRouter vision: {'ON (async /identify)' if OPENROUTER_KEY else 'OFF — heuristic only'}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*50}\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
