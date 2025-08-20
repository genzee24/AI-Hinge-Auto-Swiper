import os, json, time, subprocess, argparse, tempfile, re, io
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageOps
import pytesseract
import cv2  # type: ignore
from openai import OpenAI
from dotenv import load_dotenv

# ---------- ADB helpers ----------
def sh(cmd: List[str], capture=True, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=capture, text=True, check=check)


import subprocess, unicodedata, string

SAFE_CHARS = set(string.ascii_letters + string.digits + "@#%+=-_/.,:;!?()[]{}'\"")

def _ascii_sanitize(s: str) -> str:
    repl = {
        '—':'-','–':'-','…':'...','’':"'", '‘':"'", '“':'"', '”':'"',
        '\u00A0':' ','\u2009':' ','\u2007':' ','\u200b':' ','\u202f':' ',
        ';':',',            # <<< NEW: avoid ADB shell metachar
        '&':' and ',        # optional safety
        '|':' / ',          # optional safety
    }
    s = ''.join(repl.get(c, c) for c in s)
    import unicodedata
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = ''.join(c if 32 <= ord(c) < 127 else ' ' for c in s)
    s = ' '.join(s.split())
    return s

def _adb_input_text(device: str | None, text: str) -> bool:
    """Try single-shot ADB input text (spaces as %s). Return True on success."""
    safe = text.replace(' ', '%s')
    try:
        adb(device, "shell", "input", "text", safe)
        return True
    except subprocess.CalledProcessError:
        return False

def _clipboard_paste(device: str | None, text: str) -> bool:
    """Android 10+ supports cmd clipboard; paste with KEYCODE_PASTE (279)."""
    try:
        adb(device, "shell", "cmd", "clipboard", "set", text)
        time.sleep(0.2)
        adb(device, "shell", "input", "keyevent", "279")  # KEYCODE_PASTE
        return True
    except subprocess.CalledProcessError:
        return False

_SHELL_META = set('&|;()<>$`\\\"\'*!?[]{}#~')

def _type_char_by_char(device: str | None, text: str):
    for ch in text:
        if ch == ' ':
            adb(device, "shell", "input", "keyevent", "62")  # SPACE
        else:
            send = ch
            if ch in _SHELL_META:
                # Try to escape for adb shell; if that still fails, fall back to '-'
                try:
                    adb(device, "shell", "input", "text", "\\" + ch)
                except subprocess.CalledProcessError:
                    adb(device, "shell", "input", "text", "-")
                time.sleep(0.02)
                continue
            try:
                adb(device, "shell", "input", "text", send)
            except subprocess.CalledProcessError:
                adb(device, "shell", "input", "text", "-")
            time.sleep(0.01)

def type_text(device: str | None, text: str):
    # 1) sanitize to ASCII to avoid adb crashes
    s = _ascii_sanitize(text)

    # 2) try single-shot
    if _adb_input_text(device, s):
        return

    # 3) try clipboard paste (better fidelity/emoji still ok if phone supports it)
    if _clipboard_paste(device, s):
        return

    # 4) degrade gracefully: char-by-char
    _type_char_by_char(device, s)


def adb(device: Optional[str], *args: str, capture=True, check=True):
    base = ["adb"]
    if device:
        base += ["-s", device]
    return sh(base + list(args), capture=capture, check=check)

def get_resolution(device: Optional[str]) -> Tuple[int, int]:
    out = adb(device, "shell", "wm", "size").stdout.strip()
    m = re.search(r'Physical size:\s*(\d+)x(\d+)', out)
    if not m:
        # Fallback via dumpsys
        out = adb(device, "shell", "dumpsys", "display").stdout
        m = re.search(r'cur=\s*(\d+)x(\d+)', out)
    if not m:
        raise RuntimeError("Could not determine screen size")
    return int(m.group(1)), int(m.group(2))


def to_px(xy: Tuple[float,float], res: Tuple[int,int]) -> Tuple[int,int]:
    x, y = xy
    W, H = res
    if 0 <= x <= 1 and 0 <= y <= 1:
        return int(round(x*W)), int(round(y*H))
    return int(x), int(y)

def screencap(device: Optional[str], dest_path: Path):
    # Use exec-out for speed
    with open(dest_path, "wb") as f:
        p = subprocess.Popen(["adb"] + (["-s", device] if device else []) + ["exec-out","screencap","-p"], stdout=f)
        p.wait(10)

def swipe(device: Optional[str], xy1: Tuple[int,int], xy2: Tuple[int,int], duration_ms: int=400):
    adb(device, "shell", "input", "swipe", str(xy1[0]), str(xy1[1]), str(xy2[0]), str(xy2[1]), str(duration_ms))

def tap(device: Optional[str], xy: Tuple[int,int]):
    adb(device, "shell", "input", "tap", str(xy[0]), str(xy[1]))

def input_text(device: Optional[str], text: str):
    # Simple escape for ADB input; spaces -> %s, strip unsupported chars
    safe = text.replace(" ", "%s")
    adb(device, "shell", "input", "text", safe)

# ---------- Image diff / overlap ----------
def img_md5(im: Image.Image) -> str:
    import hashlib
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return hashlib.md5(buf.getvalue()).hexdigest()

def stationary(prev: Image.Image, curr: Image.Image, thresh: float=0.003) -> bool:
    a = np.asarray(prev.convert("L"), dtype=np.int16)
    b = np.asarray(curr.convert("L"), dtype=np.int16)
    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    a, b = a[:h,:w], b[:h,:w]
    diff = np.mean(np.abs(a-b))/255.0
    return diff < thresh

def crop_overlap(prev: Image.Image, curr: Image.Image, band_h: int=160) -> Image.Image:
    """
    Find how much of prev's bottom band appears at top of curr; crop it out to avoid duplicates.
    """
    prev_band = np.asarray(prev.convert("RGB"))[-band_h:, :, :]
    curr_np   = np.asarray(curr.convert("RGB"))

    res = cv2.matchTemplate(curr_np, prev_band, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    # If very high match starting at y=0, we will crop exactly that overlap
    if max_val > 0.88 and max_loc[1] <= 4:
        ov = prev_band.shape[0]
        return Image.fromarray(curr_np[ov:, :, :])
    # Otherwise do a conservative small crop to reduce chance of dup lines
    return curr.crop((0, 12, curr.width, curr.height))

# ---------- OCR ----------
def tesser_text(img: Image.Image) -> str:
    # light cleanup
    gray = ImageOps.grayscale(img)
    arr = np.array(gray)
    arr = cv2.bilateralFilter(arr, 7, 30, 30)
    thr = cv2.adaptiveThreshold(arr,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,35,11)
    proc = Image.fromarray(thr)
    cfg = "--oem 3 --psm 6"
    return pytesseract.image_to_string(proc, lang="eng", config=cfg)

# ---------- OpenAI ----------
def call_openai(profile_bullets: List[str], user_prompt: str) -> str:
    load_dotenv()
    model = os.getenv("MODEL","gpt-4o-mini")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    system = (
        "You craft respectful, specific dating openers.\n"
        "Constraints: 1-2 short lines, reference concrete details; no generic compliments, no cringey pickup lines. And be funny witty.\n"
    )
    system += " Use ONLY plain ASCII characters (no emojis, no smart quotes/dashes)."
    bullets = "\n".join(f"- {b}" for b in profile_bullets[:25])
    full_user = (
        f"Profile details (OCR; noisy text possible):\n{bullets}\n\n"
        f"Task:  Write like this \n"
        f"Apparently Hinge is showing us most compatible. 'Enter some intresting thing here based on her profile here', shall we test it? \n"    
        f"Return ONLY the opener text."
    )
    resp = client.responses.create(
        model=model,
        instructions=system,
        input=full_user,
    )
    return resp.output_text.strip()

# ---------- Main scroll->OCR->compose->act ----------
@dataclass
class Config:
    device: Optional[str]
    scroll_start: Tuple[float,float]
    scroll_end: Tuple[float,float]
    duration_ms: int
    max_scrolls: int
    stop_when_stationary: int
    button_sequence: List[dict]

def dedupe_lines(text: str) -> List[str]:
    seen = set()
    out = []
    for raw in text.splitlines():
        line = re.sub(r'\s+', ' ', raw.strip())
        if len(line) < 2: 
            continue
        key = line.lower()
        if key not in seen:
            seen.add(key)
            out.append(line)
    return out

def run_flow(pkg: str, cfg: Config, out_txt: Path, prompt: str) -> str:
    """
    Pixel-based: scroll, screenshot, OCR, compose opener, then tap/type using exact pixels.
    STOP rule: if the current full screenshot is EXACTLY the same as the previous, break immediately.
    """
    import tempfile, time
    from PIL import Image

    # Bring app to foreground (optional)
    if pkg:
        adb(cfg.device, "shell", "monkey", "-p", pkg, "-c",
            "android.intent.category.LAUNCHER", "1")
        time.sleep(0.6)

    # Scroll endpoints (pixels OK; to_px also accepts raw pixels)
    W, H = get_resolution(cfg.device)
    s_px = to_px(cfg.scroll_start, (W, H))
    e_px = to_px(cfg.scroll_end, (W, H))

    tmp_dir = Path(tempfile.mkdtemp())
    ocr_accum = []
    prev_full = None
    prev_arr  = None  # exact pixel array for equality check

    def screenshot_full() -> Image.Image:
        pth = tmp_dir / f"scr_{int(time.time()*1000)}.png"
        screencap(cfg.device, pth)
        return Image.open(pth).convert("RGB")

    for i in range(10):
        im_full = screenshot_full()
        curr_arr = np.asarray(im_full)

        # --- exact stop: identical to previous full screenshot ---
        if prev_arr is not None and curr_arr.shape == prev_arr.shape and np.array_equal(curr_arr, prev_arr):
            print("[stop] Exact same screenshot as previous; breaking.")
            break

        # OCR (crop overlap against previous to avoid duplicates)
        if prev_full is not None:
            im_cropped = crop_overlap(prev_full, im_full, band_h=160)
        else:
            im_cropped = im_full
        ocr_accum.append(tesser_text(im_cropped))

        # Next scroll attempt (pixels)
        swipe(cfg.device, s_px, e_px, cfg.duration_ms)
        time.sleep(1)

        prev_full = im_full
        prev_arr  = curr_arr

    # Consolidate OCR + ask OpenAI
    all_text = "\n".join(ocr_accum)
    lines = dedupe_lines(all_text)
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    opener = call_openai(lines, prompt)
    print("\n=== GPT Opener ===\n" + opener + "\n==================\n")

    # Execute your button sequence (PIXELS)
    for step in cfg.button_sequence:
        # inside the button loop in run_flow
        if "tap_px" in step:
            x, y = map(int, step["tap_px"])
            try:
                tap(cfg.device, (x, y))
            except Exception as e:
                print(f"[warn] tap_px {x},{y} failed: {e}")
            time.sleep(1)

        if "swipe_px" in step:
            x1, y1, x2, y2 = map(int, step["swipe_px"])
            dur = int(step.get("duration_ms", 400))
            try:
                swipe(cfg.device, (x1, y1), (x2, y2), dur)
            except Exception as e:
                print(f"[warn] swipe_px {x1},{y1}->{x2},{y2} failed: {e}")
            time.sleep(1)

        if step.get("type_text"):
            try:
                type_text(cfg.device, opener)
            except Exception as e:
                print(f"[warn] type_text failed: {e}")
            time.sleep(1)

    return opener

def load_cfg(path: Path) -> Config:
    d = json.loads(path.read_text())
    return Config(
        device=d.get("device_id"),
        scroll_start=tuple(d["scroll"]["start"]),
        scroll_end=tuple(d["scroll"]["end"]),
        duration_ms=int(d["scroll"].get("duration_ms", 450)),
        max_scrolls=int(d.get("max_scrolls", 6)),
        stop_when_stationary=int(d.get("stop_when_stationary", 3)),
        button_sequence=d.get("button_sequence", [])
    )

def reset_adb_and_app(pkg: str, device: Optional[str]):
    """Hard reset both sides: ADB server + target app process."""
    try:
        # Restart ADB server (covers 'device offline' / 255 errors)
        sh(["adb", "kill-server"])
        sh(["adb", "start-server"])
        if device:
            # Wait so the device is really back
            sh(["adb", "-s", device, "wait-for-device"])
    except Exception as e:
        print(f"[recover] adb restart failed: {e}")

    if pkg:
        try:
            # Force-stop and relaunch the app
            adb(device, "shell", "am", "force-stop", pkg)
            time.sleep(0.5)
            adb(device, "shell", "monkey", "-p", pkg, "-c",
                "android.intent.category.LAUNCHER", "1")
            time.sleep(1.2)
        except Exception as e:
            print(f"[recover] app restart failed: {e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkg", default="", help="App package (e.g., co.hinge.app)")
    ap.add_argument("--cfg", default="actions.json", help="Path to actions.json")
    ap.add_argument("--out", default="ocr_dump.txt", help="Where to save deduped OCR text")
    ap.add_argument("--prompt", required=True, help="Prompt to guide the opener style")
    ap.add_argument("--sleep_between_cycles", type=float, default=5.0, help="Pause (s) after each cycle")
    args = ap.parse_args()

    cfg = load_cfg(Path(args.cfg))
    out_txt = Path(args.out)

    cycle = 0
    while True:
        try:
            cycle += 1
            print(f"\n===== CYCLE {cycle} =====")
            opener = run_flow(args.pkg, cfg, out_txt, args.prompt)
            print(f"[cycle {cycle}] done.")
            time.sleep(args.sleep_between_cycles)
        except KeyboardInterrupt:
            print("\n[exit] Ctrl+C received. Bye.")
            break
        except Exception as e:
            # Log, try full recovery, and continue
            import traceback
            print(f"[error] {e}\n{traceback.format_exc()}")
            print("[recover] restarting adb and app, then continuing...")
            reset_adb_and_app(args.pkg, cfg.device)
            # Small backoff so we don't spin too fast on persistent errors
            time.sleep(2.0)
            continue
