"""(Used to make demo/churn_demo.mp4.) Edit raw.mp4 (real recording of the Streamlit app) into a 30 s, 1080p silent demo.
Needs the IBM Plex Sans Arabic TTFs in ~/.fonts (same fonts as the app)."""
import json, os, subprocess
from PIL import Image, ImageDraw, ImageFilter, ImageFont

D = os.path.dirname(os.path.abspath(__file__))
F = os.path.expanduser("~/.fonts/IBMPlexSansArabic-")
INK, PAPER, SOFT = (28, 35, 51), (247, 248, 250), (154, 166, 200)
W, H = 1920, 1080
CW, CH = 1648, 927                      # recording window size inside the frame
CX, CY = (W - CW) // 2, 44              # window position; caption strip sits underneath
TITLE, OUTRO, XF = 2.4, 2.6, 0.5        # card durations and crossfade length
TOTAL = 30.0

marks = {m["name"]: m["t"] for m in json.load(open(f"{D}/marks.json"))}
START, END = marks["ready"] - 0.1, marks["why_end"] + 0.2
main_len = TOTAL - TITLE - OUTRO + 2 * XF            # output seconds available for the app footage
SPEED = (END - START) / main_len
print(f"raw {END - START:.1f}s -> {main_len:.1f}s at {SPEED:.2f}x")


def font(weight, size):
    return ImageFont.truetype(f"{F}{weight}.ttf", size)


def out_t(raw_t):                                     # raw timestamp -> time inside the main clip
    return (raw_t - START) / SPEED


# ---- static assets ------------------------------------------------------------------------
bg = Image.new("RGB", (W, H), INK)
shadow = Image.new("L", (W, H), 0)
ImageDraw.Draw(shadow).rounded_rectangle((CX, CY + 14, CX + CW, CY + CH + 14), 22, fill=120)
shadow = shadow.filter(ImageFilter.GaussianBlur(26))
bg.paste(Image.new("RGB", (W, H), (8, 11, 18)), (0, 0), shadow)
bg.save(f"{D}/bg.png")
mask = Image.new("L", (CW, CH), 0)
ImageDraw.Draw(mask).rounded_rectangle((0, 0, CW - 1, CH - 1), 18, fill=255)
mask.save(f"{D}/mask.png")


def card(path, lines):
    im = Image.new("RGB", (W, H), INK)
    d = ImageDraw.Draw(im)
    y = (H - sum(h for _, _, h, _ in lines)) // 2
    for text, f, h, color in lines:
        w = d.textlength(text, font=f)
        d.text(((W - w) / 2, y), text, font=f, fill=color)
        y += h
    # the app's own signature: the three decision tiers on the test customers, no action -> offer -> call first
    sx, sy, sw = (W - 360) // 2, y + 18, 360
    parts = [((138, 148, 166), .48), ((176, 125, 0), .32), ((212, 86, 46), .20)]
    for i, (c, frac) in enumerate(parts):
        x0 = sx + sum(p for _, p in parts[:i]) * sw
        d.rounded_rectangle((x0 + (2 if i else 0), sy, x0 + frac * sw, sy + 12), 6, fill=c)
    im.save(path)


card(f"{D}/title.png", [("Churn risk", font("SemiBold", 92), 128, PAPER),
                        ("7,043 telecom customers: who will leave, and why", font("Regular", 38), 70, SOFT),
                        ("UneeQ Interns, Machine Learning track", font("Regular", 28), 40, SOFT)])
card(f"{D}/outro.png", [("streamlit run app.py", font("SemiBold", 76), 116, PAPER),
                        ("Five pages: Score a customer, Contact list, Thresholds, Results, Why customers leave",
                         font("Regular", 34), 64, SOFT),
                        ("Michael Danial, October 2026", font("Regular", 28), 40, SOFT)])

nav = lambda scene: marks[f"{scene}_start"] - 1.5          # roughly when the nav link is clicked
captions = [  # (text, raw start, raw end)
    ("Score any customer: a churn probability and the reasons behind it", marks["score_start"], marks["quiet_loaded"] - 1.0),
    ("A real held-out customer the model missed: a long-tenure quiet leaver", marks["quiet_loaded"] - 1.0, nav("contacts")),
    ("Contact the riskiest 20% and reach half of everyone who leaves", nav("contacts"), nav("thr")),
    ("Move the line: the default 0.5 costs more than contacting everyone", nav("thr"), nav("results")),
    ("Tuned XGBoost ties a simple baseline; the threshold moves the money", nav("results"), nav("why")),
    ("Month-to-month customers: 55% of the base, 89% of everyone who leaves", nav("why"), END),
]
cap_files = []
for i, (text, a, b) in enumerate(captions):
    im = Image.new("RGBA", (W, H - CY - CH), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    f = font("Medium", 30)
    w = d.textlength(text, font=f)
    d.text(((W - w) / 2, 26), text, font=f, fill=PAPER + (255,))
    p = f"{D}/cap_{i}.png"
    im.save(p)
    cap_files.append((p, max(0, out_t(a)), min(main_len, out_t(b))))

# ---- ffmpeg graph ---------------------------------------------------------------------------
inputs = ["-ss", f"{START:.2f}", "-to", f"{END:.2f}", "-i", f"{D}/raw.mp4",
          "-loop", "1", "-t", f"{main_len:.2f}", "-i", f"{D}/bg.png",
          "-loop", "1", "-i", f"{D}/mask.png",
          "-loop", "1", "-t", f"{TITLE:.2f}", "-i", f"{D}/title.png",
          "-loop", "1", "-t", f"{OUTRO:.2f}", "-i", f"{D}/outro.png"]
for p, _, _ in cap_files:
    inputs += ["-loop", "1", "-t", f"{main_len:.2f}", "-i", p]

g = ["[0:v]setpts=PTS-STARTPTS[raw]",
     f"[raw]setpts=PTS/{SPEED:.4f},fps=30,scale={CW}:{CH}:flags=lanczos,format=rgba[rec]",
     f"[2:v]format=gray,scale={CW}:{CH}[m]",
     "[rec][m]alphamerge[win]",
     "[1:v]fps=30,format=rgba[bgv]",
     f"[bgv][win]overlay={CX}:{CY}:shortest=1[v0]"]
last = "v0"
for i, (_, a, b) in enumerate(cap_files):
    k = 5 + i
    fi, fo = 0.3, 0.3
    g.append(f"[{k}:v]fps=30,format=rgba,fade=in:st={a:.2f}:d={fi}:alpha=1,"
             f"fade=out:st={max(a + fi, b - fo):.2f}:d={fo}:alpha=1[c{i}]")
    g.append(f"[{last}][c{i}]overlay=0:{CY + CH}:shortest=1[v{i + 1}]")
    last = f"v{i + 1}"
g.append(f"[{last}]trim=duration={main_len:.2f},setpts=PTS-STARTPTS,format=yuv420p[main]")
g.append("[3:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[t]")
g.append("[4:v]fps=30,format=yuv420p,setpts=PTS-STARTPTS[o]")
g.append(f"[t][main]xfade=transition=fade:duration={XF}:offset={TITLE - XF:.2f}[tm]")
g.append(f"[tm][o]xfade=transition=fade:duration={XF}:offset={TITLE + main_len - 2 * XF:.2f},"
         f"fade=out:st={TOTAL - 0.6:.2f}:d=0.6[out]")

cmd = ["ffmpeg", "-y", "-v", "error", *inputs, "-filter_complex", ";".join(g), "-map", "[out]",
       "-t", f"{TOTAL:.2f}", "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
       "-movflags", "+faststart", "-an", f"{D}/churn_demo.mp4"]
subprocess.run(cmd, check=True)
print("done")
