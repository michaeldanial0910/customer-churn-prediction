"""(Used to make demo/churn_demo.mp4.) Drive the real Streamlit app in a visible Chromium on Xvfb and log timestamps of each scene.
Run under:  xvfb-run -s "-screen 0 1920x1080x24" python demo/record.py  (app running on :8501; ffmpeg x11grab started here)
"""
import json, os, subprocess, time
from playwright.sync_api import sync_playwright

OUT = os.path.dirname(os.path.abspath(__file__))
URL = "http://localhost:8501"
CHROME = 88                      # px of browser chrome pushed above the screen (kiosk has no effect without a WM)
QUIET = "A long-tenure churner the model missed"
marks = []
T0 = None


def mark(name):
    marks.append({"name": name, "t": round(time.time() - T0, 2)})
    print(f"{marks[-1]['t']:7.2f}  {name}", flush=True)


CURSOR_JS = """
(() => {
  if (window.__cursor) return;
  const c = document.createElement('div');
  c.innerHTML = `<svg width="26" height="26" viewBox="0 0 24 24"><path d="M4 2 L4 19 L8.6 14.8 L11.6 21.4 L14.4 20.2 L11.4 13.6 L17.6 13.4 Z"
     fill="#1C2333" stroke="white" stroke-width="1.6" stroke-linejoin="round"/></svg>`;
  Object.assign(c.style, {position:'fixed', left:'0px', top:'0px', zIndex: 2147483647, pointerEvents:'none',
                          transform:'translate(-4px,-2px)', transition:'transform 60ms'});
  const ring = document.createElement('div');
  Object.assign(ring.style, {position:'fixed', width:'34px', height:'34px', borderRadius:'50%', border:'2px solid #D4562E',
     zIndex: 2147483646, pointerEvents:'none', opacity:'0', transform:'translate(-17px,-17px) scale(0.4)',
     transition:'opacity 350ms, transform 350ms'});
  const add = () => { document.body.appendChild(ring); document.body.appendChild(c); };
  if (document.body) add(); else document.addEventListener('DOMContentLoaded', add);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX+'px'; c.style.top = e.clientY+'px';
                                               ring.style.left = e.clientX+'px'; ring.style.top = e.clientY+'px'; }, true);
  document.addEventListener('mousedown', () => { ring.style.transition='none'; ring.style.opacity='0.9';
      ring.style.transform='translate(-17px,-17px) scale(0.4)'; requestAnimationFrame(() => { ring.style.transition='opacity 450ms, transform 450ms';
      ring.style.opacity='0'; ring.style.transform='translate(-17px,-17px) scale(1.3)'; }); }, true);
  window.__cursor = true;
})();
"""


def warm_up(p):
    """Visit every page once so model loading, SHAP and caches are warm before recording."""
    b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    pg = b.new_page(viewport={"width": 1536, "height": 864})
    for path in ["", "contacts", "thresholds", "results", "why"]:
        pg.goto(f"{URL}/{path}", wait_until="commit")
        pg.wait_for_selector("h1", timeout=90000)
        pg.wait_for_timeout(3000)
    b.close()


def main():
    global T0
    with sync_playwright() as p:
        warm_up(p)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "x11grab", "-framerate", "30",
                           "-video_size", "1920x1080", "-draw_mouse", "0", "-i", os.environ["DISPLAY"],
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "14", "-pix_fmt", "yuv420p",
                           f"{OUT}/raw.mp4"], stdin=subprocess.PIPE)
    T0 = time.time()
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path="/opt/pw-browsers/chromium", headless=False,
                              args=["--kiosk", f"--window-position=0,-{CHROME}", f"--window-size=1920,{1080 + CHROME}",
                                    "--force-device-scale-factor=1.25", "--hide-scrollbars",
                                    "--disable-infobars", "--no-first-run"])
        ctx = b.new_context(no_viewport=True)
        ctx.add_init_script(CURSOR_JS)
        pg = ctx.new_page()
        pg.goto(URL, wait_until="commit")
        pg.wait_for_selector("text=Will this customer leave?", timeout=90000)
        pg.wait_for_selector(".verdict", timeout=90000)
        pg.wait_for_timeout(2500)          # fonts settle
        pg.evaluate(CURSOR_JS)
        m = pg.mouse
        m.move(760, 520)
        mark("ready")
        main_el = "section.stMain, [data-testid='stMain']"

        def glide(locator, steps=28, click=True, dx=0, dy=0):
            box = locator.bounding_box()
            x, y = box["x"] + box["width"] / 2 + dx, box["y"] + box["height"] / 2 + dy
            m.move(x, y, steps=steps)
            pg.wait_for_timeout(180)
            if click:
                m.click(x, y)

        def scroll_to(px, dur_ms=1600):
            pg.evaluate(f"""(() => {{ const el = document.querySelector("{main_el}") || document.scrollingElement;
               const start = el.scrollTop, end = {px}, t0 = performance.now(), d = {dur_ms};
               function step(t) {{ const k = Math.min(1, (t - t0) / d), e = k<.5 ? 2*k*k : 1-Math.pow(-2*k+2,2)/2;
                  el.scrollTop = start + (end - start) * e; if (k < 1) requestAnimationFrame(step); }}
               requestAnimationFrame(step); }})()""")
            pg.wait_for_timeout(dur_ms + 150)

        def settle(sel=None, ms=900):
            if sel:
                pg.wait_for_selector(sel, timeout=90000)
            pg.wait_for_timeout(ms)

        def drag_slider(slider, lo, hi, value, steps=45):
            """Drag a Streamlit slider's thumb to `value` (slider range lo..hi)."""
            thumb = slider.locator('input[type="range"]').first.locator("xpath=../..").bounding_box()
            track = slider.locator('[role="group"] > div').first.bounding_box()
            sx, sy = thumb["x"] + thumb["width"] / 2, thumb["y"] + thumb["height"] / 2
            tx = track["x"] + (value - lo) / (hi - lo) * track["width"]   # thumb sits at left: x% of the track
            m.move(sx, sy, steps=22)
            pg.wait_for_timeout(200)
            m.down()
            m.move(tx, sy, steps=steps)
            pg.wait_for_timeout(250)
            m.up()

        # ---- Score a customer ----
        mark("score_start")
        glide(pg.locator(".verdict .reasons"), click=False, steps=30)
        pg.wait_for_timeout(1900)
        glide(pg.locator('[data-testid="stSelectbox"]').first)
        pg.wait_for_timeout(500)
        glide(pg.get_by_role("option", name=QUIET), steps=16)
        settle("text=What really happened", 1200)
        mark("quiet_loaded")
        scroll_to(300, 1100)                   # bring "what really happened" into view
        glide(pg.locator(".verdict .truth"), click=False, steps=26)
        pg.wait_for_timeout(2200)
        mark("score_end")
        scroll_to(0, 600)

        # ---- Contact list ----
        glide(pg.get_by_role("link", name="Contact list"))
        settle("text=This month's contact list", 300)
        pg.wait_for_selector(".vega-embed", timeout=90000)
        pg.wait_for_timeout(1300)
        mark("contacts_start")
        drag_slider(pg.locator('[data-testid="stSlider"]').first, 5, 100, 20)
        settle(ms=1300)
        glide(pg.get_by_text("Show what really happened (hindsight)", exact=True), dx=-150)
        settle("text=customers who left are on the list", 1900)
        mark("contacts_hindsight")
        scroll_to(560, 1300)
        pg.wait_for_timeout(1300)
        mark("contacts_end")

        # ---- Thresholds ----
        scroll_to(0, 500)
        glide(pg.get_by_role("link", name="Thresholds"))
        settle("text=Where to draw the line", 300)
        pg.wait_for_selector(".vega-embed", timeout=90000)
        pg.wait_for_timeout(1300)
        mark("thr_start")
        drag_slider(pg.locator('[data-testid="stSlider"]').first, 0.02, 0.90, 0.50, steps=50)
        settle(ms=1900)
        mark("thr_drag")
        drag_slider(pg.locator('[data-testid="stSlider"]').first, 0.02, 0.90, 0.15, steps=40)
        settle(ms=1400)
        mark("thr_end")

        # ---- Results ----
        glide(pg.get_by_role("link", name="Results"))
        settle("text=XGBoost vs a simple baseline", 300)
        pg.wait_for_selector(".vega-embed", timeout=90000)
        pg.wait_for_timeout(1300)
        mark("results_start")
        scroll_to(330, 1300)
        pg.wait_for_timeout(1300)
        scroll_to(720, 1300)
        pg.wait_for_timeout(1500)
        mark("results_end")

        # ---- Why customers leave ----
        scroll_to(0, 500)
        glide(pg.get_by_role("link", name="Why customers leave"))
        settle("text=Why customers leave", 300)
        pg.wait_for_selector(".vega-embed", timeout=90000)
        pg.wait_for_timeout(1300)
        mark("why_start")
        scroll_to(240, 1100)
        pg.wait_for_timeout(900)
        glide(pg.locator('[data-testid="stButtonGroup"] button').filter(has_text="Tenure").first)
        settle(ms=1700)
        mark("why_tenure")
        scroll_to(760, 1400)
        pg.wait_for_timeout(1600)
        mark("why_end")
        b.close()
    ff.communicate(b"q")
    json.dump(marks, open(f"{OUT}/marks.json", "w"), indent=1)


main()
