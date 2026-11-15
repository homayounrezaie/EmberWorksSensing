"""Headless browser smoke test for web/index.html: no console errors, the
equation renders, forecasts finish, sliders refit.  Saves screenshots to
data/qa/<site>/web_*.png.  Usage: python tests/check_web.py"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PAGES = [("index.html", "HH1")]
errors = []


def wait_idle(page, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if page.evaluate("!document.body.classList.contains('busy') && (!document.getElementById('kpis') || "
                         "document.querySelectorAll('#kpis .kpi').length === 4)"):
            return time.time() - t0
        time.sleep(0.2)
    raise TimeoutError("page never finished computing")


with sync_playwright() as p:
    try:
        browser = p.chromium.launch(channel="chrome")
    except Exception:
        browser = p.chromium.launch()
    for page_file, site in PAGES:
      if not (ROOT / "web" / page_file).exists():
        continue
      page_url = (ROOT / "web" / page_file).as_uri()
      out = ROOT / "data" / "qa" / site
      out.mkdir(parents=True, exist_ok=True)
      print(f"#### {page_file}")
      for name, vp, scheme in [("desktop_light", (1360, 900), "light"), ("desktop_dark", (1360, 900), "dark"), ("phone", (390, 844), "light")]:
          ctx = browser.new_context(viewport={"width": vp[0], "height": vp[1]}, color_scheme=scheme, device_scale_factor=1)
          page = ctx.new_page()
          page.on("console", lambda m: errors.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
          page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
          t0 = time.time()
          page.goto(page_url)
          dt = wait_idle(page)
          print(f"{name}: all forecasts ready {time.time() - t0:.1f} s after load")
          eq = page.inner_text("#eq")
          print("   equation:", eq.replace("\n", " "))
          print("   status:", page.inner_text("#eqStatus").replace("\n", " | "))
          page.screenshot(path=str(out / f"web_{name}.png"), full_page=True)
          if name == "desktop_light":
              # model picker, zoom controls, slider
              for key in ("physics", "fbp", "persist", "flammap"):
                  page.click(f'#modelTable tbody tr[data-k="{key}"]'); time.sleep(0.2)
                  print(f"   model {key}: panel '{page.inner_text('#physName')}', caption '{page.inner_text('#capPhys')}'")
              # GIS layer panel: switch rasters and forecasts on and off
              for lid in ("plot", "contours", "arrows", "m:fbp", "front", "wind"):
                  page.check(f'#lyrBody input[data-id="{lid}"]'); time.sleep(0.1)
              page.screenshot(path=str(out / "web_layers.png"))
              for lid in ("plot", "contours", "arrows", "m:fbp", "front", "wind"):
                  page.uncheck(f'#lyrBody input[data-id="{lid}"]')
              print("   layers:", len(page.locator("#lyrBody input[type=checkbox]").all()), "toggles,",
                    page.locator("#lyrBody input:checked").count(), "on")
              page.click("#zoomIn"); page.click("#zoomIn"); time.sleep(0.2)
              page.screenshot(path=str(out / "web_zoomed.png"))
              page.click("#zoomFit")
              page.eval_on_selector("#passSlider", "e => { e.value = e.min; e.dispatchEvent(new Event('input')); }")
              time.sleep(0.3)
              print("   slider min ->", page.inner_text("#passLabel"))
              print("   model table:", " | ".join(r.replace("\t", " ") for r in page.locator("#modelTable tbody tr").all_inner_texts()))
          ctx.close()
    browser.close()
print("console errors/warnings:", errors if errors else "none")
sys.exit(1 if any(e.startswith(("error", "pageerror")) for e in errors) else 0)
