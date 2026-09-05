#!/usr/bin/env python3
"""Compare generated CSS token ids with expected_v2.json in Chromium.

This is external test code. The generated qwennie page itself still uses zero JavaScript.
"""
import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent


async def run(args):
    site = HERE / args.site / "index.html"
    css = site.parent / "model_v2.css"
    expected = json.loads((HERE / args.expected).read_text())["combos"]
    items = list(expected.items())[args.offset : args.offset + args.count if args.count else None]

    async with async_playwright() as p:
        launch = {"headless": True, "args": ["--no-sandbox", "--disable-dev-shm-usage"]}
        if args.chromium:
            launch["executable_path"] = args.chromium
        browser = await p.chromium.launch(**launch)
        page = await browser.new_page()
        html = site.read_text().replace(
            '<link rel="stylesheet" href="model_v2.css">',
            "<style>" + css.read_text() + "</style>",
        )
        await page.set_content(html, wait_until="load", timeout=args.timeout * 1000)
        await page.evaluate("""() => {
          window.__setPrompt = (turn, text) => {
            const ws = text.split(/\\s+/).filter(Boolean);
            for (let s = 0; s < 8; s++) {
              const el = document.getElementById(`u${turn}s${s}`);
              const want = ws[s] ?? '·';
              const opt = [...el.options].find(o => o.textContent === want);
              if (!opt) throw new Error(`token not selectable: ${want}`);
              el.value = opt.value;
              el.dispatchEvent(new Event('change', {bubbles:true}));
            }
          };
          window.__ids = turn => [...document.querySelectorAll(`.gen.t${turn}`)]
            .map(w => Number(getComputedStyle(w).getPropertyValue('--tk')));
        }""")

        seed_id = {17: "ball", 53: "bone"}
        temp_id = {"0.45": "cozy", "0.8": "waggy", "1.35": "zoomies"}
        passed = 0
        for n, (key, want) in enumerate(items, 1):
            seed, temp, q1, q2 = key.split("|")
            got = await page.evaluate("""async x => {
              document.getElementById(`seed-${x.seed}`).checked = true;
              document.getElementById(`temp-${x.temp}`).checked = true;
              __setPrompt(1, x.q1); __setPrompt(2, x.q2);
              await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
              return {ids1: __ids(1), ids2: __ids(2)};
            }""", {"seed": seed_id[int(seed)], "temp": temp_id[temp], "q1": q1, "q2": q2})
            want_ids = {"ids1": want["ids"][10:26], "ids2": want["ids"][36:64]}
            ok = got == want_ids
            print(f"{args.offset+n:02d} {'PASS' if ok else 'FAIL'} {key}", flush=True)
            if not ok:
                print(" expected", want_ids)
                print(" got     ", got)
            passed += int(ok)

        await browser.close()
        print(f"BATCH {passed}/{len(items)}")
        return 0 if passed == len(items) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default="site")
    ap.add_argument("--expected", default="expected_v2.json")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--chromium", default=None, help="optional Chromium executable path")
    ap.add_argument("--timeout", type=int, default=60)
    raise SystemExit(asyncio.run(run(ap.parse_args())))


if __name__ == "__main__":
    main()
