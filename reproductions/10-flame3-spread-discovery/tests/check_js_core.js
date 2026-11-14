// Verifies that the browser engine (web_template.html, CORE block) reproduces the
// Python pipeline: STLSQ coefficients and per-interval forecast metrics for every
// model (discovered, FlamMap MTT, Rothermel + Huygens, FBP C-7, persistence, no spread).
// usage: node tests/check_js_core.js [data.json]
const fs = require("fs"), path = require("path");
const root = path.resolve(__dirname, "..");
const tpl = fs.readFileSync(path.join(root, "pipeline", "web_template.html"), "utf8");
const src = tpl.split("// ==CORE START==")[1].split("// ==CORE END==")[0].replace(/^[^\n]*\n/, "");
const CORE = new Function("module", src + "\nreturn CORE;")({});
const file = process.argv[2] || "hanna_hammock.json";
const D = JSON.parse(fs.readFileSync(path.join(root, "web", "data", file), "utf8"));
const m = D.meta, H = m.H, W = m.W, N = H * W;
CORE.setKeys(m.terms.map(t => t.key), m.terms.map(t => t.sign || 0), m.terms.map(t => !!t.forced), m.terms.map(t => (t.pen == null ? 1 : t.pen)));
const MOD = CORE.models(m, H, W);
const fuel = CORE.decodeRLE(D.fuel, N), burnt = D.burnt.map(r => CORE.decodeRLE(r, N));
console.log(`== ${file} (${m.site}, ${m.terms.length} terms, grid ${W}x${H} @ ${m.res} m, ${D.intervals.length} intervals)`);
// relative tolerance 2 %, plus absolute floors for near-zero values (quiet intervals):
// IoU 0.005, new-area IoU 0.05 (one pixel flip when only ~20 px burn in an interval), front error half a pixel, arrival 1 s
const ABS = {iou: 0.005, iou_new: 0.05, front_mean_m: 0.5 * m.res, arrival_err_s: 1.0};
const MODELS = [["disc", "discovered"], ["baseline", "baseline"], ...m.models.map(x => [x.key, x.key])];
let fails = 0, worst = {beta: 0};
for (const k of Object.keys(D.splits)) {
  const sp = D.splits[k], f = CORE.stlsq(CORE.sumBlocks(D.blocks, new Set(sp.train)), sp.lam_cv, sp.ridge != null ? sp.ridge : m.ridge);
  const db = Math.max(...f.beta.map((b, i) => Math.abs(b - sp.py.beta[i])));
  const sameActive = f.active.every((a, i) => a === sp.py.active[i]);
  worst.beta = Math.max(worst.beta, db);
  const okB = db < 1e-4 && sameActive;
  if (!okB) fails++;
  console.log(`k=${k}: max|Δβ|=${db.toExponential(2)} sameTerms=${sameActive} ${okB ? "OK" : "MISMATCH"}`);
  for (const [model, py] of MODELS) {
    let dm = 0, where = "";
    D.intervals.forEach((iv, ii) => {
      const opt = MOD.opt(model, iv, model === "disc" ? {beta: f.beta, Iref: sp.refs.I} : {});
      const g = CORE.forecast(model, iv, burnt[iv.i], fuel, H, W, m.res, opt);
      const mt = CORE.metrics(g.pred, burnt[iv.i + 1], burnt[iv.i], fuel, g.AT, iv.dt, H, W, m.res);
      for (const key of ["iou", "iou_new", "front_mean_m", "arrival_err_s"]) {
        const a = mt[key], b = sp.py.metrics[py][key][ii];
        if (!isFinite(a) || !isFinite(b)) continue;
        const err = Math.abs(a - b) <= ABS[key] ? 0 : Math.abs(a - b) / Math.max(Math.abs(b), 1e-9);
        if (err > dm) { dm = err; where = `${iv.i}->${iv.i + 1} ${key} js ${a.toFixed(4)} py ${b}`; }
      }
    });
    const ok = dm < 0.02;
    if (!ok) fails++;
    console.log(`   ${model.padEnd(9)} max rel Δmetric=${dm.toExponential(2)} ${ok ? "OK" : "MISMATCH " + where}`);
  }
}
console.log(fails ? `FAILED (${fails})` : "ALL MATCH", JSON.stringify(worst));
process.exit(fails ? 1 : 0);
