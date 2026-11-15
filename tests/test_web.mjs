// Node checks for web/roshni.js: Pillow-identical resizing, and the decision rule.
import { createRequire } from "module";
import fs from "fs";
const require = createRequire(import.meta.url);
const R = require("../web/roshni.js");
let fails = 0;
const cases = process.argv.slice(2);   // "sw,sh,dw,dh" written by the Python side
for (const c of cases) {
  const [sw, sh, dw, dh] = c.split(",").map(Number);
  const rgb = fs.readFileSync(`/tmp/rs_${sw}x${sh}.raw`);
  const rgba = new Uint8ClampedArray(sw * sh * 4);
  for (let i = 0; i < sw * sh; i++) { rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2]; rgba[i * 4 + 3] = 255; }
  const out = R.resizePIL(rgba, sw, sh, dw, dh);
  const ref = fs.readFileSync(`/tmp/rs_${sw}x${sh}_to_${dw}x${dh}.raw`);
  let maxd = 0, nd = 0;
  for (let i = 0; i < dw * dh; i++) for (let ch = 0; ch < 3; ch++) {
    const d = Math.abs(out[i * 4 + ch] - ref[i * 3 + ch]); if (d > maxd) maxd = d; if (d) nd++;
  }
  console.log(`resize ${sw}x${sh} -> ${dw}x${dh}: max difference ${maxd}, differing values ${nd}`);
  if (maxd > 0) fails++;
}
const d = (a, b) => R.decide(a, b, 0.5, 0.5);
if (!(d(0.9, 0.9) === "REFER" && d(0.9, 0.1) === "REFER" && d(0.1, 0.9) === "RETAKE" && d(0.1, 0.1) === "RESCREEN")) { console.log("decide rule FAILED"); fails++; }
const P = R.softmax([1, 2, 3, 0.5, -1]);
if (Math.abs(P.reduce((a, b) => a + b, 0) - 1) > 1e-9) { console.log("softmax FAILED"); fails++; }
console.log(fails ? `FAILED (${fails})` : "all web checks passed");
process.exit(fails ? 1 : 0);
