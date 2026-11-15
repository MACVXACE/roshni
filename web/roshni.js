/* Roshni in the browser: the same frozen tool as the Python code, run with ONNX Runtime Web.
   The photo is processed on this device and never uploaded.
   Pure functions at the top are also tested in Node (tests/test_web.mjs). */

const R = {};

/* PIL "L" conversion, used to find the round fundus inside its black border. */
R.cropBox = function (rgba, w, h, tol) {
  let x0 = w, y0 = h, x1 = -1, y1 = -1, count = 0;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const L = (rgba[i] * 19595 + rgba[i + 1] * 38470 + rgba[i + 2] * 7471 + 0x8000) >> 16;
      if (L > tol) {
        count++;
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (count < 0.05 * w * h) return { x: 0, y: 0, w: w, h: h };   // almost all dark: leave it to the quality check
  return { x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1 };
};

/* Resize exactly like Pillow's Image.resize(..., BILINEAR), which the Python tool uses:
   a triangle filter widened when shrinking, horizontal pass then vertical, 22-bit fixed point. */
R._coeffs = function (inSize, outSize) {
  const scale = inSize / outSize, fscale = Math.max(scale, 1), support = fscale, ss = 1 / fscale;
  const PREC = 22, list = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5); if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5); if (xmax > inSize) xmax = inSize;
    const n = xmax - xmin, k = new Float64Array(n);
    let ww = 0;
    for (let x = 0; x < n; x++) { const t = Math.abs((x + xmin - center + 0.5) * ss); k[x] = t < 1 ? 1 - t : 0; ww += k[x]; }
    const ki = new Int32Array(n);
    for (let x = 0; x < n; x++) { const v = (ww !== 0 ? k[x] / ww : k[x]) * (1 << PREC); ki[x] = Math.trunc(v < 0 ? v - 0.5 : v + 0.5); }
    list.push({ xmin: xmin, k: ki });
  }
  return list;
};

R.resizePIL = function (src, sw, sh, dw, dh) {
  const PREC = 22, half = 1 << (PREC - 1), div = 1 << PREC;
  const clip = v => (v < 0 ? 0 : v > 255 ? 255 : v);
  let w = sw, h = sh, buf = src;
  if (dw !== sw) {
    const cs = R._coeffs(sw, dw), out = new Uint8ClampedArray(dw * h * 4);
    for (let y = 0; y < h; y++) for (let xx = 0; xx < dw; xx++) {
      const { xmin, k } = cs[xx];
      let r = half, g = half, b = half;
      for (let x = 0; x < k.length; x++) { const i = (y * w + xmin + x) * 4; r += buf[i] * k[x]; g += buf[i + 1] * k[x]; b += buf[i + 2] * k[x]; }
      const o = (y * dw + xx) * 4;
      out[o] = clip(Math.floor(r / div)); out[o + 1] = clip(Math.floor(g / div)); out[o + 2] = clip(Math.floor(b / div)); out[o + 3] = 255;
    }
    buf = out; w = dw;
  }
  if (dh !== sh) {
    const cs = R._coeffs(sh, dh), out = new Uint8ClampedArray(w * dh * 4);
    for (let yy = 0; yy < dh; yy++) {
      const { xmin, k } = cs[yy];
      for (let x = 0; x < w; x++) {
        let r = half, g = half, b = half;
        for (let y = 0; y < k.length; y++) { const i = ((xmin + y) * w + x) * 4; r += buf[i] * k[y]; g += buf[i + 1] * k[y]; b += buf[i + 2] * k[y]; }
        const o = (yy * w + x) * 4;
        out[o] = clip(Math.floor(r / div)); out[o + 1] = clip(Math.floor(g / div)); out[o + 2] = clip(Math.floor(b / div)); out[o + 3] = 255;
      }
    }
    buf = out; h = dh;
  }
  return buf;
};

R.softmax = function (z) {
  const m = Math.max(...z);
  const e = z.map(v => Math.exp(v - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map(v => v / s);
};

R.referableProb = function (P, from) {
  let s = 0;
  for (let k = from; k < P.length; k++) s += P[k];
  return s;
};

/* StandardScaler + logistic regression, exported from scikit-learn. */
R.poorProb = function (emb, gate) {
  if (!gate) return 0;
  let z = gate.intercept;
  for (let c = 0; c < emb.length; c++) z += ((emb[c] - gate.mean[c]) / gate.scale[c]) * gate.coef[c];
  return 1 / (1 + Math.exp(-z));
};

/* Refer whenever referable disease is suspected (a poor photo never blocks a referral);
   otherwise a poor photo gets RETAKE instead of an all-clear. */
R.decide = function (pRef, pPoor, refThr, poorThr) {
  if (pRef >= refThr) return "REFER";
  if (pPoor >= poorThr) return "RETAKE";
  return "RESCREEN";
};

/* Grad-CAM for the referable score, computed exactly without gradients: the network ends in
   global average pooling and one linear layer, so the gradient at the last feature map is the same at
   every position: sum_k (dscore/dlogit_k) * W[k][c]. Matches drscreen.gradcam up to a positive scale. */
R.heatmap = function (fmap, C, H, W, P, headW, from) {
  let sRef = 0;
  for (let k = from; k < P.length; k++) sRef += P[k];
  const g = P.map((p, k) => (k >= from ? p / sRef : 0) - p);
  const alpha = new Float32Array(C);
  for (let c = 0; c < C; c++) {
    let a = 0;
    for (let k = 0; k < P.length; k++) a += g[k] * headW[k][c];
    alpha[c] = a;
  }
  const cam = new Float32Array(H * W);
  for (let c = 0; c < C; c++) {
    const a = alpha[c], off = c * H * W;
    if (a === 0) continue;
    for (let i = 0; i < H * W; i++) cam[i] += a * fmap[off + i];
  }
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < cam.length; i++) {
    cam[i] = Math.max(0, cam[i]);
    if (cam[i] < mn) mn = cam[i];
    if (cam[i] > mx) mx = cam[i];
  }
  for (let i = 0; i < cam.length; i++) cam[i] = (cam[i] - mn) / (mx - mn + 1e-8);
  return cam;
};

if (typeof module !== "undefined") module.exports = R;

/* ------------------------------------------------------------------ browser app
   The screening itself (prepare, run, the decision rule, Grad-CAM) is unchanged.
   What is new here is only how the page shows it: one interface language for the whole page
   (English, Urdu, Uzbek), an image preview, the workflow stages and a "new assessment" button.
   For the automated tests and the video script, #status, #decision, #scores and #timing still hold
   the original English text (hidden); the visible text is translated. */
if (typeof window !== "undefined") {
  const COLORS = { REFER: "#b42318", RETAKE: "#b54708", RESCREEN: "#067647" };
  const TITLES = { REFER: "Refer to an eye specialist", RETAKE: "Retake the photo", RESCREEN: "Rescreen in 12 months" };
  const STATUS_EN = {
    loading: "Loading the model onto this device…",
    ready: "Ready. Choose a retinal photo or an example.",
    working: "Screening on this device…",
    done: "Done. Nothing was uploaded.",
    errModel: "Could not load the model: ",
    errImage: "Could not open that image.",
    errRun: "Something went wrong: ",
  };

  /* Interface text. Urdu: to be checked by Zainab. Uzbek: to be checked by Laziza.
     The patient-note texts (REFER/RETAKE/RESCREEN/DISCLAIMER) still come from roshni_web.json for
     English and Urdu; the Uzbek note below is shown with "(translation being checked)" until approved. */
  const UI = {
    en: {
      dir: "ltr", name: "English",
      brandSub: "Diabetic eye screening · research prototype",
      langLabel: "Language",
      chipLocal: "Runs in this browser", chipPhoto: "Photo is not uploaded", chipProto: "Research prototype",
      kicker: "Primary care · Pakistan",
      title: "Retinal image assessment",
      lede: "For a primary-care diabetes clinic without an eye specialist. Upload one retinal photograph; Roshni suggests one next step for the doctor to confirm.",
      privacy: "The photo is processed inside this browser and is not uploaded. After the first visit, the tool also works without internet.",
      disclaimer: "Research prototype for a student hackathon. Not medical advice, not a diagnosis, and not tested on patients in Pakistan.",
      steps: ["Upload retinal image", "Check image quality", "Analyse", "View result"],
      stepHints: ["JPG or PNG fundus photo", "Is the photo clear enough?", "Referable-disease score", "Next step for the doctor"],
      uploadTitle: "Upload a retinal photograph",
      uploadHint: "Fundus photo from the clinic camera or your files · JPG or PNG · drag and drop or click",
      uploadBtn: "Choose image", replaceBtn: "Replace image", previewCap: "Selected image",
      examplesTitle: "Or try an example photo",
      examplesCredit: "Example photos: DeepDRiD (Liu et al., Patterns 2022), CC BY-SA 4.0. Captions give the experts' grade and photo-quality label, not the tool's answer.",
      exGrade: "Expert grade", exQuality: "photo quality", good: "good", poor: "poor",
      exMissed: "Shown on purpose: one of the referable photos Roshni misses",
      noteLangs: "Languages on the patient note", checking: "translation being checked",
      status: {
        loading: "Loading the model onto this device…", ready: "Ready. Upload a retinal (fundus) photo or choose an example. Other kinds of picture give meaningless results.",
        working: "Checking image quality and analysing on this device…", done: "Done. The photo was not uploaded.",
        errModel: "Could not load the model: ", errImage: "Could not open that image. Please choose a JPG or PNG retinal photo.", errRun: "Something went wrong: ",
      },
      offlineReady: "Saved on this device: works without internet from now on.",
      resultTitle: "Result", empty: "Upload an image to begin. The result will appear here.",
      kicker2: "Suggested next step · for the doctor to confirm",
      R: {
        REFER: { tag: "REFER", head: "Referable diabetic retinopathy suspected", body: "Moderate or worse diabetic retinopathy may be present. Clinician assessment is needed.", next: "Next step: refer to an eye specialist, after the doctor confirms." },
        RETAKE: { tag: "RETAKE", head: "The image needs to be captured again", body: "The photo is not clear enough to rule out disease.", next: "Next step: retake the photo. If it is still unclear, refer to an eye specialist." },
        RESCREEN: { tag: "RESCREEN", head: "Referable diabetic retinopathy not detected in this gradable image", body: "This is not a guarantee that the eye is healthy, and other eye diseases are not checked.", next: "Next step: continue routine eye check-ups as the doctor advises (usually every 12 months)." },
      },
      scores: (r, rt, p, pt) => `Referable-disease score ${r} (refer at ${rt} or above) · Poor-photo score ${p} (retake at ${pt} or above)`,
      timing: s => `Processed on this device in ${s} s.`,
      noteTitle: "Note for the doctor and patient",
      imagesTitle: "Original photograph and heat map",
      photoCap: "Photo as the model sees it", heatCap: "Heat map (Grad-CAM) for referable disease",
      heatWhy: "The heat map shows which parts of the photo drove the score. It is not proof that disease is present or absent.",
      gradesTitle: "Estimated severity (model output)",
      grades: ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
      newBtn: "Start new assessment",
      evidenceTitle: "Evidence and limits",
      results: (a, b) => `On ${a.n} unseen photos from India: sensitivity ${a.se}%, specificity ${a.sp}%. On ${b.n} photos from Chinese hospitals it never saw: sensitivity ${b.se}%, specificity ${b.sp}%.`,
      thr: (t, s) => `Referral threshold ${t}, frozen before testing (set for ${s}% sensitivity on validation photos).`,
      model: "Model: EfficientNet-B0 fine-tuned on APTOS 2019 (India), tested on DeepDRiD (China). Not tested on Pakistani patients or low-cost cameras. Does not detect macular oedema.",
      repo: "Code, results and limitations",
      ctaStart: "Start an assessment", ctaEvidence: "Evidence and limits",
      heroCap: "Example retinal photograph · DeepDRiD, CC BY-SA 4.0",
      h1t: "On-device", h1d: "The photo is analysed in this browser and is not uploaded.",
      h2t: "Offline-ready", h2d: "After the first visit, it works without internet.",
      h3t: "Three languages", h3d: "English, Urdu and Uzbek interface; the Uzbek patient note is being checked.",
      h4t: "Open code", h4d: "Code and results are public under the MIT licence.",
      fine: "Roshni · VERITAS AI Hackathon 2026 · Research prototype, not a medical device",
    },
    ur: {
      dir: "rtl", name: "اردو",
      brandSub: "ذیابیطس سے آنکھ کی اسکریننگ · تحقیقی نمونہ",
      langLabel: "زبان",
      chipLocal: "اسی براؤزر میں چلتا ہے", chipPhoto: "تصویر اپ لوڈ نہیں ہوتی", chipProto: "تحقیقی نمونہ",
      kicker: "بنیادی صحت مرکز · پاکستان",
      title: "ریٹینا کی تصویر کا جائزہ",
      lede: "ایسے ذیابیطس کلینک کے لیے جہاں آنکھوں کا ماہر موجود نہ ہو۔ ریٹینا کی ایک تصویر اپ لوڈ کریں؛ روشنی ڈاکٹر کی تصدیق کے لیے ایک اگلا قدم تجویز کرتی ہے۔",
      privacy: "تصویر اسی براؤزر کے اندر پراسیس ہوتی ہے اور اپ لوڈ نہیں کی جاتی۔ پہلی بار کھولنے کے بعد یہ انٹرنیٹ کے بغیر بھی کام کرتی ہے۔",
      disclaimer: "طلبہ کے ہیکاتھون کے لیے تحقیقی نمونہ۔ یہ طبی مشورہ یا تشخیص نہیں، اور پاکستان میں مریضوں پر اس کی جانچ نہیں ہوئی۔",
      steps: ["ریٹینا کی تصویر اپ لوڈ کریں", "تصویر کا معیار جانچیں", "تجزیہ", "نتیجہ دیکھیں"],
      stepHints: ["JPG یا PNG تصویر", "کیا تصویر کافی واضح ہے؟", "بیماری کا اسکور", "ڈاکٹر کے لیے اگلا قدم"],
      uploadTitle: "ریٹینا کی تصویر اپ لوڈ کریں",
      uploadHint: "کلینک کے کیمرے یا اپنی فائلوں سے فنڈس تصویر · JPG یا PNG · یہاں گھسیٹیں یا کلک کریں",
      uploadBtn: "تصویر منتخب کریں", replaceBtn: "تصویر تبدیل کریں", previewCap: "منتخب تصویر",
      examplesTitle: "یا کوئی مثال آزمائیں",
      examplesCredit: "مثالی تصاویر: DeepDRiD (Liu et al., Patterns 2022)، CC BY-SA 4.0۔ عنوان ماہرین کا درجہ اور تصویر کا معیار بتاتے ہیں، آلے کا جواب نہیں۔",
      exGrade: "ماہرین کا درجہ", exQuality: "تصویر کا معیار", good: "اچھا", poor: "کمزور",
      exMissed: "جان بوجھ کر دکھائی گئی: ایسی آنکھ جسے روشنی نہیں پکڑ پاتی",
      noteLangs: "مریض کے نوٹ کی زبانیں", checking: "ترجمہ زیرِ جانچ",
      status: {
        loading: "ماڈل اس ڈیوائس پر لوڈ ہو رہا ہے…", ready: "تیار۔ ریٹینا کی تصویر اپ لوڈ کریں یا کوئی مثال منتخب کریں۔",
        working: "اسی ڈیوائس پر تصویر کا معیار جانچا اور تجزیہ کیا جا رہا ہے…", done: "مکمل۔ تصویر اپ لوڈ نہیں کی گئی۔",
        errModel: "ماڈل لوڈ نہیں ہو سکا: ", errImage: "یہ تصویر نہیں کھل سکی۔ براہِ کرم JPG یا PNG ریٹینا تصویر منتخب کریں۔", errRun: "کچھ غلط ہو گیا: ",
      },
      offlineReady: "اس ڈیوائس پر محفوظ: اب انٹرنیٹ کے بغیر بھی کام کرے گا۔",
      resultTitle: "نتیجہ", empty: "شروع کرنے کے لیے تصویر اپ لوڈ کریں۔ نتیجہ یہاں ظاہر ہوگا۔",
      kicker2: "تجویز کردہ اگلا قدم · ڈاکٹر کی تصدیق کے لیے",
      R: {
        REFER: { tag: "ریفر", head: "ذیابیطس سے ریٹینا کی ایسی بیماری کا شبہ جس پر ریفر کرنا ضروری ہے", body: "درمیانے یا اس سے زیادہ درجے کی بیماری ہو سکتی ہے۔ معالج کا معائنہ ضروری ہے۔", next: "اگلا قدم: ڈاکٹر کی تصدیق کے بعد آنکھوں کے ماہر کے پاس بھیجیں۔" },
        RETAKE: { tag: "دوبارہ تصویر", head: "تصویر دوبارہ لینی ہوگی", body: "تصویر اتنی واضح نہیں کہ بیماری کو رد کیا جا سکے۔", next: "اگلا قدم: تصویر دوبارہ لیں۔ اگر پھر بھی واضح نہ ہو تو آنکھوں کے ماہر کے پاس بھیجیں۔" },
        RESCREEN: { tag: "دوبارہ اسکریننگ", head: "اس قابلِ جائزہ تصویر میں ریفر کے قابل بیماری نہیں ملی", body: "یہ اس بات کی ضمانت نہیں کہ آنکھ صحت مند ہے، اور آنکھ کی دوسری بیماریاں نہیں جانچی جاتیں۔", next: "اگلا قدم: ڈاکٹر کی ہدایت کے مطابق آنکھوں کے معمول کے معائنے جاری رکھیں (عام طور پر ہر 12 ماہ بعد)۔" },
      },
      scores: (r, rt, p, pt) => `بیماری کا اسکور ${r} (${rt} یا زیادہ پر ریفر) · کمزور تصویر کا اسکور ${p} (${pt} یا زیادہ پر دوبارہ تصویر)`,
      timing: s => `اس ڈیوائس پر ${s} سیکنڈ میں پراسیس ہوا۔`,
      noteTitle: "ڈاکٹر اور مریض کے لیے نوٹ",
      imagesTitle: "اصل تصویر اور ہیٹ میپ",
      photoCap: "تصویر جیسے ماڈل دیکھتا ہے", heatCap: "ریفر کے قابل بیماری کا ہیٹ میپ (Grad-CAM)",
      heatWhy: "ہیٹ میپ دکھاتا ہے کہ تصویر کے کن حصوں نے اسکور پر اثر ڈالا۔ یہ بیماری کے ہونے یا نہ ہونے کا ثبوت نہیں۔",
      gradesTitle: "اندازاً شدت (ماڈل کا نتیجہ)",
      grades: ["کوئی DR نہیں", "ہلکی", "درمیانی", "شدید", "پرولیفریٹو"],
      newBtn: "نیا جائزہ شروع کریں",
      evidenceTitle: "شواہد اور حدود",
      results: (a, b) => `بھارت کی ${a.n} نئی تصاویر پر: حساسیت ${a.se}%، مخصوصیت ${a.sp}%۔ چینی ہسپتالوں کی ${b.n} ایسی تصاویر پر جو ماڈل نے پہلے نہیں دیکھیں: حساسیت ${b.se}%، مخصوصیت ${b.sp}%۔`,
      thr: (t, s) => `ریفرل کی حد ${t}، جانچ سے پہلے طے کی گئی (ویلیڈیشن تصاویر پر ${s}% حساسیت کے لیے)۔`,
      model: "ماڈل: EfficientNet-B0، APTOS 2019 (بھارت) پر تربیت یافتہ، DeepDRiD (چین) پر جانچا گیا۔ پاکستانی مریضوں یا کم قیمت کیمروں پر نہیں جانچا گیا۔ میکولر اوڈیما کی نشاندہی نہیں کرتا۔",
      repo: "کوڈ، نتائج اور حدود",
      ctaStart: "جائزہ شروع کریں", ctaEvidence: "شواہد اور حدود",
      heroCap: "ریٹینا کی مثالی تصویر · DeepDRiD، CC BY-SA 4.0",
      h1t: "اسی ڈیوائس پر", h1d: "تصویر کا تجزیہ اسی براؤزر میں ہوتا ہے اور اسے اپ لوڈ نہیں کیا جاتا۔",
      h2t: "آف لائن تیار", h2d: "پہلی بار کھولنے کے بعد انٹرنیٹ کے بغیر کام کرتا ہے۔",
      h3t: "تین زبانیں", h3d: "انگریزی، اردو اور ازبک انٹرفیس؛ ازبک میں مریض کا نوٹ زیرِ جانچ ہے۔",
      h4t: "اوپن کوڈ", h4d: "کوڈ اور نتائج MIT لائسنس کے تحت عوامی ہیں۔",
      fine: "روشنی · VERITAS AI ہیکاتھون 2026 · تحقیقی نمونہ، طبی آلہ نہیں",
    },
    uz: {
      dir: "ltr", name: "Oʻzbekcha",
      brandSub: "Diabetik koʻz skriningi · tadqiqot prototipi",
      langLabel: "Til",
      chipLocal: "Shu brauzerda ishlaydi", chipPhoto: "Rasm yuklab yuborilmaydi", chipProto: "Tadqiqot prototipi",
      kicker: "Birlamchi tibbiy yordam · Pokiston",
      title: "Toʻr parda tasvirini baholash",
      lede: "Koʻz mutaxassisi boʻlmagan birlamchi diabet klinikasi uchun. Bitta toʻr parda (retina) suratini yuklang; Roshni shifokor tasdiqlashi uchun bitta keyingi qadamni taklif qiladi.",
      privacy: "Rasm shu brauzer ichida qayta ishlanadi va hech qayerga yuklanmaydi. Birinchi tashrifdan keyin vosita internetsiz ham ishlaydi.",
      disclaimer: "Talabalar xakatoni uchun tadqiqot prototipi. Tibbiy maslahat yoki tashxis emas va Pokistondagi bemorlarda sinovdan oʻtkazilmagan.",
      steps: ["Toʻr parda rasmini yuklash", "Rasm sifatini tekshirish", "Tahlil", "Natijani koʻrish"],
      stepHints: ["JPG yoki PNG fundus rasm", "Rasm yetarlicha aniqmi?", "Kasallik balli", "Shifokor uchun keyingi qadam"],
      uploadTitle: "Toʻr parda rasmini yuklang",
      uploadHint: "Klinika kamerasidan yoki fayllaringizdan fundus rasm · JPG yoki PNG · bu yerga torting yoki bosing",
      uploadBtn: "Rasm tanlash", replaceBtn: "Rasmni almashtirish", previewCap: "Tanlangan rasm",
      examplesTitle: "Yoki namunaviy rasmni sinab koʻring",
      examplesCredit: "Namunaviy rasmlar: DeepDRiD (Liu et al., Patterns 2022), CC BY-SA 4.0. Izohlar vosita javobini emas, mutaxassislar bahosi va rasm sifatini koʻrsatadi.",
      exGrade: "Mutaxassis bahosi", exQuality: "rasm sifati", good: "yaxshi", poor: "past",
      exMissed: "Ataylab koʻrsatilgan: Roshni oʻtkazib yuboradigan koʻzlardan biri",
      noteLangs: "Bemor eslatmasi tillari", checking: "tarjima tekshirilmoqda",
      status: {
        loading: "Model shu qurilmaga yuklanmoqda…", ready: "Tayyor. Toʻr parda rasmini yuklang yoki namunani tanlang.",
        working: "Shu qurilmada rasm sifati tekshirilmoqda va tahlil qilinmoqda…", done: "Tayyor. Rasm yuklab yuborilmadi.",
        errModel: "Modelni yuklab boʻlmadi: ", errImage: "Bu rasmni ochib boʻlmadi. JPG yoki PNG toʻr parda rasmini tanlang.", errRun: "Xatolik yuz berdi: ",
      },
      offlineReady: "Shu qurilmada saqlandi: endi internetsiz ham ishlaydi.",
      resultTitle: "Natija", empty: "Boshlash uchun rasm yuklang. Natija shu yerda chiqadi.",
      kicker2: "Tavsiya etilgan keyingi qadam · shifokor tasdiqlashi uchun",
      R: {
        REFER: { tag: "YUBORISH", head: "Yuborishni talab qiluvchi diabetik retinopatiya gumon qilinmoqda", body: "Oʻrtacha yoki undan ogʻir diabetik retinopatiya boʻlishi mumkin. Shifokor koʻrigi zarur.", next: "Keyingi qadam: shifokor tasdiqlagach, koʻz mutaxassisiga yuboring." },
        RETAKE: { tag: "QAYTA OLISH", head: "Rasmni qayta olish kerak", body: "Rasm kasallikni istisno qilish uchun yetarlicha aniq emas.", next: "Keyingi qadam: rasmni qayta oling. Hali ham noaniq boʻlsa, koʻz mutaxassisiga yuboring." },
        RESCREEN: { tag: "QAYTA SKRINING", head: "Baholash mumkin boʻlgan bu rasmda yuborishni talab qiluvchi diabetik retinopatiya aniqlanmadi", body: "Bu koʻz sogʻlom ekanining kafolati emas, boshqa koʻz kasalliklari tekshirilmaydi.", next: "Keyingi qadam: shifokor maslahatiga koʻra muntazam koʻz tekshiruvlarini davom ettiring (odatda har 12 oyda)." },
      },
      scores: (r, rt, p, pt) => `Kasallik balli ${r} (${rt} va undan yuqori boʻlsa yuborish) · Past sifat balli ${p} (${pt} va undan yuqori boʻlsa qayta olish)`,
      timing: s => `Shu qurilmada ${s} soniyada qayta ishlandi.`,
      noteTitle: "Shifokor va bemor uchun eslatma",
      imagesTitle: "Asl rasm va issiqlik xaritasi",
      photoCap: "Model koʻradigan rasm", heatCap: "Yuborishni talab qiluvchi kasallik uchun issiqlik xaritasi (Grad-CAM)",
      heatWhy: "Issiqlik xaritasi rasmning qaysi qismlari ballga taʼsir qilganini koʻrsatadi. Bu kasallik bor yoki yoʻqligining isboti emas.",
      gradesTitle: "Taxminiy ogʻirlik darajasi (model natijasi)",
      grades: ["DR yoʻq", "Yengil", "Oʻrtacha", "Ogʻir", "Proliferativ"],
      newBtn: "Yangi baholashni boshlash",
      evidenceTitle: "Dalillar va cheklovlar",
      results: (a, b) => `Hindistondan ${a.n} ta yangi rasmda: sezgirlik ${a.se}%, oʻziga xoslik ${a.sp}%. Model koʻrmagan Xitoy shifoxonalaridan ${b.n} ta rasmda: sezgirlik ${b.se}%, oʻziga xoslik ${b.sp}%.`,
      thr: (t, s) => `Yuborish chegarasi ${t}, sinovdan oldin belgilangan (validatsiya rasmlarida ${s}% sezgirlik uchun).`,
      model: "Model: EfficientNet-B0, APTOS 2019 (Hindiston) asosida oʻqitilgan, DeepDRiD (Xitoy) da sinovdan oʻtkazilgan. Pokistonlik bemorlarda yoki arzon kameralarda sinalmagan. Makula shishini aniqlamaydi.",
      repo: "Kod, natijalar va cheklovlar",
      ctaStart: "Baholashni boshlash", ctaEvidence: "Dalillar va cheklovlar",
      heroCap: "Namunaviy toʻr parda rasmi · DeepDRiD, CC BY-SA 4.0",
      h1t: "Qurilmaning oʻzida", h1d: "Rasm shu brauzerda tahlil qilinadi va yuklab yuborilmaydi.",
      h2t: "Oflayn ishlaydi", h2d: "Birinchi tashrifdan keyin internetsiz ishlaydi.",
      h3t: "Uch til", h3d: "Ingliz, urdu va oʻzbek tilidagi interfeys; oʻzbekcha bemor eslatmasi tekshirilmoqda.",
      h4t: "Ochiq kod", h4d: "Kod va natijalar MIT litsenziyasi ostida ochiq.",
      fine: "Roshni · VERITAS AI Hackathon 2026 · Tadqiqot prototipi, tibbiy qurilma emas",
    },
  };
  /* Uzbek interface and patient note: drafts until Laziza (native speaker) approves them.
     Our rule: no language is shown until a native speaker has checked it. Set to true once she replies "approved". */
  const SHOW_UZBEK = false;
  if (!SHOW_UZBEK) {
    delete UI.uz;
    UI.en.h3t = "English and Urdu"; UI.en.h3d = "Interface and patient note in English and Urdu; the Urdu was checked by a native speaker.";
    UI.ur.h3t = "انگریزی اور اردو"; UI.ur.h3d = "انٹرفیس اور مریض کا نوٹ انگریزی اور اردو میں؛ اردو کی جانچ اہلِ زبان نے کی ہے۔";
  }
  /* Uzbek patient note (draft; not in roshni_web.json yet). */
  const NOTE_UZ = {
    REFER: "Yuborishni talab qiluvchi diabetik retinopatiya gumon qilinmoqda (oʻrtacha yoki ogʻirroq). Shifokor tasdiqlashi uchun tavsiya etilgan keyingi qadam: koʻz mutaxassisiga yuborish.",
    RETAKE: "Rasm kasallikni istisno qilish uchun yetarlicha aniq emas. Tavsiya etilgan keyingi qadam: rasmni qayta oling; hali ham noaniq boʻlsa, koʻz mutaxassisiga yuboring.",
    RESCREEN: "Bu rasmda yuborishni talab qiluvchi diabetik retinopatiya aniqlanmadi; bu koʻz kasalligini istisno qilmaydi. Tavsiya etilgan keyingi qadam: shifokor maslahatiga koʻra muntazam koʻz tekshiruvlarini davom ettiring (odatda har 12 oyda).",
    DISCLAIMER: "Talabalar xakatoni uchun prototip. Tibbiy maslahat yoki tashxis emas.",
  };

  let cfg = null, session = null, lastDecision = null, lastScores = null, statusKey = "loading", statusExtra = "", offlineOK = false;
  let runSeq = 0, objURL = null;   // runSeq: show only the latest photo's result; objURL: the chosen file, released when replaced
  let lang = "en";
  try { const s = localStorage.getItem("roshni-lang"); if (s && UI[s]) lang = s; } catch (e) { /* storage unavailable */ }
  const $ = id => document.getElementById(id);
  const T = () => UI[lang];

  function setStage(s) { document.body.dataset.stage = s; }
  function setStatus(key, extra) {
    statusKey = key; statusExtra = extra || "";
    $("status").textContent = STATUS_EN[key] + statusExtra;   // English, read by tests and the video script
    $("statusText").textContent = T().status[key] + statusExtra;
    $("statusText").dataset.kind = key;
  }

  /* ---------- interface language ---------- */
  function applyLang() {
    const t = T();
    document.documentElement.lang = lang;
    document.documentElement.dir = t.dir;
    document.querySelectorAll("[data-i18n]").forEach(el => { el.textContent = t[el.dataset.i18n]; });
    document.querySelectorAll("[data-step]").forEach(el => {
      const i = +el.dataset.step;
      el.querySelector("b").textContent = t.steps[i];
      el.querySelector("small").textContent = t.stepHints[i];
    });
    document.querySelectorAll("#uilang button").forEach(b => b.setAttribute("aria-pressed", b.dataset.lang === lang));
    $("statusText").textContent = t.status[statusKey] + statusExtra;
    if (offlineOK) $("offline").textContent = t.offlineReady;
    if (cfg) { buildFooter(); translateExamples(); relabelNoteLangs(); }
    if (lastDecision) { renderResultText(lastDecision); renderNote(lastDecision); }
    if (window.roshniLast) renderGrades(window.roshniLast.P);
  }
  function buildUILang() {
    const box = $("uilang");
    for (const code of Object.keys(UI)) {
      const b = document.createElement("button");
      b.type = "button"; b.dataset.lang = code; b.textContent = UI[code].name;
      b.lang = code;
      b.addEventListener("click", () => {
        lang = code;
        try { localStorage.setItem("roshni-lang", code); } catch (e) { /* ignore */ }
        applyLang();
      });
      box.appendChild(b);
    }
  }

  async function init() {
    buildUILang();
    applyLang();
    setStage("idle");
    try {
      setStatus("loading");
      cfg = await (await fetch("roshni_web.json")).json();
      // Uzbek patient note: a draft that the page marks "translation being checked"
      if (SHOW_UZBEK && !cfg.langs.uz) {
        cfg.langs.uz = { name: "Oʻzbekcha", dir: "ltr", checked: false };
        for (const k of Object.keys(NOTE_UZ)) if (cfg.report[k]) cfg.report[k].uz = NOTE_UZ[k];
      }
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create("model.onnx", { executionProviders: ["wasm"] });
      buildLanguagePicker();
      buildExamples();
      buildFooter();
      setStatus("ready");
      $("file").disabled = false;
      enableOffline();
    } catch (e) {
      setStatus("errModel", e.message);
      console.error(e);
    }
  }

  /* Keep the page, model and runtime on this device so screening works with no internet next time. */
  async function enableOffline() {
    if (!("serviceWorker" in navigator) || !window.caches) return;
    try {
      await navigator.serviceWorker.register("sw.js");
      await navigator.serviceWorker.ready;
      for (let i = 0; i < 60; i++) {   // the first visit downloads about 30 MB into the cache
        if (await caches.match("model.onnx") && await caches.match("ort/ort-wasm-simd-threaded.wasm")) {
          offlineOK = true;
          $("offline").textContent = T().offlineReady;
          window.roshniOfflineReady = true;
          return;
        }
        await new Promise(r => setTimeout(r, 1000));
      }
    } catch (e) {
      console.warn("offline mode unavailable", e);
    }
  }

  function buildLanguagePicker() {
    const box = $("langs");
    for (const [code, v] of Object.entries(cfg.langs)) {
      const lab = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.value = code;
      cb.checked = cfg.default_langs.includes(code);
      cb.addEventListener("change", () => { if (lastDecision) renderNote(lastDecision); });
      lab.appendChild(cb);
      const span = document.createElement("span");
      span.dataset.code = code;
      lab.appendChild(span);
      box.appendChild(lab);
    }
    relabelNoteLangs();
  }
  function relabelNoteLangs() {
    document.querySelectorAll("#langs span[data-code]").forEach(s => {
      const v = cfg.langs[s.dataset.code];
      s.textContent = v.name + (v.checked ? "" : ` (${T().checking})`);
    });
  }

  function captionFor(c) {
    const t = T();
    const m = /^Expert grade (\d) \((.+?)\), photo quality (good|poor)(\. Shown on purpose.*)?$/.exec(c || "");
    if (!m || lang === "en") return c;
    return `${t.exGrade} ${m[1]} (${t.grades[+m[1]]})، ${t.exQuality} ${t[m[3]]}${m[4] ? ". " + t.exMissed : ""}`.replace("،", lang === "ur" ? "،" : ",");
  }
  function buildExamples() {
    const box = $("examples");
    for (const ex of cfg.examples || []) {
      const b = document.createElement("button");
      b.className = "ex";
      b.type = "button";
      b.dataset.caption = ex.caption;
      const img = document.createElement("img");
      img.src = ex.file;
      img.alt = ex.caption;
      b.appendChild(img);
      const cap = document.createElement("span");
      b.appendChild(cap);
      b.addEventListener("click", () => { document.querySelectorAll(".ex").forEach(x => x.classList.remove("on")); b.classList.add("on"); loadURL(ex.file); });
      box.appendChild(b);
    }
    translateExamples();
  }
  function translateExamples() {
    document.querySelectorAll(".ex").forEach(b => {
      const c = captionFor(b.dataset.caption);
      b.title = c;
      b.querySelector("span").textContent = c;
    });
    $("excredit").textContent = cfg.examples_credit ? T().examplesCredit : "";
  }

  function buildFooter() {
    const s = cfg.summary || {}, t = T();
    const f = (x) => ({ n: x.n.toLocaleString("en-US"), se: (100 * x.sensitivity).toFixed(1), sp: (100 * x.specificity).toFixed(1) });
    $("results").textContent = (s.fine_tuned && s.external_all) ? t.results(f(s.fine_tuned), f(s.external_all)) : "";
    $("thr").textContent = t.thr(cfg.ref_threshold.toFixed(3), (100 * cfg.target_sensitivity).toFixed(0));
    if (cfg.repo_url) { $("repo").href = cfg.repo_url; $("repo").hidden = false; }
  }

  /* ---------- choosing a photo ---------- */
  function showPreview(url) {
    $("preview").src = url;
    document.body.classList.add("has-image");
  }
  function loadURL(url) {
    const img = new Image();
    img.onload = () => { showPreview(url); run(img); };
    img.onerror = () => { setStage("idle"); setStatus("errImage"); };
    img.src = url;
  }

  function fileURL(f) {
    if (objURL) URL.revokeObjectURL(objURL);
    objURL = URL.createObjectURL(f);
    return objURL;
  }
  $("file").addEventListener("change", e => {
    const f = e.target.files[0];
    if (!f) return;
    document.querySelectorAll(".ex").forEach(x => x.classList.remove("on"));
    loadURL(fileURL(f));
    e.target.value = "";   // choosing the same file again still runs
  });
  const drop = $("drop");
  ["dragenter", "dragover"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); if (!$("file").disabled) drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", e => {
    if ($("file").disabled) return;
    const f = e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) { document.querySelectorAll(".ex").forEach(x => x.classList.remove("on")); loadURL(fileURL(f)); }
  });

  $("newBtn").addEventListener("click", () => {
    lastDecision = null; window.roshniLast = undefined;
    ["card", "notebox", "heatbox"].forEach(id => { $(id).hidden = true; });
    $("file").value = "";
    $("preview").removeAttribute("src");
    if (objURL) { URL.revokeObjectURL(objURL); objURL = null; }
    document.body.classList.remove("has-image");
    document.querySelectorAll(".ex").forEach(x => x.classList.remove("on"));
    setStage("idle");
    setStatus("ready");
    $("tool").scrollIntoView({ behavior: "smooth", block: "start" });
  });

  /* ---------- screening (unchanged) ---------- */
  function prepare(img) {
    const S = cfg.size;
    const c0 = document.createElement("canvas");
    c0.width = img.naturalWidth; c0.height = img.naturalHeight;
    const x0 = c0.getContext("2d", { willReadFrequently: true });
    x0.drawImage(img, 0, 0);
    const box = R.cropBox(x0.getImageData(0, 0, c0.width, c0.height).data, c0.width, c0.height, cfg.crop_tol);
    const side = Math.max(box.w, box.h);
    const c1 = document.createElement("canvas");
    c1.width = side; c1.height = side;
    const x1 = c1.getContext("2d", { willReadFrequently: true });
    x1.fillStyle = "#000"; x1.fillRect(0, 0, side, side);
    x1.drawImage(c0, box.x, box.y, box.w, box.h, Math.floor((side - box.w) / 2), Math.floor((side - box.h) / 2), box.w, box.h);
    const src = x1.getImageData(0, 0, side, side).data;
    const px = side === S ? src : R.resizePIL(src, side, side, S, S);
    const c2 = $("photo");
    c2.width = S; c2.height = S;
    c2.getContext("2d").putImageData(new ImageData(new Uint8ClampedArray(px), S, S), 0, 0);
    const t = new Float32Array(3 * S * S);
    for (let i = 0; i < S * S; i++) {
      for (let ch = 0; ch < 3; ch++) t[ch * S * S + i] = (px[i * 4 + ch] / 255 - cfg.mean[ch]) / cfg.std[ch];
    }
    return { tensor: new ort.Tensor("float32", t, [1, 3, S, S]), pixels: px };
  }

  async function run(img) {
    const my = ++runSeq;   // if photos are chosen quickly one after another, show only the last one's result
    try {
      setStage("working");
      setStatus("working");
      await new Promise(r => requestAnimationFrame(() => setTimeout(r, 0)));   // let the stage indicator paint first
      const t0 = performance.now();
      const { tensor, pixels } = prepare(img);
      const out = await session.run({ image: tensor });
      if (my !== runSeq) return;
      const P = R.softmax(Array.from(out.logits.data));
      const emb = out.features.data;
      const pRef = R.referableProb(P, cfg.referable_from);
      const pPoor = R.poorProb(emb, cfg.gate);
      const d = R.decide(pRef, pPoor, cfg.ref_threshold, cfg.poor_threshold);
      const ms = performance.now() - t0;
      window.roshniLast = { P: P, features: Array.from(emb) };   // for automated checks
      const [, C, H, W] = out.feature_map.dims;
      const cam = R.heatmap(out.feature_map.data, C, H, W, P, cfg.head_weight, cfg.referable_from);
      drawHeat(cam, H, W, pixels);
      renderCard(d, pRef, pPoor, ms);
      renderGrades(P);
      renderNote(d);
      lastDecision = d;
      setStage("done");
      setStatus("done");
      // On a phone the result sits below the photo list: bring it into view so the answer is not missed.
      const card = $("card");
      if (card && card.getBoundingClientRect().top > window.innerHeight * 0.6) card.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (e) {
      if (my !== runSeq) return;
      setStage("idle");
      setStatus("errRun", e.message);
      console.error(e);
    }
  }

  function drawHeat(cam, H, W, pixels) {
    const S = cfg.size;
    const small = document.createElement("canvas");
    small.width = W; small.height = H;
    const xs = small.getContext("2d");
    const im = xs.createImageData(W, H);
    for (let i = 0; i < H * W; i++) { const v = Math.round(cam[i] * 255); im.data.set([v, v, v, 255], i * 4); }
    xs.putImageData(im, 0, 0);
    const big = document.createElement("canvas");
    big.width = S; big.height = S;
    const xb = big.getContext("2d", { willReadFrequently: true });
    xb.imageSmoothingEnabled = true;
    xb.drawImage(small, 0, 0, S, S);
    const h = xb.getImageData(0, 0, S, S).data;
    const c = $("heat");
    c.width = S; c.height = S;
    const xc = c.getContext("2d");
    const o = xc.createImageData(S, S);
    for (let i = 0; i < S * S; i++) {
      const a = 0.45 * h[i * 4] / 255;
      o.data[i * 4] = pixels[i * 4] * (1 - a) + 255 * a;
      o.data[i * 4 + 1] = pixels[i * 4 + 1] * (1 - a) + 40 * a;
      o.data[i * 4 + 2] = pixels[i * 4 + 2] * (1 - a);
      o.data[i * 4 + 3] = 255;
    }
    xc.putImageData(o, 0, 0);
    $("heatbox").hidden = false;
  }

  /* ---------- showing the result ---------- */
  function renderCard(d, pRef, pPoor, ms) {
    const card = $("card");
    card.hidden = false;
    card.dataset.decision = d;
    // English originals (hidden), unchanged, for the tests
    $("decision").textContent = TITLES[d];
    $("decision").style.color = COLORS[d];
    const poor = cfg.gate ? `${pPoor.toFixed(2)} (retake at ${cfg.poor_threshold.toFixed(2)} or above)` : "not available";
    $("scores").textContent = `Referable-disease score ${pRef.toFixed(2)} (refer at ${cfg.ref_threshold.toFixed(2)} or above) · Poor-photo score ${poor}`;
    $("timing").textContent = `Screened on this device in ${(ms / 1000).toFixed(2)} s.`;
    lastScores = { r: pRef.toFixed(2), rt: cfg.ref_threshold.toFixed(2), p: cfg.gate ? pPoor.toFixed(2) : "–", pt: cfg.poor_threshold.toFixed(2), s: (ms / 1000).toFixed(2) };
    renderResultText(d);
  }
  function renderResultText(d) {
    const t = T(), r = t.R[d];
    $("rTag").textContent = r.tag;
    $("rHead").textContent = r.head;
    $("rBody").textContent = r.body;
    $("rNext").textContent = r.next;
    if (lastScores) {
      $("rScores").textContent = t.scores(lastScores.r, lastScores.rt, lastScores.p, lastScores.pt);
      $("rTiming").textContent = t.timing(lastScores.s);
    }
  }

  function renderGrades(P) {
    const box = $("grades");
    box.replaceChildren();
    P.forEach((p, k) => {
      const row = document.createElement("div");
      row.className = "g";
      const lab = document.createElement("span");
      lab.textContent = `${k} ${T().grades[k] || cfg.grades[k]}`;
      const bar = document.createElement("span");
      bar.className = "bar";
      const fill = document.createElement("span");
      fill.style.width = `${(100 * p).toFixed(1)}%`;
      bar.appendChild(fill);
      const val = document.createElement("span");
      val.className = "pct";
      val.textContent = `${(100 * p).toFixed(0)}%`;
      row.append(lab, bar, val);
      box.appendChild(row);
    });
  }

  function renderNote(d) {
    const chosen = Array.from(document.querySelectorAll("#langs input:checked")).map(x => x.value);
    const langs = chosen.length ? chosen : ["en"];
    const box = $("note");
    box.replaceChildren();
    for (const code of Object.keys(cfg.langs)) {
      if (!langs.includes(code)) continue;
      if (!cfg.report[d][code]) continue;
      const div = document.createElement("div");
      div.dir = cfg.langs[code].dir;
      div.lang = code;
      div.className = "n";
      const b = document.createElement("b");
      b.textContent = cfg.langs[code].name + (cfg.langs[code].checked ? "" : ` (${T().checking})`) + ": ";
      div.appendChild(b);
      div.appendChild(document.createTextNode(cfg.report[d][code]));
      const small = document.createElement("small");
      small.textContent = cfg.report.DISCLAIMER[code];
      div.appendChild(document.createElement("br"));
      div.appendChild(small);
      box.appendChild(div);
    }
    $("notebox").hidden = false;
  }

  init();
}
