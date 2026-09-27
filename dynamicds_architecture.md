# ⚡ AutoDS / DynamicDS — Complete End-to-End Architecture

> Every function, every regex, every cache, every disk write. Full signal path from browser click to JSON response.

---

## 🌐 ZONE 1 — Frontend Entry Points

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🌐  BROWSER / FRONTEND CLIENT                        ║
║                                                                              ║
║   [Upload CSV/XLSX]   [View EDA]   [Train Model]   [Predict]   [Download]  ║
╚══════════════════════╦═══════════╦══════════════╦═══════════╦═══════════════╝
                       ║           ║              ║           ║
            POST /upload  GET /eda  POST /train  POST /predict GET /report
```

---

## 🚪 ZONE 2 — FastAPI Application Entry (`main.py`)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         app/main.py  — ASGI Bootstrap                       ║
║                                                                              ║
║  app = FastAPI(title="Dynamic DS", version="1.0.0")                         ║
║  app.include_router(datasets.router)    ← prefix: /api/datasets              ║
║  app.include_router(experiments.router) ← prefix: /api                       ║
║                                                                              ║
║  Endpoints in main.py:                                                       ║
║  ┌─────────────────────────────────────────────────────────┐                 ║
║  │ GET /api/health  → {status:"ok", max_upload_mb, ...}    │                 ║
║  │ GET /api/config  → {model_statuses, thresholds:{...}}   │                 ║
║  │   uses: C.MODEL_STATUSES, C.SKEW_*, C.CORR_*, C.MISSING_*│               ║
║  │ GET /           → FileResponse(frontend/dist/index.html) │                 ║
║  │   or fallback   → JSONResponse({message: "See /docs"})  │                 ║
║  └─────────────────────────────────────────────────────────┘                 ║
║  StaticFiles: /assets → frontend/dist/assets (if built)                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📁 ZONE 3A — Dataset Upload Flow (`datasets.py` → `storage.py` → `profiling.py`)

```
POST /api/datasets/upload  (multipart/form-data)
│
▼
╔══════════════════════════════════════════════════════════════════════════════╗
║           app/api/datasets.py  ::  upload(file: UploadFile)                  ║
║                                                                              ║
║  1. Stream bytes in 1MB chunks until EOF                                     ║
║     while chunk := await file.read(1024 * 1024):                            ║
║         data.extend(chunk)                                                   ║
║         if len(data) > C.MAX_UPLOAD_MB * 1024 * 1024:                      ║
║             raise HTTPException(413, "File exceeds limit")                   ║
╚══════════════════╦═══════════════════════════════════════════════════════════╝
                   ║
                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║           app/services/storage.py  ::  parse_upload(filename, data)          ║
║                                                                              ║
║  SECURITY CHECKS:                                                            ║
║  ├─ ext = Path(filename).suffix.lower()                                      ║
║  ├─ if ext not in {".csv", ".xlsx"} → UploadError                           ║
║  ├─ if len(data) == 0 → UploadError                                         ║
║  └─ if len(data) > MAX_UPLOAD_MB * 1024²  → UploadError                    ║
║                                                                              ║
║  CSV PATH:                                                                   ║
║  ├─ head = data[:20000]  ← first 20KB for detection                         ║
║  ├─ Encoding Detection:                                                      ║
║  │     enc = "utf-8-sig"                                                     ║
║  │     try: head.decode(enc)                                                 ║
║  │     except UnicodeDecodeError: enc = "latin-1"                           ║
║  ├─ Delimiter Detection:                                                     ║
║  │     line = head.decode(enc).splitlines()[0]                               ║
║  │     sep = max([",", ";", "\t", "|"], key=line.count)                     ║
║  └─ df = pd.read_csv(BytesIO(data), sep=sep, encoding=enc,                  ║
║                       dtype=str, keep_default_na=False, na_values=[""])      ║
║                                                                              ║
║  XLSX PATH:                                                                  ║
║  ├─ Magic Byte Check: if not data.startswith(b"PK"):                        ║
║  │     raise UploadError("Not a valid .xlsx workbook")                       ║
║  └─ df = pd.read_excel(BytesIO(data), dtype=str, engine="openpyxl")         ║
║                                                                              ║
║  POST-PARSE GUARDS:                                                          ║
║  ├─ if df.empty or df.shape[1] == 0 → UploadError                          ║
║  ├─ if len(df) > C.MAX_ROWS (500,000) → UploadError                        ║
║  ├─ if df.shape[1] > C.MAX_COLS (300) → UploadError                        ║
║  └─ df.columns = [str(c).strip() or f"unnamed_{i}" for i, c in enumerate]  ║
║                                                                              ║
║  returns: raw_df (all-string DataFrame)                                      ║
╚══════════════════╦═══════════════════════════════════════════════════════════╝
                   ║
                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║           app/services/profiling.py  ::  build_profile(raw, filename, size)  ║
║                                                                              ║
║  Step 1: normalize_placeholders(raw)    ← data_quality.py                   ║
║  Step 2: infer_types(norm)              ← type_inference.py                  ║
║  Step 3: coerce_types(norm, meta)       ← type_inference.py                  ║
║  Step 4: build_quality(typed, norm, meta, placeholders)  ← data_quality.py  ║
║  Step 5: Build overview dict {rows, columns, type_counts, memory_mb}        ║
║                                                                              ║
║  returns: (typed_df, norm_df, profile_dict, quality_dict)                   ║
╚══════════════════╦═══════════════════════════════════════════════════════════╝
                   ║
         ┌─────────┴──────────┐
         ▼                    ▼
╔═════════════════╗  ╔══════════════════════════════════════════════════════╗
║ TYPE INFERENCE  ║  ║           DATA QUALITY ENGINE                       ║
║ type_inference  ║  ║           app/services/data_quality.py               ║
║                 ║  ║                                                      ║
║ infer_column(): ║  ║  normalize_placeholders():                           ║
║ Regex patterns: ║  ║  ├─ key = nn.str.strip().str.lower()                 ║
║ _FORMATTED =    ║  ║  ├─ mask = key.isin(C.PLACEHOLDER_TOKENS)            ║
║  r"^[-+]?[$€£]? ║  ║  │   tokens: {"nan","null","none","na","n/a",       ║
║  [\d{1,3}(,\d{3}║  ║  │            "unknown","?","??","-","--","nil"...} ║
║  )+|\d+]..."    ║  ║  └─ out.loc[mask, c] = np.nan                       ║
║                 ║  ║                                                      ║
║ DT_FORMATS:     ║  ║  analyze_missing():                                  ║
║  %Y-%m-%d       ║  ║  ├─ missing_level(pct):                             ║
║  %d/%m/%Y       ║  ║  │   0%       → "none"                              ║
║  %m/%d/%Y       ║  ║  │   < 5%    → "low"    (C.MISSING_LOW)            ║
║  %d.%m.%Y       ║  ║  │   < 20%   → "moderate" (C.MISSING_MODERATE)     ║
║  %b %d, %Y      ║  ║  │   < 80%   → "high"   (C.MISSING_HIGH)           ║
║  (12 formats)   ║  ║  │   >= 80%  → "very_high"                         ║
║                 ║  ║  └─ flagged rows: missing / total >= C.ROW_MISSING_FLAG║
║ _AMBIGUOUS:     ║  ║                                                      ║
║  {%d/%m, %m/%d} ║  ║  analyze_constants():                               ║
║                 ║  ║  ├─ single unique value → "constant"                ║
║ Semantic Types: ║  ║  └─ top value >= C.NEAR_CONSTANT_SHARE (95%) → "near_constant"║
║  boolean   ─── ║  ║                                                      ║
║  numerical ─── ║  ║  analyze_formatting():                               ║
║  datetime  ─── ║  ║  ├─ _EMAIL = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"         ║
║  text      ─── ║  ║  ├─ _URL   = r"^(https?://|www\.)\S+$"             ║
║  identifier─── ║  ║  ├─ _PHONE = r"^\+?[\d\s\-().]{7,20}$"             ║
║  categorical─  ║  ║  └─ sample capped at C.VIS_SAMPLE_ROWS (50k rows)   ║
╚═════════════════╝  ╚══════════════════════════════════════════════════════╝
         ║                    ║
         └─────────┬──────────┘
                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║           app/services/storage.py  ::  save_dataset(filename, data, ...)     ║
║                                                                              ║
║  did = uuid.uuid4().hex[:12]           e.g. "a1b2c3d4e5f6"                  ║
║  dataset_dir(did):                                                           ║
║    _ID_RE = r"^[a-f0-9]{12}$"          ← validates did before path build    ║
║    → DATA_DIR / "datasets" / did                                             ║
║                                                                              ║
║  safe_filename(name):                                                        ║
║    base = Path(name).name              ← strips directory traversal          ║
║    re.sub(r"[^\w.\- ]", "_", base)[:120]  ← replaces illegal chars         ║
║                                                                              ║
║  Writes to disk:                                                             ║
║  ├─ data/datasets/{did}/data.csv       ← raw.to_csv(index=False)            ║
║  └─ data/datasets/{did}/meta.json  ←────────────────────┐                   ║
║       {                                                   │                  ║
║         id, filename, uploaded_at (UTC ISO),             │                  ║
║         size_bytes, sha256 (hashlib.sha256(data).hexdigest()),│             ║
║         profile: {overview, columns:[...]},              │                  ║
║         quality: {missing, duplicates, constants, ...}   │                  ║
║       }                                                   │                  ║
╚══════════════════════════════════════════════════════════╪══════════════════╝
                                                           │
                   ┌───────────────────────────────────────┘
                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║  HTTP 200 Response  →  { id, filename, profile }                             ║
║  Wrapped by: to_jsonable()  (converts NaN/inf→None, np.int64→int,            ║
║              np.float64→float, Timestamp→ISO string, pd.NaT→None)           ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 📊 ZONE 3B — EDA & Warnings Flow (`datasets.py` → `eda.py` → `warnings_engine.py`)

```
GET /api/datasets/{id}/eda?target=price&problem_type=regression
│
▼
╔══════════════════════════════════════════════════════════════════════════════╗
║           app/api/datasets.py  ::  _eda(dataset_id, target, problem_type)    ║
║                                                                              ║
║  LRU CACHE (in-process dict, max 20 entries):                               ║
║  key = (dataset_id, target, problem_type)                                    ║
║  if key not in _eda_cache:                                                   ║
║      if len(_eda_cache) > 20:                                                ║
║          _eda_cache.pop(next(iter(_eda_cache)))  ← evict oldest entry        ║
║      _eda_cache[key] = to_jsonable(eda_svc.build_eda(...))                   ║
║  return _eda_cache[key]                                                       ║
╚══════════════════╦═══════════════════════════════════════════════════════════╝
                   ║ (cache miss)
                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║           app/services/storage.py  ::  load_typed(dataset_id)                ║
║                                                                              ║
║  THREAD-SAFE LRU (OrderedDict, max 4 DataFrames):                           ║
║  with _lock:                                                                 ║
║      if dataset_id in _cache:                                                ║
║          _cache.move_to_end(dataset_id)   ← mark MRU                        ║
║          return _cache[dataset_id]         ← instant return                  ║
║                                                                              ║
║  [Lock released] ← heavy ops run unlocked to not block other threads         ║
║  raw = pd.read_csv("data.csv", dtype=str, na_values=[""])                   ║
║  norm, _ = normalize_placeholders(raw)                                       ║
║  typed = coerce_types(norm, meta["profile"]["columns"])                      ║
║  [Lock re-acquired] → _cache[dataset_id] = typed                            ║
║  while len(_cache) > 4: _cache.popitem(last=False)  ← evict LRU            ║
╚══════════════════╦═══════════════════════════════════════════════════════════╝
                   ║
                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║           app/services/eda.py  ::  build_eda(df, profile, target, pt)        ║
║                                                                              ║
║  meta = {m["name"]: m for m in profile["columns"]}                          ║
║  by = lambda t: [c for c, m in meta.items() if m["semantic_type"] == t]     ║
║                                                                              ║
║  ┌─── NUMERIC COLUMNS ──────────────────────────────────────────────────┐   ║
║  │ numeric_stats(s, n_rows):                                             │   ║
║  │  ├─ Quantiles: q1, median, q3 = x.quantile([0.25, 0.5, 0.75])       │   ║
║  │  ├─ IQR Outliers: lo = Q1 - C.OUTLIER_IQR_K*IQR (1.5×IQR)          │   ║
║  │  │                hi = Q3 + C.OUTLIER_IQR_K*IQR                     │   ║
║  │  ├─ Skewness: skew_label() uses C.SKEW_SYMMETRIC(0.5), C.SKEW_STRONG(1.0)│
║  │  ├─ Histogram: bins = min(30, max(5, sqrt(cnt)))                     │   ║
║  │  │             np.histogram(x, bins=bins)                            │   ║
║  │  ├─ KDE: if std>0 & unique>10: scipy.stats.gaussian_kde on 80-point grid│
║  │  │       sample capped at C.VIS_SAMPLE_ROWS (50k)                   │   ║
║  │  └─ Normality: scipy.stats.shapiro()                                 │   ║
║  │                sample capped at C.SHAPIRO_MAX (50k)                  │   ║
║  │                p<0.05 → "Normality rejected at 5% level"             │   ║
║  └───────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─── CATEGORICAL COLUMNS ──────────────────────────────────────────────┐   ║
║  │ categorical_stats(s, n_rows):                                         │   ║
║  │  ├─ cardinality: <=C.CARD_LOW(10)→"low", <=C.CARD_HIGH(50)→"medium"  │   ║
║  │  ├─ rare categories: share < C.RARE_CATEGORY_SHARE (0.01 = 1%)       │   ║
║  │  └─ top 20 values with counts and percentages                         │   ║
║  └───────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─── DATETIME COLUMNS ─────────────────────────────────────────────────┐   ║
║  │ datetime_stats(s, n_rows):                                            │   ║
║  │  └─ calls timeutils.infer_frequency(ts):                              │   ║
║  │      u = DatetimeIndex(sorted unique timestamps)                      │   ║
║  │      freq = pd.infer_freq(u)   ← try pandas built-in                  │   ║
║  │      fallback: median_days = median(u[i+1]-u[i])                     │   ║
║  │        27-32 days → "MS", 88-93 days → "QS", 360-370 days → "YS"    │   ║
║  │      normalise_alias(): "M"→"MS", "Q"→"QS", "A"→"YS", "Y"→"YS"     │   ║
║  │      regularize_index(): snap to period start                         │   ║
║  │        freq in ("MS","QS","YS"): ts.dt.to_period(p).dt.to_timestamp() │   ║
║  │      gaps = len(full_grid) - idx.nunique()                            │   ║
║  └───────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─── TEXT COLUMNS ─────────────────────────────────────────────────────┐   ║
║  │ text_stats(s, n_rows):                                                │   ║
║  │  ├─ sample capped at 20,000 rows                                      │   ║
║  │  ├─ regex: r"[a-z']{3,}"  ← extract words >= 3 chars                 │   ║
║  │  ├─ _STOP = {the, and, for, with, ...}  ← 30 stop words filtered out │   ║
║  │  └─ Counter.most_common(15) → top token frequencies                   │   ║
║  └───────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─── CORRELATION ANALYSIS ─────────────────────────────────────────────┐   ║
║  │ correlation_analysis(df, num_cols, meta):                             │   ║
║  │  ├─ Pearson: df[cols].corr()   (up to C.CORR_MAX_COLS=40 columns)    │   ║
║  │  ├─ Spearman: df.corr(method="spearman") (sampled ≤200k rows)        │   ║
║  │  ├─ High pairs: |r| >= C.CORR_HIGH (0.7) → "multicollinearity"       │   ║
║  │  └─ Cramér's V (categorical pairs):                                   │   ║
║  │       ct = pd.crosstab(a, b)                                          │   ║
║  │       chi2 = scipy.stats.chi2_contingency(ct, correction=False)[0]   │   ║
║  │       phi2c = max(0, chi2/n - (k-1)(r-1)/(n-1))   ← bias correction  │   ║
║  │       V = sqrt(phi2c / min(kc-1, rc-1))                               │   ║
║  └───────────────────────────────────────────────────────────────────────┘   ║
║                                                                              ║
║  ┌─── TARGET ANALYSIS (if target provided) ─────────────────────────────┐   ║
║  │ analyze_target(df, meta, target, pt):                                 │   ║
║  │  ├─ recommend_problem_type(): boolean/cat→classification               │   ║
║  │  │   numerical with 2 unique→classification, else regression           │   ║
║  │  ├─ Association measures per feature:                                  │   ║
║  │  │   num vs regression  → Pearson r, Spearman rs                       │   ║
║  │  │   num vs classification → correlation_ratio(cat, x): eta            │   ║
║  │  │      eta = sqrt( SSB / SST )  (between/total variance ratio)        │   ║
║  │  │   cat vs classification → cramers_v(a, b)                           │   ║
║  │  ├─ leakage_suspects: association >= C.LEAKAGE_ASSOC (0.95)           │   ║
║  │  │   or identical values > 99% of rows                                 │   ║
║  │  └─ sample capped at C.ASSOC_SAMPLE_ROWS (100k rows)                  │   ║
║  └───────────────────────────────────────────────────────────────────────┘   ║
╚══════════════════╦═══════════════════════════════════════════════════════════╝
                   ║
                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║       app/services/warnings_engine.py  ::  build_warnings(profile,quality,eda)║
║                                                                              ║
║  SEVERITY ORDERING: HIGH(0) > WARNING(1) > INFO(2)                          ║
║  _w(sev, col, issue, evidence, recommendation, category)                     ║
║                                                                              ║
║  ─── HIGH ──────────────────────────────────────────────────────────────    ║
║  • Target missing > 20%                                                      ║
║  • Leakage suspect (association >= C.LEAKAGE_ASSOC)                          ║
║                                                                              ║
║  ─── WARNING ───────────────────────────────────────────────────────────    ║
║  • rows < 200                   "Small dataset"                              ║
║  • column missingness level in ("high","very_high")                          ║
║  • flagged_row_count > 0        (>= ROW_MISSING_FLAG fraction missing)       ║
║  • exact_duplicates > 0         or excluding_strong_identifiers > 0         ║
║  • identifier columns detected                                               ║
║  • high_cardinality categorical (> 50 unique)                                ║
║  • near-constant columns (>= 95% same value)                                 ║
║  • |pearson| >= C.CORR_WARN (0.9)  → multicollinearity                      ║
║  • imbalanced target (minority < C.IMBALANCE_MINORITY_SHARE = 20%)           ║
║                                                                              ║
║  ─── INFO ──────────────────────────────────────────────────────────────    ║
║  • placeholder tokens converted to NaN                                       ║
║  • formatted numbers (currency, %, thousands-sep)                            ║
║  • email/URL/phone-like columns detected                                     ║
║  • datetime column found                                                     ║
║  • strongly skewed numerical (|skew| >= C.SKEW_STRONG = 1.0)                ║
║  • outliers > 0 (INFO) or > C.OUTLIER_WARN_PCT=5% (WARNING)                 ║
║                                                                              ║
║  W.sort(key=lambda w: ORDER[w["severity"]])   ← HIGH first                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🧠 ZONE 4 — AutoML Training Flow (`experiments.py` → `modeling.py` / `forecasting.py`)

```
POST /api/experiments              POST /api/experiments/{id}/train
│                                  │
▼                                  ▼
╔══════════════════════╗           ╔════════════════════════════════════════════╗
║ create_experiment()  ║           ║  train(exp_id, background: BackgroundTasks)║
║                      ║           ║                                            ║
║ Validates:           ║           ║  if exp["status"] in ("queued","running"): ║
║  • dataset_id exists ║           ║      raise HTTPException(409)              ║
║  • target in columns ║           ║  exp["status"] = "queued"                  ║
║  • keep/drop valid   ║           ║  tracker.save(exp)                         ║
║  • mode="forecast":  ║           ║  background.add_task(run_training, exp_id) ║
║    datetime_col must ║           ║  return HTTP 202 Accepted ← immediately    ║
║    be datetime type  ║           ╚═══════════════╦════════════════════════════╝
║  • mode="standard":  ║                           ║ background thread
║    problem_type req'd║                           ▼
║                      ║           ╔════════════════════════════════════════════╗
║ tracker.create()     ║           ║       run_training(exp_id)  [THREAD]       ║
║  exp_id = uuid[:10]  ║           ║                                            ║
║  version auto-incr   ║           ║  exp["status"] = "running"                 ║
║  status = "created"  ║           ║  tracker.save(exp)                         ║
║  registry = "Dev"    ║           ║                                            ║
╚══════════════════════╝           ║  df = storage.load_typed(dataset_id)       ║
                                   ║  profile = storage.load_meta(dataset_id)   ║
                                   ║                                            ║
                                   ║  if exp["mode"] == "forecast":             ║
                                   ║      results, art = forecasting.train_forecast()║
                                   ║  else:                                     ║
                                   ║      results, art = modeling.train_tabular()║
                                   ║                                            ║
                                   ║  tracker.save_artifact(exp_id, art)        ║
                                   ║  exp["results"] = results                  ║
                                   ║  exp["status"] = "completed"               ║
                                   ║  tracker.log_mlflow(exp) ← if DYNAMICDS_MLFLOW=1║
                                   ║  tracker.save(exp)                         ║
                                   ╚═══════════════╦════════════════════════════╝
                            ┌──────────────────────┴─────────────────────────┐
                            ▼                                                  ▼
╔═══════════════════════════════════════╗   ╔══════════════════════════════════╗
║  modeling.py  ::  train_tabular()     ║   ║  forecasting.py :: train_forecast║
║                                       ║   ║                                  ║
║  1. select_features():                ║   ║  prepare_series():               ║
║     Excludes: "empty","constant",     ║   ║  ├─ pd.to_datetime(errors=coerce)║
║     "identifier"(unless kept),        ║   ║  ├─ infer_frequency():           ║
║     "text" columns                    ║   ║  │   pd.infer_freq(u)            ║
║                                       ║   ║  │   fallback: median day diff   ║
║  2. make_spec(): maps storage_type    ║   ║  │   27-32d→MS, 88-93d→QS       ║
║     → kind (numeric/bool/datetime/cat)║   ║  ├─ regularize_index():          ║
║                                       ║   ║  │   ts.dt.to_period().dt.to_ts()║
║  3. expand_features():                ║   ║  ├─ duplicate timestamps:         ║
║     datetime → year, month, day,      ║   ║  │   s.groupby(level=0).mean()  ║
║       weekday + sin/cos cyclical      ║   ║  ├─ reindex to full date_range   ║
║                                       ║   ║  └─ ffill() ← no future leakage  ║
║  4. LabelEncoder for classification   ║   ║                                  ║
║                                       ║   ║  detect_season():                ║
║  5. train_test_split():               ║   ║  ├─ detrend: y - polyfit(y,deg=1)║
║     test_size=0.2, stratify=y         ║   ║  ├─ acf(x, nlags=m, fft=True)[m]║
║                                       ║   ║  └─ threshold: acf >= 0.30       ║
║  6. imbalance check:                  ║   ║                                  ║
║     minority < C.IMBALANCE_MINORITY   ║   ║  Models:                         ║
║     (0.20 = 20%) → imbalanced=True    ║   ║  naive:  repeat(y[-1], h)        ║
║                                       ║   ║  snaive: y[-m + (i % m)]         ║
║  7. build_preprocessor(X_train, linear)║   ║  ets:    ExponentialSmoothing(   ║
║     Numeric columns:                  ║   ║    trend="add", damped_trend=True)║
║      log_cols: skew>C.SKEW_STRONG(1.0)║   ║  lag_xgb: LagModel:             ║
║       → log1p + median impute + scale ║   ║   lags: {1,2,3,m,2m}            ║
║      plain: median impute (+indicator)║   ║   windows: {3,m}                ║
║     Categorical:                      ║   ║   calendar: month,weekday,hour   ║
║      <= 30 unique → OneHotEncoder     ║   ║   XGBRegressor(n=300, lr=0.05)  ║
║       (min_freq=0.01, drop=if_binary) ║   ║   forecast(): recursive step    ║
║      > 30 unique → FrequencyEncoder   ║   ║    hist.append(pred) each step   ║
║                                       ║   ║                                  ║
║  8. Models evaluated:                 ║   ║  Chronological split:            ║
║     dummy → DummyClassifier/Regressor ║   ║  ev.chronological_split(n):      ║
║     logreg → LogisticRegression       ║   ║   a = int(n * 0.70)  ← train end║
║     rf     → RandomForestClassifier   ║   ║   b = int(n * 0.85)  ← val end  ║
║     xgb    → XGBClassifier           ║   ║                                  ║
║     all wrapped in Pipeline:          ║   ║  leader = min(cands,             ║
║      Pipeline([("prep",ct),("model",m)])║   ║    key=lambda r: r["primary_val"])║
║     cross_validate(pipe, X_tr, y_tr,  ║   ║                                  ║
║      cv=StratifiedKFold(k))           ║   ║  beats = leader["primary_test"]  ║
║                                       ║   ║    < base_best["primary_test"]   ║
║  9. Leader = best on primary_cv       ║   ║                                  ║
║     delta_vs_baseline computed        ║   ║  Future forecast:                ║
║                                       ║   ║  fut_idx = date_range(           ║
║  10. Explainability (leader only):    ║   ║    idx[-1], periods=horizon+1    ║
║      permutation_importances():       ║   ║    freq=fi["freq"])[1:]          ║
║       shuffle each col 5×, measure   ║   ╚══════════════════════════════════╝
║       score drop, cap at 500 rows     ║
║      shap_summary() (tree models):    ║
║       shap.TreeExplainer(model)       ║
║       .shap_values(Z) on 100-row samp ║
║       to_original(): strips "num__",  ║
║       "cat__","freq__" prefixes       ║
║      linear_effects() (linear models):║
║       model.coef_.ravel()             ║
║       sorted by |coefficient|         ║
╚═══════════════════════════════════════╝
```

---

## 🏛️ ZONE 5 — Model Artifact Persistence (`tracker.py`)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║              app/services/tracker.py — Experiment Registry                   ║
║                                                                              ║
║  _ID = re.compile(r"^[a-f0-9]{10}$")   ← validates exp_id before any ops   ║
║  _lock = threading.Lock()               ← concurrent-safe writes             ║
║                                                                              ║
║  save(exp):                                                                  ║
║  ├─ exp["updated_at"] = _now()          ← UTC ISO timestamp always updated  ║
║  ├─ tmp = p.with_suffix(".tmp")         ← write to temp file first           ║
║  ├─ tmp.write_text(json.dumps(to_jsonable(exp)))                             ║
║  └─ os.replace(tmp, p)                 ← atomic rename (crash-safe)         ║
║                                                                              ║
║  save_artifact(exp_id, art):                                                 ║
║  └─ joblib.dump(art, "data/models/{exp_id}.joblib")                         ║
║     art = { pipelines: {xgb: Pipeline(...), rf: ...},                       ║
║              spec, input_schema, baseline_row, leader, classes }             ║
║                                                                              ║
║  load_artifact(exp_id):                                                      ║
║  ├─ _check(exp_id)  ← regex guard                                            ║
║  └─ joblib.load("data/models/{exp_id}.joblib")                              ║
║                                                                              ║
║  set_registry_status(exp_id, status):                                        ║
║  ├─ status must be in ["Development","Candidate","Production","Archived"]    ║
║  ├─ exp["status"] must be "completed"                                        ║
║  └─ if status == "Production":                                               ║
║         for each existing experiment with same model name:                   ║
║             if registry_status == "Production": → "Archived"                 ║
║             (single Production per model name enforced)                      ║
║                                                                              ║
║  log_mlflow(exp):                                                            ║
║  ├─ only if DYNAMICDS_MLFLOW="1" or AUTODS_MLFLOW="1" env var set          ║
║  ├─ import mlflow  ← lazy optional import (won't crash if not installed)    ║
║  └─ mlflow.log_params({dataset_id, mode, target, model_version, ...})       ║
║                                                                              ║
║  DISK LAYOUT:                                                                ║
║  data/                                                                       ║
║   experiments/                                                               ║
║     {exp_id}.json ← {id, status, config, results, model:{name,version,reg}} ║
║   models/                                                                    ║
║     {exp_id}.joblib ← fitted pipelines + input schemas                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🔮 ZONE 6 — Inference / Prediction Flow

```
POST /api/models/{exp_id}/predict      POST /api/models/{exp_id}/batch-predict
│                                      │
▼                                      ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║         _art(exp_id):                                                       ║
║         ├─ _exp(exp_id) → tracker.get(exp_id)                              ║
║         ├─ if exp["status"] != "completed" → HTTPException(409)            ║
║         └─ tracker.load_artifact(exp_id) → art dict                        ║
╚══════════════╦══════════════════════════════════════════════════════════════╝
               ║
               ▼
╔═════════════════════════════════════════════════════════════════════════════╗
║    modeling.py  ::  predict_tabular(art, records, model_key, explain)       ║
║                                                                             ║
║    _prepare_input(art, records):                                            ║
║    ├─ Fill missing columns with np.nan                                     ║
║    ├─ .map(lambda v: np.nan if v is None or isnan or strip()=="" else str) ║
║    ├─ normalize_placeholders(raw)  ← same cleaning as training             ║
║    ├─ coerce_types(norm, spec)     ← same type casting as training         ║
║    └─ expand_features(typed, spec) ← same feature expansion as training    ║
║                                                                             ║
║    Classification:                                                          ║
║    ├─ proba = pipe.predict_proba(X)                                        ║
║    ├─ idx = proba.argmax(axis=1)                                           ║
║    └─ {prediction: classes[idx], confidence: proba[i,idx],                ║
║         probabilities: {c: p for c, p in zip(classes, proba[i])}}         ║
║                                                                             ║
║    Regression:                                                              ║
║    └─ {prediction: float(pipe.predict(X)[i])}                              ║
║                                                                             ║
║    If explain=True and single row:                                          ║
║    ex.local_explanation(pipe, X, art["baseline_row"], pt, cls_i):          ║
║    ├─ variants = concat([row] * (K+1))  ← K+1 copies of the input row     ║
║    ├─ for i, c in enumerate(cols):                                         ║
║    │     variants.loc[i+1, c] = baseline[c]  ← replace with median/mode   ║
║    ├─ predict all variants                                                  ║
║    └─ contribution = ref_pred - occluded_pred  (sorted by |contribution|)  ║
║                                                                             ║
║    Batch:  parse_upload() → predict → append "prediction"+"confidence" cols║
║    └─ return CSV + header: "X-Missing-Columns": ",".join(missing_cols)     ║
╚═════════════════════════════════════════════════════════════════════════════╝
```

---

## 📐 ZONE 7 — Configuration Backbone (`config.py`)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    app/config.py  — Central Constants                        ║
║                                                                              ║
║  PATHS:          DATA_DIR = env(DYNAMICDS_DATA_DIR) or ROOT/data            ║
║  UPLOAD LIMITS:  MAX_UPLOAD_MB=50  MAX_ROWS=500k  MAX_COLS=300              ║
║  PLACEHOLDERS:   {nan, null, none, na, n/a, ?, --, unknown, nil, ...}       ║
║  MISSINGNESS:    LOW=5%   MODERATE=20%   HIGH=80%   ROW_FLAG=0.70           ║
║  SKEWNESS:       SYMMETRIC<0.5   STRONG>=1.0                                ║
║  CORRELATION:    LOW<0.3   HIGH>=0.7   WARN>=0.9   LEAKAGE>=0.95           ║
║  CARDINALITY:    CARD_LOW=10   CARD_HIGH=50   RARE=0.01                    ║
║  OUTLIERS:       IQR_K=1.5   WARN_PCT=5.0%                                 ║
║  TEXT DETECT:    MIN_AVG_LEN=40   MIN_MEDIAN_TOKENS=5   MIN_UNIQUE=0.30    ║
║  SAMPLING:       VIS_SAMPLE_ROWS=50k   ASSOC_SAMPLE=100k   SHAPIRO=50k    ║
║  MODELING:       OHE_MAX=30   IMBALANCE=0.20   MIN_ROWS=30  SEED=42       ║
║  FORECASTING:    TRAIN_FRAC=0.70   VAL_FRAC=0.15   MIN_ROWS=30           ║
║  REGISTRY:       ["Development","Candidate","Production","Archived"]        ║
║  All overridable by env: DYNAMICDS_* or AUTODS_* prefix                     ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🎯 COMPLETE DEPENDENCY GRAPH

```
                         main.py
                        /        \
              datasets.py       experiments.py
              /    |    \        /    |    \
       storage  eda  warnings  storage tracker modeling/forecasting
          |      |      |        |       |       |
     type_inf  eda.py  eda.py  type_inf joblib  preprocessing
     data_q   timeutils config  data_q  mlflow  evaluation
        |                |             |        explainability
     config.py        config.py     config.py      shap
                                                 config.py
                         ↑
             utils/jsonable.py  (used by ALL API layers)
             utils/timeutils.py (used by eda + forecasting)
```
