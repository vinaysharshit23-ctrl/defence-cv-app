// Defence CV — inference via server.py (same origin)
const API_URL = '';
const CLASSES = ["aircraft", "drone", "helicopter", "military-vehicle", "naval"];
const CLASS_CONFIG = {
  aircraft:           { label: 'Fighter Aircraft',         sublabel: 'e.g. F-22, Su-57, Rafale',   color: '#f87171' },
  drone:              { label: 'Drone / UAV',               sublabel: 'e.g. MQ-9 Reaper, TB2',      color: '#a78bfa' },
  helicopter:         { label: 'Military Helicopter',       sublabel: 'e.g. AH-64 Apache, Mi-28',   color: '#fb923c' },
  'military-vehicle': { label: 'Military Ground Vehicle',   sublabel: 'e.g. Tank, APC, IFV',        color: '#60a5fa' },
  naval:              { label: 'Naval Vessel',              sublabel: 'e.g. Destroyer, Frigate',    color: '#34d399' },
};

let serverReady = false;
let confidenceThreshold = 0.30;
let webcamRunning = false;
let webcamStream = null;
let webcamRAF = null;
let webcamBusy = false;
let lastFPSTime = performance.now();
let fpsFrames = 0;

function dbg(msg) {
  console.log('[DefenceCV]', msg);
  const el = document.getElementById('model-error');
  if (el) { el.style.display = 'block'; el.style.color = '#facc15'; el.textContent = msg; }
}

window.addEventListener('DOMContentLoaded', () => {
  dbg('JS loaded, starting health check...');
  setModelStatus('loading', 'Connecting...');
  doHealthCheck();
});

async function doHealthCheck() {
  dbg('Fetching /health...');
  try {
    const res = await fetch('/health');
    dbg('Got response: ' + res.status);
    if (!res.ok) { throw new Error('HTTP ' + res.status); }
    const data = await res.json();
    dbg('Server OK: ' + JSON.stringify(data));
    serverReady = true;
    setModelStatus('ready', 'Ready — ' + data.classes.length + ' defence classes');
    const errEl = document.getElementById('model-error');
    if (errEl) errEl.style.display = 'none';
  } catch (e) {
    dbg('Health check FAILED: ' + e.message + ' | ' + e.toString());
    setModelStatus('error', 'Server error — see yellow text');
    showError('Error: ' + e.message + '\nMake sure server cmd window is open.\nRetry below.');
  }
}

window.retryConnect = function() {
  dbg('Retry clicked');
  setModelStatus('loading', 'Retrying...');
  doHealthCheck();
};

function setModelStatus(state, text) {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('model-status-text');
  if (dot) { dot.className = 'status-dot ' + state; }
  if (txt) txt.textContent = text;
}

function showProcessing(s) {
  document.getElementById('processing-overlay').style.display = s ? 'flex' : 'none';
  const scanLine = document.getElementById('scan-line');
  if (scanLine) scanLine.style.display = s ? 'block' : 'none';
}

function showError(msg) {
  const el = document.getElementById('model-error');
  if (!el) return;
  el.textContent = msg;
  el.style.color = '#ef4444';
  el.style.display = 'block';
}

// ── Mode switching ────────────────────────────────────────────────────────────
function switchMode(mode) {
  if (webcamRunning) stopWebcam();
  clearResults();
  document.querySelectorAll('.mode-section').forEach(s => s.classList.remove('active'));
  document.getElementById('mode-' + mode).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + mode).classList.add('active');
}

function updateThreshold(val) {
  confidenceThreshold = parseInt(val) / 100;
  document.getElementById('threshold-val').textContent = val;
}

// ── Image loading ─────────────────────────────────────────────────────────────
function handleDragOver(e) { e.preventDefault(); document.getElementById('drop-zone').classList.add('drag-over'); }
function handleDragLeave()  { document.getElementById('drop-zone').classList.remove('drag-over'); }
function handleDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith('image/')) loadFile(f);
}
function handleFileSelect(e) { if (e.target.files[0]) loadFile(e.target.files[0]); }

function loadFile(file) {
  if (!serverReady) { alert('Server not ready. Check the yellow debug text in the sidebar.'); return; }
  showProcessing(true);
  const url = URL.createObjectURL(file);
  const img = document.getElementById('result-image');
  img.onload = async () => {
    showCanvasWrapper(true);
    img.style.display = 'block';
    document.getElementById('webcam-video').style.display = 'none';
    try { await sendImageFile(file, img); }
    catch (e) { alert('Inference failed: ' + e.message); console.error(e); }
    showProcessing(false);
    URL.revokeObjectURL(url);
  };
  img.onerror = () => { showProcessing(false); alert('Could not display image.'); };
  img.src = url;
}

function loadFromURL() {
  const u = document.getElementById('url-input').value.trim();
  if (!u) return;
  if (!serverReady) { alert('Server not ready.'); return; }
  showProcessing(true);
  fetch(u)
    .then(r => r.blob())
    .then(blob => {
      const file = new File([blob], 'img.jpg', { type: blob.type || 'image/jpeg' });
      const url  = URL.createObjectURL(blob);
      const img  = document.getElementById('result-image');
      img.onload = async () => {
        showCanvasWrapper(true);
        img.style.display = 'block';
        try { await sendImageFile(file, img); } catch(e) { alert(e.message); }
        showProcessing(false);
      };
      img.src = url;
    })
    .catch(e => { showProcessing(false); alert('Cannot fetch URL: ' + e.message); });
}

// ── Send to server (two-phase) ────────────────────────────────────────────────
async function sendImageFile(file, imgEl) {
  const fd = new FormData();
  fd.append('image', file, file.name || 'image.jpg');

  // Phase 1 — instant TFLite result
  dbg('POSTing to /predict...');
  const res = await fetch('/predict', { method: 'POST', body: fd });

  if (res.status === 422) {
    // User input error — show friendly message, don't proceed to /identify
    const err = await res.json();
    showInputError(err.error || 'Invalid image file.');
    return;
  }
  if (!res.ok) {
    const txt = await res.text();
    throw new Error('Server error ' + res.status + ': ' + txt);
  }

  const data = await res.json();
  const predictions = data.predictions;
  const top = predictions[0];
  dbg('Predict OK: ' + top.class + ' ' + top.score);

  drawHUD(imgEl, predictions);
  renderChips(predictions);

  // Show preprocessing info panel
  if (data.preprocessing) showPreprocessInfo(data.preprocessing);

  // Show low-confidence warning if TFLite is uncertain
  if (top.low_confidence) {
    const reason = top.low_confidence_reason || '';
    const reasonSuffix = reason ? ` (${reason.toLowerCase()})` : '';
    showWarning('Result may be unreliable' + reasonSuffix + '. Try a clearer, better-lit, or less obscured image.');
    // Also mark the top chip with an amber indicator
    const firstChipDot = document.querySelector('#detections-list .detection-chip .chip-dot');
    if (firstChipDot) {
      firstChipDot.style.background = '#f59e0b';
      firstChipDot.style.boxShadow = '0 0 6px #f59e0b';
      firstChipDot.title = 'Low confidence';
    }
  } else {
    clearWarning();
  }

  // Phase 2 — async: get real name + reason from OpenRouter
  updateChipIdentifying(true);
  try {
    const fd2 = new FormData();
    fd2.append('image', file, file.name || 'image.jpg');
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 45000);
    const res2 = await fetch('/identify', { method: 'POST', body: fd2, signal: ctrl.signal });
    clearTimeout(timer);
    if (res2.ok) {
      const id = await res2.json();
      predictions[0].specific_name = id.identified_by === 'heuristic'
        ? id.specific_name + ' (est.)'
        : id.specific_name;
      predictions[0].reason = id.reason || '';
      predictions[0].identified_by = id.identified_by;
      // Store AI's 2nd and 3rd guesses on predictions[1] and predictions[2]
      if (id.guess2) {
        predictions[1].ai_guess_name = id.guess2;
      }
      if (id.guess3) {
        predictions[2].ai_guess_name = id.guess3;
      }
      dbg('Identify: ' + id.specific_name + ' | #2: ' + (id.guess2||'') + ' | #3: ' + (id.guess3||''));
      drawHUD(imgEl, predictions);
      renderChips(predictions);
    }
  } catch (e) {
    predictions[0].specific_name = 'Unknown';
    predictions[0].reason = '';
    predictions[0].identified_by = 'unknown';
    dbg('Identify failed: ' + e.message);
    drawHUD(imgEl, predictions);
    renderChips(predictions);
  } finally {
    updateChipIdentifying(false);
  }
}

function updateChipIdentifying(active) {
  const firstChip = document.querySelector('#detections-list .detection-chip strong');
  if (!firstChip) return;
  if (active) {
    firstChip.dataset.origText = firstChip.textContent;
    firstChip.textContent = firstChip.dataset.origText + ' 🔍';
    firstChip.style.opacity = '0.7';
  } else {
    // restored by renderChips rerender — just clean up if still showing
    if (firstChip.dataset.origText) {
      firstChip.style.opacity = '';
      delete firstChip.dataset.origText;
    }
  }
}

async function sendCanvas(canvas) {
  const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.8));
  const file = new File([blob], 'frame.jpg', { type: 'image/jpeg' });
  // Webcam: only TFLite (phase 1), skip /identify for real-time performance
  const fd = new FormData();
  fd.append('image', file, 'frame.jpg');
  const res = await fetch('/predict', { method: 'POST', body: fd });
  if (!res.ok) throw new Error('predict ' + res.status);
  const data = await res.json();
  drawHUD(canvas, data.predictions);
  renderChips(data.predictions);
}

// ── Canvas HUD ─────────────────────────────────────────────────────────────────
function drawHUD(source, results) {
  const canvas = document.getElementById('overlay-canvas');
  const ctx    = canvas.getContext('2d');

  // Use the rendered display size of the image/video, not its natural resolution
  const rect = source.getBoundingClientRect ? source.getBoundingClientRect() : null;
  const displayW = rect ? rect.width  : (source.videoWidth  || source.naturalWidth  || 400);
  const displayH = rect ? rect.height : (source.videoHeight || source.naturalHeight || 300);

  canvas.width  = displayW;
  canvas.height = displayH;

  // Position the canvas exactly over the image within the container
  const container = document.getElementById('canvas-container');
  const containerRect = container ? container.getBoundingClientRect() : null;
  if (rect && containerRect) {
    canvas.style.top    = (rect.top  - containerRect.top)  + 'px';
    canvas.style.left   = (rect.left - containerRect.left) + 'px';
    canvas.style.width  = displayW + 'px';
    canvas.style.height = displayH + 'px';
  } else {
    canvas.style.top = '0'; canvas.style.left = '0';
    canvas.style.width = '100%'; canvas.style.height = '100%';
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const top = results[0];
  if (!top) return;
  const cfg = CLASS_CONFIG[top.class] || { label: top.class, color: '#6b7280' };
  const specificName = top.specific_name || cfg.label;

  ctx.strokeStyle = cfg.color; ctx.lineWidth = 2;
  ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);

  const cs = 20; ctx.lineWidth = 3; ctx.beginPath();
  [
    [2, 2+cs, 2, 2, 2+cs, 2],
    [canvas.width-2-cs, 2, canvas.width-2, 2, canvas.width-2, 2+cs],
    [2, canvas.height-2-cs, 2, canvas.height-2, 2+cs, canvas.height-2],
    [canvas.width-2-cs, canvas.height-2, canvas.width-2, canvas.height-2, canvas.width-2, canvas.height-2-cs]
  ].forEach(([x1,y1,cx,cy,x2,y2]) => { ctx.moveTo(x1,y1); ctx.lineTo(cx,cy); ctx.lineTo(x2,y2); });
  ctx.strokeStyle = cfg.color; ctx.stroke();

  const hasReason = top.reason && top.reason.length > 0;
  const barH = hasReason ? 80 : 52;
  const barY = canvas.height - barH;
  ctx.fillStyle = 'rgba(0,0,0,0.78)';
  ctx.fillRect(0, barY, canvas.width, barH);
  ctx.fillStyle = cfg.color + '44';
  ctx.fillRect(0, barY, canvas.width * top.score, barH);

  // Row 1: specific name + category tag + confidence %
  ctx.font = 'bold 15px "Segoe UI",system-ui,sans-serif';
  ctx.fillStyle = cfg.color;
  ctx.textAlign = 'left';
  ctx.fillText(specificName, 10, barY + 20);
  // Category pill after the name
  const nameW = ctx.measureText(specificName).width;
  const catTag = cfg.label.toUpperCase();
  ctx.font = 'bold 9px "Segoe UI",system-ui,sans-serif';
  const tagW = ctx.measureText(catTag).width + 10;
  const tagX = 10 + nameW + 8;
  const tagY = barY + 7;
  const tagH = 16;
  ctx.fillStyle = cfg.color + '33';
  ctx.strokeStyle = cfg.color + '88';
  ctx.lineWidth = 1;
  const r2 = 3;
  ctx.beginPath();
  ctx.roundRect(tagX, tagY, tagW, tagH, r2);
  ctx.fill(); ctx.stroke();
  ctx.fillStyle = cfg.color;
  ctx.textAlign = 'left';
  ctx.fillText(catTag, tagX + 5, tagY + 11);
  // Confidence
  ctx.font = 'bold 15px "Segoe UI",system-ui,sans-serif';
  const confText = `${Math.round(top.score * 100)}%`;
  ctx.fillStyle = cfg.color;
  ctx.textAlign = 'right';
  ctx.fillText(confText, canvas.width - 8, barY + 20);
  ctx.textAlign = 'left';

  // Row 2: category label (keep as subtitle)
  ctx.font = '11px "Segoe UI",system-ui,sans-serif';
  ctx.fillStyle = 'rgba(255,255,255,0.45)';
  ctx.fillText(cfg.label, 10, barY + 38);

  // Row 3-4: AI reason wrapped to 2 lines
  if (hasReason) {
    ctx.font = '10px "Segoe UI",system-ui,sans-serif';
    ctx.fillStyle = 'rgba(255,255,255,0.65)';
    const maxW = canvas.width - 20;
    const words = top.reason.split(' ');
    let line1 = '', line2 = '';
    let building = '';
    let splitDone = false;
    for (const w of words) {
      const test = building ? building + ' ' + w : w;
      if (!splitDone && ctx.measureText(test).width > maxW) {
        line1 = building;
        building = w;
        splitDone = true;
      } else {
        building = test;
      }
    }
    if (!splitDone) {
      line1 = building;
    } else {
      line2 = building;
      // Truncate line2 if still too long
      while (line2.length > 4 && ctx.measureText(line2 + '…').width > maxW) {
        line2 = line2.slice(0, -1);
      }
      if (line2 !== building) line2 += '…';
    }
    ctx.fillText(line1, 10, barY + 56);
    if (line2) ctx.fillText(line2, 10, barY + 70);
  }

  // Low-confidence amber badge — top-right corner
  if (top.low_confidence) {
    const badgeText = '⚠ UNCERTAIN';
    ctx.font = 'bold 11px "Segoe UI",system-ui,sans-serif';
    const bw = ctx.measureText(badgeText).width + 16;
    const bh = 22;
    const bx = canvas.width - bw - 8;
    const by = 8;
    ctx.fillStyle = 'rgba(245,158,11,0.18)';
    ctx.beginPath();
    ctx.roundRect ? ctx.roundRect(bx, by, bw, bh, 4) : ctx.rect(bx, by, bw, bh);
    ctx.fill();
    ctx.strokeStyle = 'rgba(245,158,11,0.7)';
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.fillStyle = '#fbbf24';
    ctx.textAlign = 'center';
    ctx.fillText(badgeText, bx + bw / 2, by + 15);
    ctx.textAlign = 'left';
  }
}

// ── Chips ─────────────────────────────────────────────────────────────────────
function renderChips(results) {
  const panel = document.getElementById('detections-panel');
  const list  = document.getElementById('detections-list');
  list.innerHTML = '';
  panel.style.display = 'block';

  // Show all 5 but visually separate top-3
  results.forEach((r, i) => {
    const cfg = CLASS_CONFIG[r.class] || { label: r.class, sublabel: '', color: '#6b7280' };
    // Chip 0: AI primary name + reason
    // Chip 1: AI's 2nd guess (or TFLite category if not yet identified)
    // Chip 2: AI's 3rd guess (or TFLite category if not yet identified)
    // Chips 3-4: TFLite remaining categories (dimmed)
    let specificName, subtitle, dotColor;
    if (i === 0) {
      specificName = r.specific_name || cfg.label;
      subtitle     = r.reason || cfg.label;
      dotColor     = cfg.color;
    } else if ((i === 1 || i === 2) && r.ai_guess_name) {
      specificName = r.ai_guess_name;
      subtitle     = 'AI alternative guess';
      dotColor     = i === 1 ? '#94a3b8' : '#64748b';
    } else {
      specificName = cfg.label;
      subtitle     = cfg.sublabel || cfg.label;
      dotColor     = cfg.color;
    }
    const isTop3 = i < 3;

    const chip = document.createElement('div');
    chip.className = 'detection-chip' + (i === 0 ? ' chip-primary' : '');
    chip.style.opacity = r.score >= confidenceThreshold ? (isTop3 ? 1 : 0.35) : 0.25;
    if (!isTop3) chip.style.fontSize = '11px';

    const categoryBadge = `<span style="
      display:inline-block;
      margin-left:7px;
      padding:1px 7px;
      border-radius:4px;
      font-size:9px;
      font-weight:600;
      letter-spacing:.06em;
      text-transform:uppercase;
      background:${dotColor}22;
      border:1px solid ${dotColor}66;
      color:${dotColor};
      vertical-align:middle;
      line-height:1.6;
    ">${cfg.label}</span>`;

    chip.innerHTML = `
      <span class="chip-dot" style="background:${dotColor}"></span>
      <span class="chip-label">
        <strong>${specificName}</strong>${i === 0 ? categoryBadge : ''}
        <span style="font-size:${i===0?'10px':'9px'};opacity:${i===0?'0.75':'0.55'};display:block;line-height:1.5;margin-top:2px;white-space:normal">${subtitle}</span>
      </span>
      <span class="chip-conf" style="color:${i===0?'var(--accent)':'var(--text2)'}">${Math.round(r.score*100)}%</span>`;
    list.appendChild(chip);
  });
}

// ── Warning / error banners ───────────────────────────────────────────────────
function showWarning(msg) {
  let el = document.getElementById('confidence-warning');
  if (!el) {
    el = document.createElement('div');
    el.id = 'confidence-warning';
    el.style.cssText = [
      'margin:0 24px 10px',
      'padding:10px 16px',
      'border-radius:8px',
      'font-size:13px',
      'font-weight:700',
      'background:rgba(245,158,11,0.15)',
      'border:1.5px solid rgba(245,158,11,0.55)',
      'color:#fbbf24',
      'display:flex',
      'align-items:center',
      'gap:10px',
      'letter-spacing:.03em',
      'box-shadow:0 0 12px rgba(245,158,11,0.12)',
    ].join(';');
    document.querySelector('.main').appendChild(el);
  }
  el.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#fbbf24" stroke-width="2.5" style="flex-shrink:0">
      <path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/>
      <line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>
    </svg>
    <div>
      <div style="font-size:13px;font-weight:700;color:#fbbf24">UNCERTAIN PREDICTION</div>
      <div style="font-size:11px;font-weight:500;opacity:.8;margin-top:2px">${msg.replace('⚠ Low confidence — ', '')}</div>
    </div>`;
  el.style.display = 'flex';
}

function clearWarning() {
  const el = document.getElementById('confidence-warning');
  if (el) el.style.display = 'none';
}

function showInputError(msg) {
  showProcessing(false);
  clearWarning();
  let el = document.getElementById('input-error-banner');
  if (!el) {
    el = document.createElement('div');
    el.id = 'input-error-banner';
    el.style.cssText = 'margin:0 24px 8px;padding:10px 16px;border-radius:6px;font-size:13px;font-weight:600;background:rgba(239,68,68,0.12);border:1px solid rgba(239,68,68,0.35);color:#ef4444;display:flex;align-items:center;gap:10px;';
    document.querySelector('.main').appendChild(el);
  }
  el.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> ${msg}`;
  el.style.display = 'flex';
  setTimeout(() => { if (el) el.style.display = 'none'; }, 6000);
}

function showPreprocessInfo(meta) {
  let el = document.getElementById('preprocess-info');
  if (!el) {
    el = document.createElement('details');
    el.id = 'preprocess-info';
    el.style.cssText = 'margin:0 24px 8px;border-radius:6px;font-size:11.5px;background:rgba(6,182,212,0.05);border:1px solid rgba(6,182,212,0.18);color:var(--text-muted,#94a3b8);overflow:hidden;';
    document.querySelector('.main').appendChild(el);
  }
  const stepsHtml = (meta.steps || []).map(s =>
    `<span style="display:inline-flex;align-items:center;gap:4px;padding:2px 7px;border-radius:3px;background:rgba(6,182,212,0.1);margin:2px 2px 2px 0">
      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>${s}
    </span>`
  ).join('');
  el.innerHTML = `
    <summary style="padding:7px 12px;cursor:pointer;list-style:none;display:flex;align-items:center;gap:8px;user-select:none;">
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="#06b6d4" stroke-width="2.5"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
      <span style="color:#06b6d4;font-weight:600;letter-spacing:.05em">PREPROCESSING</span>
      <span style="margin-left:auto;opacity:.6">${meta.original_size} · ${meta.original_format} · ${meta.file_size_kb} KB</span>
    </summary>
    <div style="padding:6px 12px 10px;border-top:1px solid rgba(6,182,212,0.12);">
      <div style="margin-bottom:6px;">${stepsHtml}</div>
      <div style="display:flex;gap:16px;flex-wrap:wrap;font-size:11px;opacity:.7;margin-top:4px;">
        <span>Input: <b style="color:#e2e8f0">${meta.original_size} ${meta.original_mode}</b></span>
        <span>→ Model: <b style="color:#e2e8f0">${meta.model_input}</b></span>
      </div>
    </div>`;
  el.style.display = 'block';
}

function clearPreprocessInfo() {
  const el = document.getElementById('preprocess-info');
  if (el) el.style.display = 'none';
}

// ── Webcam ─────────────────────────────────────────────────────────────────────
async function toggleWebcam() {
  if (webcamRunning) { stopWebcam(); return; }
  const btn = document.getElementById('webcam-btn');
  btn.disabled = true; btn.textContent = 'Starting...';
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
  } catch (e) {
    alert('Camera denied: ' + e.message);
    btn.disabled = false; btn.textContent = '▶ Start Camera';
    return;
  }
  const video = document.getElementById('webcam-video');
  video.srcObject = webcamStream;
  video.style.display = 'block';
  document.getElementById('result-image').style.display = 'none';
  showCanvasWrapper(true);
  video.onloadedmetadata = () => {
    webcamRunning = true;
    btn.disabled = false; btn.textContent = '⏹ Stop Camera';
    document.getElementById('fps-badge').style.display = 'inline';
    webcamLoop();
  };
}

function stopWebcam() {
  webcamRunning = false;
  cancelAnimationFrame(webcamRAF);
  if (webcamStream) { webcamStream.getTracks().forEach(t => t.stop()); webcamStream = null; }
  document.getElementById('webcam-video').srcObject = null;
  document.getElementById('fps-badge').style.display = 'none';
  document.getElementById('webcam-btn').textContent = '▶ Start Camera';
}

async function webcamLoop() {
  if (!webcamRunning) return;
  const video = document.getElementById('webcam-video');
  if (video.readyState === 4 && !webcamBusy) {
    webcamBusy = true;
    const oc = document.createElement('canvas');
    oc.width = 224; oc.height = 224;
    oc.getContext('2d').drawImage(video, 0, 0, 224, 224);
    try {
      await sendCanvas(oc);
      fpsFrames++;
      const now = performance.now();
      if (now - lastFPSTime >= 1000) {
        document.getElementById('fps-badge').textContent = fpsFrames + ' FPS';
        fpsFrames = 0; lastFPSTime = now;
      }
    } catch (_) {}
    webcamBusy = false;
  }
  webcamRAF = requestAnimationFrame(webcamLoop);
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function showCanvasWrapper(s) { document.getElementById('canvas-wrapper').style.display = s ? 'flex' : 'none'; }
function clearResults() {
  clearWarning();
  clearPreprocessInfo();
  const errBanner = document.getElementById('input-error-banner');
  if (errBanner) errBanner.style.display = 'none';
  showCanvasWrapper(false);
  document.getElementById('detections-panel').style.display = 'none';
  document.getElementById('result-image').src = '';
  document.getElementById('result-image').style.display = 'none';
  const c = document.getElementById('overlay-canvas');
  c.getContext('2d').clearRect(0, 0, c.width, c.height);
  const fi = document.getElementById('file-input');
  if (fi) fi.value = '';
}
