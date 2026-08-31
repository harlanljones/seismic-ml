/**
 * SeismicML — Live Forecast Interactive Frontend
 * Geophysical Observatory visual world. Static-first, hybrid live-upgrade.
 *
 * Data contract (window.SEISMIC_BAKED or live API):
 *   manifest  — {baked_at, model_version, forecast_period, verifiable_after, grid_size_deg, history}
 *   forecast  — {forecast_period, verifiable_after, generated_at, grid_size_deg, model_version, features:[{lat_bin,lon_bin,period,prob}]}
 *   receipts  — {model_version, test_acc, train_acc, val_acc, train_val_gap, test_auc, confusion:{tp,fp,fn,tn}, hit_rate_pct, false_alarm_rate_pct, markers:[{lat,lon,prob,gt,pred,color,label}]}
 *   events    — [{lon, lat, mag, time}]
 *   model     — {model_version, input_dim, feature_cols, grid_size_deg, time_step_days, min_mag, target_mag, ...}
 *   history   — [{ts, forecast_period, verifiable_after, generated_at, features(count)}]
 */

(function () {
  "use strict";

  /* ================================================================
   * CONSTANTS — mirror src/serve/map_server.py semantics
   * ================================================================ */

  var TRAIN_HEAT = {
    radius: 12,
    blur: 10,
    minOpacity: 0.2,
    gradient: { 0.2: "#2244ff", 0.8: "#00d2ff" },
  };

  var PRED_HEAT = {
    radius: 16,
    blur: 14,
    minOpacity: 0.4,
    gradient: { 0.4: "#ff9e4a", 1.0: "#ff3c3c" },
  };

  // Confusion-matrix marker colors (map.py:66-95 / map_server.py:_make_markers)
  var MARKER_TP = "#2bd97a";
  var MARKER_FP = "#ffb02e";
  var MARKER_FN = "#ff4d5e";

  var GRID_SIZE = 2.0; // fallback if data absent

  // --- Beat catalogue -------------------------------------------------
  var BEATS = [
    {
      id: 1,
      tag: "Observation",
      head: "The Watch",
      cta: "Beat 2 — The Signal \u2192",
      hint: "Earthquakes don't happen in isolation. The model looks for patterns " +
        "in 30-day seismic precursor fields across a 2\u00b0 geographic grid \u2014 " +
        "event density, mean depth, azimuthal gap, and signal \u2014 before asking " +
        "whether those signals presage a larger rupture ahead.",
      sigKey: "Signals",
      legKey: "Recent events",
    },
    {
      id: 2,
      tag: "Signal",
      head: "The Signal",
      cta: "Beat 3 — The Forecast \u2192",
      hint: "Zoom into one high-risk cell to examine its 30-day precursor signature. " +
        "What does the model see that makes it ask: <em>will an M\u22654.5 strike in " +
        "the next 7 days here</em>?",
      sigKey: "Precursors",
      legKey: "Heat",
    },
    {
      id: 3,
      tag: "Forecast",
      head: "The Forecast",
      cta: "Beat 4 — The Receipts \u2192",
      hint: "The model maps its forecast across every grid bin for the 7-day window. " +
        "Hotter = higher probability of M\u22654.5+. The heatmap threshold is " +
        "adjustable \u2014 raising it means fewer alerts but higher confidence per alert.",
      sigKey: "Forecast signals",
      legKey: "Risk level",
    },
    {
      id: 4,
      tag: "Receipts",
      head: "The Receipts",
      cta: "Beat 5 — Playground \u2192",
      hint: "Honest verification on the held-out test partition. True positives " +
        "confirm the model sees what matters. False alarms and misses are shown " +
        "with equal weight \u2014 there is no escaping them in seismic forecasting.",
      sigKey: "Verification",
      legKey: "Verification",
    },
    {
      id: 5,
      tag: "Playground",
      head: "Take Control",
      cta: "Share this view",
      hint: "Full interactive dashboard. Adjust the alert threshold, drill into " +
        "individual cells, browse the top-10 highest-risk regions, and step " +
        "through forecast history snapshots.",
      sigKey: "Dashboard",
      legKey: "Markers",
    },
  ];

  /* ================================================================
   * STATE
   * ================================================================ */

  var state = {
    baked: null,
    live: false,
    forecast: null,
    model: null,
    receipts: null,
    events: [],
    history: [],
    manifest: null,
    beat: 1,
    threshold: 50,
    activeSnapshot: null,
    map: null,
    layers: {},
         activeHighCellIndex: null,
  };

  /* ================================================================
   * DOM HELPERS
   * ================================================================ */

  function el(tag, props, children) {
    var node = document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function (k) {
        if (k === "className") node.className = props[k];
        else if (k === "html") node.innerHTML = props[k];
        else if (k === "text") node.textContent = props[k];
        else if (k.indexOf("on") === 0) node.addEventListener(k.slice(2), props[k]);
        else if (props[k] != null) node.setAttribute(k, props[k]);
      });
    }
    if (children) {
      children.forEach(function (c) {
        if (typeof c === "string") node.appendChild(document.createTextNode(c));
        else node.appendChild(c);
      });
    }
    return node;
  }

  function setText(id, text) {
    var node = document.getElementById(id);
    if (node) node.textContent = text;
  }

  function setHTML(id, html) {
    var node = document.getElementById(id);
    if (node) node.innerHTML = html;
  }

  function fmtDate(ts) {
    if (!ts) return "\u2014";
    var d = new Date(ts);
    if (isNaN(d.getTime())) return ts;
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    });
  }

  function fmtPeriod(ts) {
    if (!ts) return "pending";
    return ts.replace("T", " ").slice(0, 16);
  }

  function fmtNum(n, d) {
    d = d == null ? 1 : d;
    if (n == null || isNaN(n)) return "\u2014";
        return Number(n).toFixed(d);
  }

  /* ================================================================
   * TOAST
   * ================================================================ */

  var toastTimer = null;
  function showToast(msg, type) {
    var t = document.getElementById("toast");
    if (!t) return;
    t.textContent = msg;
    t.className = "toast " + (type || "info");
    t.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { t.hidden = true; }, 3500);
  }

  /* ================================================================
   * DATA LOADING — static first, live upgrade
   * ================================================================ */

  async function probeLive() {
    try {
      var resp = await fetch("/health", { cache: "no-store" });
      if (resp.ok && resp.headers.get("content-type") &&
          resp.headers.get("content-type").indexOf("application/json") >= 0) {
        var h = await resp.json();
        return h.status === "ok";
      }
    } catch (e) { /* offline or no backend */ }
    return false;
  }

  async function fetchLive() {
    var fResp = await fetch("/forecast", { cache: "no-store" });
    var mResp = await fetch("/model", { cache: "no-store" });
    var geo = await fResp.json();
    var meta = await mResp.json();

    var features = (geo.features || []).map(function (ft) {
      return {
        lat_bin: ft.properties.lat_bin,
        lon_bin: ft.properties.lon_bin,
        period: ft.properties.period,
        prob: ft.properties.prob,
      };
    });

    var forecast = {
      forecast_period: geo.forecast_period,
      verifiable_after: geo.verifiable_after,
      generated_at: Date.now(),
      grid_size_deg: geo.grid_size_deg || GRID_SIZE,
      model_version: meta.model_version,
      features: features,
    };
    return { forecast: forecast, model: meta };
  }

  async function initData() {
    state.baked = window.SEISMIC_BAKED || null;

    if (!state.baked) {
      if (await probeLive()) {
        state.live = true;
        try {
          var live = await fetchLive();
          state.forecast = live.forecast;
          state.model = live.model;
          state.receipts = null;
          state.events = [];
          state.history = [];
        } catch (e) {
          showToast("Live backend reachable but data fetch failed.", "err");
          renderEmptyStatePlaceholder();
          return false;
        }
      } else {
        showToast("No data — run bake or start the API server.", "err");
        renderEmptyStatePlaceholder();
        return false;
      }
    } else {
      state.manifest = state.baked.manifest || {};
      state.history = state.baked.history || state.manifest.history || [];
      state.receipts = state.baked.receipts || null;
      state.events = state.baked.events || [];
      state.forecast = state.baked.forecast || null;
      state.model = state.baked.model || null;

      if (await probeLive()) {
        state.live = true;
        try {
          var upgraded = await fetchLive();
          state.forecast = upgraded.forecast;
          state.model = upgraded.model;
        } catch (e) {
          showToast("Backend online but live fetch failed \u2014 showing baked data.", "warn");
        }
      }
    }

    if (!state.receipts && state.forecast) {
      state.receipts = {
        markers: [],
        confusion: { tp: 0, fp: 0, fn: 0, tn: 0 },
        label: "Live forecast (no test-set receipts available)",
            };
    }
    return true;
  }

  /* ================================================================
   * MAP
   * ================================================================ */

  function initMap() {
    var map = L.map("map", {
      zoomControl: true,
      attributionControl: true,
      zoomSnap: 0.5,
      zoomDelta: 0.5,
      scrollWheelZoom: true,
    }).setView([14, -30], 2);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19,
      attribution: "&copy; CartoDB | USGS | SeismicML",
      tileSize: 256,
      detectRetina: true,
    }).addTo(map);

    state.layers.events = L.layerGroup().addTo(map);
    state.layers.trainHeat = L.heatLayer([], TRAIN_HEAT);
    state.layers.predHeat = L.heatLayer([], PRED_HEAT);
    state.layers.markerCluster = L.markerClusterGroup();
    state.layers.markers = L.layerGroup().addTo(map);

    state.layers.trainHeat.addTo(map);
    state.layers.predHeat.addTo(map);

    state.map = map;
    window.addEventListener("resize", function () {
      setTimeout(function () { map.invalidateSize(); }, 100);
    });
  }

  function clearLayers() {
    Object.keys(state.layers).forEach(function (k) {
      if (state.layers[k] && typeof state.layers[k].clearLayers === "function") {
        state.layers[k].clearLayers();
      }
    });
  }

  function renderEvents() {
    var layer = state.layers.events;
    layer.clearLayers();
    var events = state.events || [];
    if (!events.length) {
      if (state.beat === 1) showToast("No recent events in dataset.", "warn");
      return;
    }
    events.forEach(function (e) {
      var radius = 3 + (e.mag - 2.5) * 2;
      var fill = e.mag >= 4.5 ? "#ff5c39" : "#39d0ff";
      L.circleMarker([e.lat, e.lon], {
        radius: radius, fillColor: fill, color: "#fff",
        weight: 0.5, fillOpacity: 0.85, stroke: false,
      }).bindPopup(
        '<div class="popup"><b>M' + fmtNum(e.mag, 1) + '</b> \u00b7 ' + fmtDate(e.time) + '</div>'
      ).addTo(layer);
    });
    if (events.length > 1) {
      var group = new L.featureGroup(
        events.map(function (e) { return L.latLng(e.lat, e.lon); })
      );
      state.map.fitBounds(group.getBounds().pad(0.3), { animate: true, duration: 1.5 });
    } else {
      state.map.setView([events[0].lat, events[0].lon], 4);
    }
  }

  function renderPredictionHeat(threshold) {
    threshold = threshold == null ? state.threshold / 100 : threshold;
    var layer = state.layers.predHeat;
    layer.clearLayers();
    var features = (state.forecast && state.forecast.features) || [];
    if (!features.length) {
      if (state.beat >= 3) showToast("No forecast data on this snapshot.", "warn");
      return;
    }
    var heatData = [];
    features.forEach(function (f) {
      if (f.prob >= threshold) {
        heatData.push([f.lat_bin + GRID_SIZE / 2, f.lon_bin + GRID_SIZE / 2, f.prob]);
      }
    });
    if (heatData.length) {
      layer.setLatLngs(heatData);
      if (!state.map.hasLayer(layer)) layer.addTo(state.map);
    }
    if (heatData.length > 0) {
      var all = features.map(function (f) {
        return L.latLng(f.lat_bin, f.lon_bin);
      });
      state.map.fitBounds(L.latLngBounds(all).pad(0.2), { animate: true, duration: 1.5 });
    }
    updateThresholdMarkers(threshold);
  }

  function updateThresholdMarkers(threshold) {
    var layer = state.layers.markers;
    layer.clearLayers();
    var features = (state.forecast && state.forecast.features) || [];
    var activeIdx = -1;
    features.forEach(function (f, i) {
      if (f.prob < threshold) return;
      var color = f.prob >= 0.75 ? "#ff3c3c" : (f.prob >= 0.5 ? "#ff9e4a" : "#ffcb4a");
      L.circleMarker([f.lat_bin + GRID_SIZE / 2, f.lon_bin + GRID_SIZE / 2], {
        radius: 3 + f.prob * 6, fillColor: color, color: "#fff",
        weight: 1, fillOpacity: 0.85,
      }).bindPopup(
        '<div class="popup"><b class="mono">' + fmtNum(f.prob * 100, 1) + '%</b> risk<br/>' +
        '<span class="small">Bin ' + (i + 1) + ' \u00b7 ' + fmtPeriod(f.period) + '</span>' +
        '<div class="popup-actions"><button class="ghost-btn small" ' +
        'onclick="window.drillCell(' + i + ')">Details</button></div></div>'
      ).addTo(layer);
      if (activeIdx === -1 || f.prob > features[activeIdx].prob) activeIdx = i;
    });
    state.activeHighCellIndex = activeIdx;
  }

  function renderMarkers() {
    var layer = state.layers.markers;
    layer.clearLayers();
    if (!state.receipts || !state.receipts.markers || !state.receipts.markers.length) {
      if (state.beat >= 4) showToast("No verification markers available.", "warn");
      return;
    }
    state.receipts.markers.forEach(function (m) {
      var type = m.label.toLowerCase().indexOf("true") >= 0 ? "tp" :
                 m.label.toLowerCase().indexOf("false positive") >= 0 ? "fp" : "fn";
      L.circleMarker([m.lat, m.lon], {
        radius: 5 + m.prob * 5, fillColor: m.color, color: "#fff",
        weight: 1, fillOpacity: 0.85,
      }).bindPopup(
        '<div class="popup"><b style="color:' + m.color + '">' + m.label + '</b><br/>' +
        '<span class="small">Risk: ' + fmtNum(m.prob * 100, 1) + '%</span><br/>' +
        '<span class="small">Actual M\u22654.5: ' + (m.gt ? "Yes" : "No") + '</span></div>'
      ).addTo(layer);
    });
  }

  function renderTrainDensity() {
    var layer = state.layers.trainHeat;
    layer.clearLayers();
    var events = state.events || [];
    if (!events.length) return;
    var seen = {};
    events.forEach(function (e) {
      var key = Math.round(e.lat / GRID_SIZE) * 1000 + Math.round(e.lon / GRID_SIZE);
      if (!seen[key]) seen[key] = { lat: e.lat, lon: e.lon, count: 0 };
      seen[key].count += 1;
    });
        var heatData = [];
    Object.keys(seen).forEach(function (k) {
      var p = seen[k];
      if (p.count >= 2) heatData.push([p.lat, p.lon, p.count / events.length * 8]);
    });
    if (heatData.length) layer.setLatLngs(heatData);
  }

  /* ================================================================
   * BEAT NAVIGATION
   * ================================================================ */

  function updateRail() {
    var rail = document.getElementById("rail");
    if (!rail) return;
    rail.innerHTML = "";
    BEATS.forEach(function (beat, i) {
      var dot = el("span", {
        className: "dot" + (i + 1 <= state.beat ? " done" : "") +
          (i + 1 === state.beat ? " active" : ""),
      });
      rail.appendChild(dot);
      if (i < BEATS.length - 1) rail.appendChild(el("span", { className: "link" }));
    });
  }

  function updateStageLabel() {
    setText("stage-label", "Tour \u00b7 " + state.beat + " / " + BEATS.length);
  }

  function updateNav() {
    var back = document.getElementById("back-btn");
    var cta = document.getElementById("cta-btn");
    back.disabled = state.beat === 1;
    if (state.beat === BEATS.length) {
      cta.innerHTML = "\u21bb Refresh live data";
      cta.onclick = function () { refreshAndToast(); };
    } else {
      cta.innerHTML = BEATS[state.beat - 1].cta;
      cta.onclick = function () { advanceBeat(); };
    }
  }

  function setBeat(n) {
    state.beat = n;
    renderBeat(n);
    updateRail();
    updateStageLabel();
    updateNav();
  }

  function advanceBeat() {
    if (state.beat < BEATS.length) setBeat(state.beat + 1);
  }

  function backBeat() {
    if (state.beat > 1) setBeat(state.beat - 1);
  }

  /* ================================================================
   * BEAT RENDERING
   * ================================================================ */

  function renderBeat(n) {
    clearLayers();
    var beat = BEATS[n - 1];
    if (!beat) return;

    setText("beatnum", "BEAT " + n + " / " + BEATS.length);
    setText("beattag", beat.tag);
    setHTML("beathead", beat.head);
    setHTML("beathint", '<p class="hint">' + beat.hint + "</p>");
    setHTML("signals", "");
    setHTML("beatlegend", "");

    var dash = document.getElementById("dash");
    var legend = document.getElementById("maplegend");
    dash.hidden = n < BEATS.length;
    legend.hidden = n < BEATS.length;

    if (n === 1) {
      renderEvents();
      renderTrainDensity();
      renderBeat1Signals();
      renderBeat1Legend();
    } else if (n === 2) {
      renderPredictionHeat(state.threshold / 100);
      renderBeat2Signals();
      renderBeat2Legend();
      if (state.activeHighCellIndex != null && state.activeHighCellIndex >= 0) {
        var f = state.forecast.features[state.activeHighCellIndex];
        if (f) {
          state.map.setView([f.lat_bin + GRID_SIZE / 2, f.lon_bin + GRID_SIZE / 2], 5,
            { animate: true, duration: 1.5 });
        }
      }
    } else if (n === 3) {
      renderPredictionHeat(state.threshold / 100);
      renderBeat3Signals();
      renderBeat3Legend();
    } else if (n === 4) {
      renderPredictionHeat(state.threshold / 100);
      renderMarkers();
      renderBeat4Signals();
      renderBeat4Legend();
    } else if (n === 5) {
      renderPredictionHeat(state.threshold / 100);
      renderMarkers();
      renderDashboard();
      renderBeat5Signals();
            renderBeat5Legend();
    }
  }

  /* ---- Per-beat signal/legend renderers ---- */

  function renderSignals(container, items) {
    if (!container) return;
    container.innerHTML = "";
    items.forEach(function (item) {
      container.appendChild(el("div", { className: "sig" }, [
        el("span", { className: "k", html: item.k }),
        el("span", { className: "v " + (item.cls || ""), html: item.v }),
      ]));
    });
  }

  function renderLegend(container, items) {
    if (!container) return;
    container.innerHTML = "";
    items.forEach(function (item) {
      container.appendChild(el("span", { className: "item" }, [
        el("span", { className: "sw", style: "background:" + item.color }),
        el("span", { text: item.label }),
      ]));
    });
  }

  function renderBeat1Signals() {
    var events = state.events || [];
    var mags = events.length ? events.map(function (e) { return e.mag; }) : [0, 0];
    renderSignals(document.getElementById("signals"), [
      { k: "Catalog events", v: events.length + " recent", cls: "up" },
      { k: "Magnitude range", v: fmtNum(Math.min.apply(null, mags), 1) +
        " \u2013 M" + fmtNum(Math.max.apply(null, mags), 1), cls: "acc" },
      { k: "Forecast window", v: fmtPeriod(state.forecast &&
        state.forecast.forecast_period), cls: "acc" },
      { k: "Horizon", v: "30-day precursor", cls: "acc" },
        ]);
  }

  function renderBeat1Legend() {
    renderLegend(document.getElementById("beatlegend"), [
      { color: "#ff5c39", label: "M\u22654.5 (target)" },
      { color: "#39d0ff", label: "M2.5\u20134.5" },
    ]);
  }

  function renderBeat2Signals() {
    var features = (state.forecast && state.forecast.features) || [];
    var best = null;
    var bestIdx = -1;
    features.forEach(function (f, i) {
      if (!best || f.prob > best.prob) { best = f; bestIdx = i; }
    });
    state.activeHighCellIndex = bestIdx;
    renderSignals(document.getElementById("signals"), [
      { k: "Peak risk", v: best ? fmtNum(best.prob * 100, 1) + "%" : "\u2014", cls: "up" },
      { k: "Location", v: best ? fmtLocation(best) : "\u2014", cls: "acc" },
      { k: "Period", v: best ? fmtPeriod(best.period) : "\u2014", cls: "acc" },
      { k: "Forecast horizon", v: (state.model && state.model.time_step_days || 7) + " days", cls: "acc" },
    ]);
  }

  function renderBeat2Legend() {
    renderLegend(document.getElementById("beatlegend"), [
      { color: "#ff9e4a", label: "elevated risk" },
      { color: "#ff3c3c", label: "peak risk" },
    ]);
  }

  function renderBeat3Signals() {
    var features = (state.forecast && state.forecast.features) || [];
    var active = features.filter(function (f) { return f.prob >= state.threshold / 100; });
    var peak = features.reduce(function (a, b) {
      return (a && a.prob >= b.prob) ? a : b;
    }, null);
    renderSignals(document.getElementById("signals"), [
      { k: "Peak risk", v: peak ? fmtNum(peak.prob * 100, 1) + "%" : "\u2014", cls: "up" },
      { k: "Alert cells", v: active.length, cls: "warn" },
      { k: "Coverage", v: features.length + " cells", cls: "acc" },
      { k: "Horizon", v: (state.model && state.model.time_step_days || 7) + " d", cls: "acc" },
    ]);
  }

  function renderBeat3Legend() {
    var thr = state.threshold;
    renderLegend(document.getElementById("beatlegend"), [
      { color: "#ff9e4a", label: "predicted \u2265 " + thr + "%" },
      { color: "#ffb02e", label: "pending verify" },
      { color: "#39d0ff", label: "this cell" },
    ]);
  }

  function renderBeat4Signals() {
    var r = state.receipts || {};
    var c = r.confusion || {};
    renderSignals(document.getElementById("signals"), [
      { k: "Accuracy", v: fmtNum(r.test_acc * 100, 1) + "%", cls: "acc" },
      { k: "Train/Val gap", v: fmtNum(r.train_val_gap, 2),
        cls: Math.abs(r.train_val_gap) <= 0.05 ? "up" : "warn" },
      { k: "True Positives", v: c.tp || 0, cls: "up" },
      { k: "False Positives", v: c.fp || 0, cls: "warn" },
      { k: "False Negatives", v: c.fn || 0,
        cls: c.fn === 0 ? "up" : "bad" },
      { k: "Hit rate", v: fmtNum(r.hit_rate_pct, 1) + "%", cls: "acc" },
    ]);
  }

  function renderBeat4Legend() {
    renderLegend(document.getElementById("beatlegend"), [
      { color: MARKER_TP, label: "True Positive" },
      { color: MARKER_FP, label: "False Positive" },
      { color: MARKER_FN, label: "False Negative" },
    ]);
  }

  function renderBeat5Signals() {
    var features = (state.forecast && state.forecast.features) || [];
    var above = features.filter(function (f) { return f.prob >= state.threshold / 100; });
    var items = [
      { k: "Threshold", v: state.threshold + "%", cls: "acc" },
      { k: "Cells above", v: above.length, cls: "warn" },
      { k: "Model", v: (state.model && state.model.model_version) || "\u2014", cls: "acc" },
    ];
    if (state.history && state.history.length > 1) {
      items.push({ k: "History", v: state.history.length + " snapshots", cls: "acc" });
    }
    renderSignals(document.getElementById("signals"), items);
  }

  function renderBeat5Legend() {
    renderLegend(document.getElementById("beatlegend"), [
      { color: "#ff9e4a", label: "\u2265 " + state.threshold + "%" },
      { color: MARKER_TP, label: "verified TP" },
      { color: MARKER_FP, label: "verified FP" },
      { color: MARKER_FN, label: "verified FN" },
    ]);
  }

  /* ================================================================
   * DASHBOARD
   * ================================================================ */

  function fmtLocation(f) {
    var lat = f.lat_bin != null ? f.lat_bin : f.lat;
    var lon = f.lon_bin != null ? f.lon_bin : f.lon;
    if (lat == null || lon == null) {
      return "Bin " + ((f.period || "").slice(0, 10) || "?");
    }
    var ns = lat >= 0 ? "N" : "S";
    var ew = lon >= 0 ? "E" : "W";
    return Math.abs(lat).toFixed(0) + "\u00b0" + ns + " \u00b7 " +
      Math.abs(lon).toFixed(0) + "\u00b0" + ew;
  }

  function renderChips() {
    var model = state.model || {};
    var forecast = state.forecast || {};
    var mc = document.getElementById("chip-model");
    if (mc) mc.innerHTML = "<b>Model</b> \u2014 " + (model.model_version || "&mdash;");
    var wc = document.getElementById("chip-window");
    if (wc) wc.innerHTML = "<b>Window</b> \u2014 " +
      (fmtPeriod(forecast.forecast_period) || "&mdash;");
    var modeChip = document.getElementById("chip-mode");
    if (modeChip) {
      modeChip.className = state.live ? "chip live" : "chip baked";
      modeChip.innerHTML = state.live
        ? '<span class="pulse"></span> Live'
        : '<span class="pulse"></span> Baked';
    }
  }

  function renderScorecard() {
    var sc = document.getElementById("scorecard");
    if (!sc) return;
    sc.innerHTML = "";
    var r = state.receipts || {};
    var c = r.confusion || {};
    var items = [
      { k: "Acc", v: fmtNum(r.test_acc * 100, 1), dir: "acc" },
      { k: "t/v gap", v: fmtNum(r.train_val_gap, 2),
        dir: Math.abs(r.train_val_gap || 0) <= 0.05 ? "up" : "warn" },
      { k: "TP", v: c.tp || 0, dir: "up" },
      { k: "FP", v: c.fp || 0, dir: c.fp > (c.tp || 0) ? "warn" : "up" },
      { k: "FN", v: c.fn || 0, dir: c.fn === 0 ? "up" : "bad" },
      { k: "AUC", v: fmtNum(r.test_auc, 3), dir: "acc" },
    ];
    items.forEach(function (item) {
      sc.appendChild(el("div", { className: "stat" }, [
        el("span", { className: "k", text: item.k }),
        el("span", { className: "v " + (item.dir || ""), html: item.v }),
      ]));
    });
  }

  function renderTopList() {
    var list = document.getElementById("toplist");
    if (!list) return;
    list.innerHTML = "";
    var features = (state.forecast && state.forecast.features) || [];
    if (!features.length) {
      list.appendChild(el("div", { className: "trow", text: "No forecast data" }));
      return;
    }
    var sorted = features.slice()
      .sort(function (a, b) { return b.prob - a.prob; })
      .slice(0, 10);
    sorted.forEach(function (f, i) {
      var idx = features.indexOf(f);
      list.appendChild(el("div", { className: "trow",
        onclick: function () { window.drillCell(idx); }
      }, [
        el("span", { className: "r", html: (i + 1) + "." }),
        el("span", { className: "nm", html: fmtLocation(f) }),
        el("div", { className: "bar" }, [
          el("i", { style: "width:" + (f.prob * 100) + "%" }),
        ]),
        el("span", { className: "pv", text: fmtNum(f.prob * 100, 1) + "%" }),
      ]));
    });
  }

  function renderDashboard() {
    renderScorecard();
    renderTopList();
    renderChips();
  }

  /* ================================================================
   * INTERACTIONS
   * ================================================================ */

  function setupInteractions() {
    // Back button
    var backBtn = document.getElementById("back-btn");
    if (backBtn) backBtn.onclick = backBeat;

    // Threshold slider
    var thr = document.getElementById("thr");
    var thrVal = document.getElementById("thr-val");
    if (thr) {
      thr.oninput = function () {
        state.threshold = parseInt(thr.value, 10);
        if (thrVal) thrVal.textContent = state.threshold + "%";
        renderPredictionHeat(state.threshold / 100);
        if (state.beat === 5) {
          renderDashboard();
          renderBeat5Legend();
        } else if (state.beat === 3) {
          renderBeat3Signals();
          renderBeat3Legend();
        } else if (state.beat === 4) {
          renderBeat5Legend();
        }
      };
    }

    // Refresh button
    var refresh = document.getElementById("refresh-btn");
    if (refresh) refresh.onclick = refreshAndToast;
  }

  /* ---- Drill-down ---- */

  window.drillCell = function (idx) {
    var features = (state.forecast && state.forecast.features) || [];
    var f = features[idx];
    if (!f) return;

    var prob = f.prob;
    var level = prob >= 0.75 ? "high" : (prob >= 0.5 ? "elevated" : "moderate");

    var detail = state.model || {};
    var featCols = detail.feature_cols ||
      ["lat_bin", "lon_bin", "event_count", "mean_depth", "mean_gap", "mean_sig"];

    var html = '<div class="drill-popup">' +
      '<h3>' + fmtLocation(f) + '</h3>' +
      '<div class="stat"><span class="k">Risk</span><span class="v ' + level + '">' +
        fmtNum(prob * 100, 1) + '%</span></div>' +
      '<div class="stat"><span class="k">Period</span><span class="v acc">' +
      fmtPeriod(f.period) + '</span></div>' +
      '<div class="stat"><span class="k">Bin size</span><span class="v">' +
      (state.forecast.grid_size_deg || GRID_SIZE) + '&#176;</span></div>' +
      '<div class="stat"><span class="k">Verifiable</span><span class="v">' +
      fmtDate(state.forecast.verifiable_after) + '</span></div>' +
      '<hr class="drill-hr">' +
      '<div class="small muted">This forecast was published ' +
      fmtDate(state.forecast.generated_at) + '. It is verifiable only after ' +
      fmtDate(state.forecast.verifiable_after) + ' \u2014 the label-latency boundary.</div>' +
      '</div>';

    state.map.closePopup();
    L.popup({ className: "drill-popup-card" })
      .setLatLng([f.lat_bin + GRID_SIZE / 2, f.lon_bin + GRID_SIZE / 2])
      .setContent(html)
      .openOn(state.map);
    state.map.setView([f.lat_bin + GRID_SIZE / 2, f.lon_bin + GRID_SIZE / 2], 5,
      { animate: true, duration: 1 });
  };

  /* ---- Live refresh ---- */

  async function refreshAndToast() {
    showToast("Refreshing live data\u2026", "info");
    try {
      if (await probeLive()) {
        state.live = true;
        var upgraded = await fetchLive();
        state.forecast = upgraded.forecast;
        state.model = upgraded.model;
        showToast("Live data updated.", "info");
      } else {
        showToast("Backend not reachable \u2014 showing baked data.", "warn");
        state.live = false;
      }
      renderChips();
      if (state.beat === 5) renderDashboard();
      renderBeat(state.beat);
    } catch (e) {
            showToast("Live refresh failed: " + e.message, "err");
    }
  }

  /* ================================================================
   * EMPTY STATE
   * ================================================================ */

  function renderEmptyStatePlaceholder() {
    if (!state.map) {
      // Map not initialized — try anyway
      document.getElementById("map").innerHTML = renderEmptyHTML();
    } else {
      state.map.eachLayer(function (layer) {
        state.map.removeLayer(layer);
      });
      L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
        maxZoom: 19,
        attribution: "&copy; CartoDB | USGS | SeismicML",
      }).addTo(state.map);

      L.marker([14, -30], {
        icon: L.divIcon({
          className: "empty-icon-wrap",
          html: renderEmptyHTML(),
          iconSize: [400, 300],
          iconAnchor: [200, 150],
        }),
      }).addTo(state.map);
    }

    document.getElementById("maplegend").hidden = true;
    document.getElementById("dash").hidden = true;
    document.getElementById("back-btn").disabled = true;
    document.getElementById("cta-btn").style.display = "none";

    showToast("No data available", "err");
  }

  function renderEmptyHTML() {
    return '<div class="empty-state">' +
      '<div class="empty-icon">\u25ed</div>' +
      '<h2>No Data Available</h2>' +
      '<p>This is normally populated by <code>baked.js</code> or a live ' +
      'API at <code>/</code> on <code>localhost:8000</code>.</p>' +
      '<p>To generate data: ' +
      '<code>USGS_LIVE_FETCH=1 python -m src.serve.bake</code></p>' +
      '<button class="cta" onclick="location.reload()">Retry</button>' +
      '</div>';
  }

  /* ================================================================
   * INIT
   * ================================================================ */

  async function init() {
    // Initialize map
    try {
      initMap();
    } catch (e) {
      console.error("Map init failed:", e);
      document.getElementById("map").innerHTML =
        '<div style="color:var(--ink);padding:20px">Map failed to load: ' +
        e.message + "</div>";
    }

    // Load data
    var ok = await initData();
    if (!ok) return;

    // Populate chips
    renderChips();

    // Set up interactions
    setupInteractions();

    // Sync threshold slider
    var thr = document.getElementById("thr");
    if (thr) thr.value = state.threshold;
    var thrVal = document.getElementById("thr-val");
    if (thrVal) thrVal.textContent = state.threshold + "%";

    // Render initial beat
    setBeat(1);

    // Footer latency note
    var footerNote = document.getElementById("latency-note");
    if (footerNote) {
      var fc = state.forecast || {};
      var status = state.live ? " LIVE FEED" : " baked";
      var pending = "";
      if (fc.verifiable_after) {
        pending = " Published " + (fmtDate(fc.generated_at) || fmtPeriod(fc.forecast_period));
        pending += " \u2014 verifiable after " + fmtDate(fc.verifiable_after);
      }
      footerNote.innerHTML =
        '<span class="mono">' + status + "</span>" +
        (pending ? ' <span class="mono">' + pending + "</span>" : "");
    }

    // Toast
    if (state.live) {
      showToast("Live data connected. Serving fresh forecast.", "info");
    }
  }

  // Run
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();