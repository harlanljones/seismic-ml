/* SeismicML live forecast frontend (no build step).
 * Reads the inlined FORECAST GeoJSON (or re-fetches /forecast) and renders the
 * three layers mirroring src/viz/map.py semantics:
 *   - predicted-risk heatmap  (L.heatLayer, radius/blur/opacity/gradient)
 *   - alert markers for prob >= 0.5 (pending alert, label not yet known)
 *   - (placeholder) train-density heatmap if present
 */
(function () {
  "use strict";

  var fc = window.FORECAST || { type: "FeatureCollection", features: [] };
  var features = fc.features || [];

  var latSum = 0, lonSum = 0, n = 0;
  features.forEach(function (f) {
    var c = f.geometry.coordinates; // [lon, lat]
    latSum += c[1]; lonSum += c[0]; n++;
  });
  var center = n ? [latSum / n, lonSum / n] : [20, 0];

  var map = L.map("map").setView(center, 3);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: "&copy; OpenStreetMap contributors"
  }).addTo(map);

  // Predicted-risk heatmap: mirrors PRED_HEAT in map_server.py
  var heatPoints = features.map(function (f) {
    var c = f.geometry.coordinates;
    return [c[1], c[0], f.properties.prob]; // [lat, lon, prob]
  });
  if (heatPoints.length && L.heatLayer) {
    L.heatLayer(heatPoints, {
      radius: 16,
      blur: 14,
      minOpacity: 0.4,
      gradient: { 0.4: "orange", 1.0: "red" }
    }).addTo(map);
  }

  // Alert markers for prob >= 0.5. Until outcomes are verifiable the live view
  // renders them as "pending alerts" (orange) — see LIVE_APP_ROADMAP §5.3.1.
  features.forEach(function (f) {
    if (f.properties.prob >= 0.5) {
      var c = f.geometry.coordinates;
      L.circleMarker([c[1], c[0]], {
        radius: 6,
        color: "#FFA500",
        fillColor: "#FFA500",
        fillOpacity: 0.8,
        weight: 1
      }).addTo(map).bindPopup(
        "<b>Pending Alert</b><br>Predicted Risk: " +
        (f.properties.prob * 100).toFixed(1) + "%"
      );
    }
  });
})();
