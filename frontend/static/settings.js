// Settings page — MQTT credential form + test button + live status.

const els = {
  enabled:    document.getElementById("mq-enabled"),
  broker:     document.getElementById("mq-broker"),
  port:       document.getElementById("mq-port"),
  transport:  document.getElementById("mq-transport"),
  wsRow:      document.getElementById("mq-ws-row"),
  wsPath:     document.getElementById("mq-ws-path"),
  tls:        document.getElementById("mq-tls"),
  username:   document.getElementById("mq-username"),
  password:   document.getElementById("mq-password"),
  clientId:   document.getElementById("mq-client-id"),
  topicPref:  document.getElementById("mq-topic-prefix"),
  haDisc:     document.getElementById("mq-ha-discovery"),
  haPrefix:   document.getElementById("mq-ha-prefix"),
  deviceName: document.getElementById("mq-device-name"),
  status:     document.getElementById("set-status"),
  // Status block
  stPaho:     document.getElementById("st-paho"),
  stConn:     document.getElementById("st-conn"),
  stLast:     document.getElementById("st-last"),
  stMsg:      document.getElementById("st-msg"),
};

const SENTINEL_PASSWORD = "__keep__";

let _loadedPassword = "";  // whatever the form was loaded with (sentinel or empty)

function showStatus(text, kind = "neutral") {
  els.status.textContent = text;
  els.status.className = "set-status status-" + kind;
}

function applyTransport() {
  els.wsRow.hidden = els.transport.value !== "websockets";
}

els.transport.addEventListener("change", applyTransport);

async function loadSettings() {
  try {
    const r = await fetch("/api/settings", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const s = await r.json();
    const m = s.mqtt || {};
    els.enabled.checked   = !!m.enabled;
    els.broker.value      = m.broker || "";
    els.port.value        = m.port || 1883;
    els.transport.value   = m.transport || "tcp";
    els.wsPath.value      = m.ws_path || "/mqtt";
    els.tls.checked       = !!m.use_tls;
    els.username.value    = m.username || "";
    els.password.value    = m.password === SENTINEL_PASSWORD ? "" : (m.password || "");
    els.password.placeholder = m.password === SENTINEL_PASSWORD ? "(unchanged — type to overwrite)" : "";
    _loadedPassword       = m.password === SENTINEL_PASSWORD ? SENTINEL_PASSWORD : "";
    els.clientId.value    = m.client_id || "primeanalyze-hub";
    els.topicPref.value   = m.topic_prefix || "primeanalyze";
    els.haDisc.checked    = !!m.ha_discovery;
    els.haPrefix.value    = m.ha_discovery_prefix || "homeassistant";
    els.deviceName.value  = m.device_name || "PrimeAnalyze Baby Monitor";
    applyTransport();
  } catch (e) {
    showStatus("Failed to load settings: " + e.message, "error");
  }
}

function formToMqtt() {
  const pw = els.password.value;
  return {
    enabled:             els.enabled.checked,
    broker:              els.broker.value.trim(),
    port:                Number(els.port.value) || 1883,
    transport:           els.transport.value,
    ws_path:             els.wsPath.value.trim() || "/mqtt",
    use_tls:             els.tls.checked,
    username:            els.username.value.trim(),
    // If the user didn't touch the password field, keep the stored one.
    password:            pw ? pw : _loadedPassword,
    client_id:           els.clientId.value.trim() || "primeanalyze-hub",
    topic_prefix:        els.topicPref.value.trim() || "primeanalyze",
    ha_discovery:        els.haDisc.checked,
    ha_discovery_prefix: els.haPrefix.value.trim() || "homeassistant",
    device_name:         els.deviceName.value.trim() || "PrimeAnalyze Baby Monitor",
  };
}

document.getElementById("mq-test").addEventListener("click", async () => {
  showStatus("Testing…", "pending");
  try {
    const r = await fetch("/api/settings/mqtt/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mqtt: formToMqtt() }),
    });
    const body = await r.json();
    if (body.ok) {
      showStatus("✓ Connected — broker accepted credentials", "ok");
    } else {
      showStatus("✗ " + (body.message || "test failed"), "error");
    }
  } catch (e) {
    showStatus("✗ Request failed: " + e.message, "error");
  }
});

document.getElementById("mq-save").addEventListener("click", async () => {
  showStatus("Saving…", "pending");
  try {
    const r = await fetch("/api/settings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mqtt: formToMqtt() }),
    });
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || `HTTP ${r.status}`);
    showStatus("✓ Saved. Reloading status…", "ok");
    // Re-fetch so the password sentinel updates correctly.
    await loadSettings();
    await refreshStatus();
  } catch (e) {
    showStatus("✗ Save failed: " + e.message, "error");
  }
});

async function refreshStatus() {
  try {
    const r = await fetch("/api/settings/mqtt/status", { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const st = await r.json();
    els.stPaho.textContent = st.have_paho ? "yes" : "NO — pip install paho-mqtt";
    els.stPaho.className = "set-stat-v " + (st.have_paho ? "stat-ok" : "stat-error");
    if (st.connected) {
      els.stConn.textContent = "yes";
      els.stConn.className = "set-stat-v stat-ok";
    } else if (st.enabled) {
      els.stConn.textContent = "no — " + (st.last_status || "disconnected");
      els.stConn.className = "set-stat-v stat-warn";
    } else {
      els.stConn.textContent = "disabled";
      els.stConn.className = "set-stat-v stat-muted";
    }
    if (st.last_state == null) {
      els.stLast.textContent = "—";
    } else {
      els.stLast.textContent = st.last_state ? "ON (baby_in_crib)" : "OFF";
    }
    els.stMsg.textContent = st.last_status || "—";
  } catch (e) {
    els.stConn.textContent = "?";
    els.stMsg.textContent = "Status fetch failed: " + e.message;
  }
}

loadSettings().then(refreshStatus);
setInterval(refreshStatus, 5000);
