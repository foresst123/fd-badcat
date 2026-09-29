(() => {
  "use strict";

  const TARGET_SAMPLE_RATE = 16_000;
  const FRAME_SAMPLES = 256;
  const MAX_BUFFERED_BYTES = 512 * 1024;
  // Covers measured VieNeu CPU INT8 jitter without buffering a full sentence.
  const STREAM_START_DELAY_SECONDS = 0.2;
  const STREAM_RECOVERY_DELAY_SECONDS = 0.02;
  const TRACE_EVENT_LIMIT = 300;
  const TRACE_VISIBLE_LIMIT = 140;

  const elements = {
    start: document.querySelector("#startButton"),
    stop: document.querySelector("#stopButton"),
    clear: document.querySelector("#clearButton"),
    language: document.querySelector("#languageSelect"),
    statusDot: document.querySelector("#statusDot"),
    connection: document.querySelector("#connectionText"),
    activity: document.querySelector("#activityText"),
    meter: document.querySelector("#meterFill"),
    messages: document.querySelector("#messages"),
    empty: document.querySelector("#emptyMessage"),
    log: document.querySelector("#eventLog"),
    traceIdentity: document.querySelector("#traceIdentity"),
    traceState: document.querySelector("#traceState"),
    traceTurnGeneration: document.querySelector("#traceTurnGeneration"),
    traceDecision: document.querySelector("#traceDecision"),
    traceDecisionMeta: document.querySelector("#traceDecisionMeta"),
    traceDecisionQueue: document.querySelector("#traceDecisionQueue"),
    traceAsrQueue: document.querySelector("#traceAsrQueue"),
    traceTtsQueue: document.querySelector("#traceTtsQueue"),
    traceDecisionLatency: document.querySelector("#traceDecisionLatency"),
    traceVadLatency: document.querySelector("#traceVadLatency"),
    traceTtfa: document.querySelector("#traceTtfa"),
    traceCancelLatency: document.querySelector("#traceCancelLatency"),
    traceAlerts: document.querySelector("#traceAlerts"),
    traceFilter: document.querySelector("#traceFilter"),
    tracePause: document.querySelector("#tracePauseButton"),
    traceExport: document.querySelector("#traceExportButton"),
    traceClear: document.querySelector("#traceClearButton"),
    traceTimeline: document.querySelector("#traceTimeline"),
    traceStages: ["Input", "Vad", "Mllm", "Tts", "Browser"].map(
      (name) => document.querySelector(`#traceStage${name}`),
    ),
  };

  const state = {
    websocket: null,
    mediaStream: null,
    audioContext: null,
    sourceNode: null,
    captureNode: null,
    silentGain: null,
    resampler: null,
    frame: new Float32Array(FRAME_SAMPLES),
    frameOffset: 0,
    currentPlayback: null,
    playbackQueue: [],
    pendingTts: null,
    pcmStream: null,
    assistantDrafts: new Map(),
    playing: false,
    stopping: false,
    expectedClose: false,
    trace: null,
  };

  class StreamingResampler {
    constructor(sourceRate, targetRate) {
      this.ratio = sourceRate / targetRate;
      this.buffer = new Float32Array(0);
      this.position = 0;
    }

    push(samples) {
      const joined = new Float32Array(this.buffer.length + samples.length);
      joined.set(this.buffer);
      joined.set(samples, this.buffer.length);

      const output = [];
      while (this.position + 1 < joined.length) {
        const index = Math.floor(this.position);
        const fraction = this.position - index;
        output.push(
          joined[index] + (joined[index + 1] - joined[index]) * fraction,
        );
        this.position += this.ratio;
      }

      const consumed = Math.min(
        Math.floor(this.position),
        Math.max(0, joined.length - 1),
      );
      this.buffer = joined.slice(consumed);
      this.position -= consumed;
      return Float32Array.from(output);
    }
  }

  function websocketUrl() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${protocol}//${window.location.host}/realtime`;
  }

  function setConnection(label, kind = "") {
    elements.connection.textContent = label;
    elements.statusDot.className = `status-dot ${kind}`.trim();
  }

  function setActivity(label) {
    elements.activity.textContent = label;
  }

  function logEvent(label, data = null) {
    const time = new Date().toLocaleTimeString("vi-VN", { hour12: false });
    const detail = data ? ` ${JSON.stringify(data)}` : "";
    const lines = `${elements.log.textContent}[${time}] ${label}${detail}\n`
      .split("\n")
      .filter(Boolean)
      .slice(-80);
    elements.log.textContent = `${lines.join("\n")}\n`;
    elements.log.scrollTop = elements.log.scrollHeight;
  }

  function newTraceState() {
    return {
      events: [], alerts: [], paused: false, startedAt: performance.now(),
      traceId: null, tracePath: null, intervalMs: 1000,
      currentState: "IDLE", turn: null, generation: null, flag: null,
      decisionSource: null, rawDecision: null, parseValid: null,
      decisionQueue: 0, maxDecisionQueue: 0,
      asrQueue: 0, maxAsrQueue: 0, ttsQueue: 0, maxTtsQueue: 0,
      decisionLatency: null, vadLatency: null,
      ttfa: null, cancelLatency: null, vadEnds: new Map(),
      responses: new Map(), lastS2L: null,
    };
  }

  function numeric(value) {
    const number = Number(value);
    return Number.isFinite(number) ? number : null;
  }

  function seconds(value) {
    const number = numeric(value);
    return number === null ? "–" : `${number.toFixed(3)}s`;
  }

  function eventElapsed(data) {
    return numeric(data.timestamp)
      ?? ((performance.now() - state.trace.startedAt) / 1000);
  }

  function eventCategory(event) {
    if (/error|failed|fallback/.test(event)) return "error";
    if (/^(paper_|duplex_decision|live_prefill|vad_segment_(tick|transition|replayed|replaced|superseded|continue_timeout))/.test(event)) return "decision";
    if (/^(vad_|asr_|tts_|playback_)/.test(event)) return "audio";
    return "lifecycle";
  }

  function eventTone(event, data) {
    const infer = numeric(data.infer_time);
    if (/error|failed/.test(event)) return "error";
    if (
      /fallback|replayed|coalesced|decision_retry|continue_timeout_fired/.test(event)
      || Number(data.queue_depth || 0) >= 3
      || (infer !== null && infer * 1000 > state.trace.intervalMs * 1.25)
      || (data.flag === "ks" && data.vad_end === true)
    ) return "warning";
    if (
      ["l2s", "s2l"].includes(data.flag)
      || ["tts_first_audio", "stop_audio", "response_complete"].includes(event)
    ) return "success";
    return "normal";
  }

  function eventSummary(event, data) {
    const labels = {
      trace_ready: "Trace JSONL sẵn sàng",
      paper_unit_ready: "Paper Unit sẵn sàng",
      vad_segment_ready: "VAD Segment + Streaming ASR sẵn sàng",
      vad_segment_finalized: `Chốt Segment ${data.segment ?? "?"}`,
      vad_segment_resumed: `Nối tiếp Segment ${data.segment ?? "?"}`,
      vad_segment_tick: `Segment ${data.segment ?? "?"} → ${String(data.flag || data.decision || "?").toUpperCase()}`,
      vad_segment_decision_fallback: "Fallback decision theo state",
      vad_segment_continue_timeout_armed: "KL \u00b7 ch\u1edd fallback 2,5 gi\u00e2y",
      vad_segment_continue_timeout_cancelled: "H\u1ee7y fallback KL",
      vad_segment_continue_timeout_skipped: "B\u1ecf fallback KL stale",
      vad_segment_continue_timeout_fired: "Fallback 2,5 gi\u00e2y \u2192 L2S",
      vad_segment_transition: `Chuyển trạng thái → ${String(data.flag || "?").toUpperCase()}`,
      vad_segment_replaced: `Thay Segment ${data.dropped_segment ?? "?"} bằng ${data.latest_segment ?? "?"}`,
      vad_segment_superseded: `Bỏ transition Segment ${data.segment ?? "?"}; đã có segment mới`,
      vad_segment_replayed: `Đánh giá lại Segment ${data.segment ?? "?"}`,
      asr_stream_started: "Mở streaming ASR",
      asr_partial: "ASR partial",
      asr_stream_final: "ASR final",
      asr_context_cached: "ASR lưu cho quyết định kế tiếp",
      vad_start: data.state === "SPEAK" ? "Phát hiện chen ngang" : "Mở vùng nói",
      vad_done: "Đóng vùng nói",
      paper_barge_in_started: "Mở barge-in candidate",
      paper_barge_in_merged: "Ghép vùng VAD chen ngang",
      paper_barge_in_finalized: "Chốt barge-in utterance",
      paper_units_coalesced: "Ưu tiên Unit hoàn chỉnh",
      paper_unit_tick: `Unit ${data.unit ?? "?"} → ${String(data.flag || data.decision || "?").toUpperCase()}`,
      paper_decision_retry: `Retry → ${String(data.flag || data.decision || "?").toUpperCase()}`,
      paper_unit_replayed: `Replay Unit ${data.unit ?? "?"}`,
      duplex_decision: `Controller → ${String(data.flag || "?").toUpperCase()}`,
      asr_started: "ASR bắt đầu", asr_completed: "ASR hoàn tất",
      asr_done: "Transcript hội thoại",
      response_started: "Bắt đầu response", llm_done: "MLLM hoàn tất",
      tts_stream_start: "Mở PCM stream", tts_first_audio: "PCM đầu tiên",
      tts_phrase_ready: "Hybrid phrase sẵn sàng cho TTS",
      tts_stream_end: "TTS server hoàn tất",
      response_server_complete: "Server xong; browser còn phát",
      playback_acknowledged: "Browser xác nhận trạng thái playback",
      playback_ack_ignored: "Bỏ ACK playback cũ",
      playback_ack_timeout: "Hết hạn chờ playback ACK",
      generation_cancel_requested: "Yêu cầu hủy generation",
      generation_cancel_finished: "Đã hủy generation",
      generation_cancel_noop: "Cleanup generation: không có tác vụ active",
      stop_audio: "Dừng audio browser", response_complete: "Response hoàn tất",
    };
    return labels[event] || event;
  }

  function eventDetail(data) {
    const value = data.message ?? data.transcript ?? data.asr_context
      ?? data.reason ?? data.content;
    if (value === undefined || value === null) return "";
    const text = typeof value === "string" ? value : JSON.stringify(value);
    return text.length > 180 ? `${text.slice(0, 180)}…` : text;
  }

  function warnTrace(key, tone, message) {
    const alert = state.trace.alerts.find((item) => item.key === key);
    if (alert) {
      alert.count += 1;
      alert.tone = tone;
      alert.message = message;
      alert.updatedAt = Date.now();
    } else {
      state.trace.alerts.push({ key, tone, message, count: 1, updatedAt: Date.now() });
    }
    state.trace.alerts.sort((a, b) => b.updatedAt - a.updatedAt);
    state.trace.alerts.length = Math.min(state.trace.alerts.length, 8);
  }

  function activateStage(index, tone = "normal") {
    elements.traceStages.forEach((stage, current) => {
      stage.className = "trace-stage";
      if (current < index) stage.classList.add("success");
      if (current === index) {
        stage.classList.add("active");
        if (!["normal", "success"].includes(tone)) stage.classList.add(tone);
      }
    });
  }

  function updatePipeline(event, data, tone) {
    if (["trace_ready", "paper_unit_ready", "vad_segment_ready"].includes(event)) activateStage(0);
    else if (event === "vad_start") activateStage(1, data.state === "SPEAK" ? "warning" : tone);
    else if (/^(vad_done|paper_|duplex_|asr_|response_started|llm_done)/.test(event)) activateStage(2, tone);
    else if (event === "assistant_delta") activateStage(2);
    else if (event.startsWith("tts_")) activateStage(3, tone);
    else if (/^(playback_|stop_audio|response_complete|response_server_complete|generation_cancel)/.test(event)) activateStage(4, tone);
  }

  function updateTrace(event, data, elapsed, tone) {
    const trace = state.trace;
    if (event === "trace_ready") {
      trace.traceId = data.trace_id || null;
      trace.tracePath = data.path || null;
    }
    if (event === "paper_unit_ready" && numeric(data.decision_interval_ms) !== null) {
      trace.intervalMs = Number(data.decision_interval_ms);
    }
    if (event === "vad_segment_ready" && numeric(data.endpoint_ms) !== null) {
      trace.intervalMs = Number(data.endpoint_ms);
    }
    trace.currentState = String(data.current_state ?? data.state_after ?? data.state ?? trace.currentState);
    if (data.turn !== undefined && data.turn !== null) trace.turn = data.turn;
    const generation = data.current_generation ?? data.generation;
    if (generation !== undefined && generation !== null) trace.generation = generation;
    if (data.flag) trace.flag = String(data.flag).toUpperCase();

    if (data.decision_source) {
      trace.decisionSource = data.decision_source;
      trace.rawDecision = data.raw_decision == null ? null : String(data.raw_decision);
      trace.parseValid = data.parse_valid == null ? null : Boolean(data.parse_valid);
    } else {
      if (data.raw_decision !== undefined && data.raw_decision !== null) {
        trace.rawDecision = String(data.raw_decision);
      }
      if (data.parse_valid !== undefined && data.parse_valid !== null) {
        trace.parseValid = Boolean(data.parse_valid);
      }
    }

    const decisionQueue = numeric(data.decision_queue_depth);
    if (decisionQueue !== null) {
      trace.decisionQueue = decisionQueue;
      trace.maxDecisionQueue = Math.max(trace.maxDecisionQueue, decisionQueue);
      if (decisionQueue > 1) {
        warnTrace("decision-queue", "warning", `Decision queue vượt giới hạn: ${decisionQueue}.`);
      }
    }
    const asrQueue = numeric(data.asr_frame_queue_depth);
    if (asrQueue !== null) {
      trace.asrQueue = asrQueue;
      trace.maxAsrQueue = Math.max(trace.maxAsrQueue, asrQueue);
    }
    const ttsQueue = numeric(data.tts_text_queue_depth);
    if (ttsQueue !== null) {
      trace.ttsQueue = ttsQueue;
      trace.maxTtsQueue = Math.max(trace.maxTtsQueue, ttsQueue);
    }
    if (["paper_unit_tick", "vad_segment_tick"].includes(event) && numeric(data.infer_time) !== null) {
      trace.decisionLatency = Number(data.infer_time);
      if (event === "paper_unit_tick" && trace.decisionLatency * 1000 > trace.intervalMs * 1.25) {
        warnTrace("slow", "warning", `Decision ${trace.decisionLatency.toFixed(3)}s chậm hơn chu kỳ ${(trace.intervalMs / 1000).toFixed(3)}s.`);
      }
    }
    if (event === "vad_done") trace.vadEnds.set(String(data.state || "unknown"), elapsed);
    if (event === "duplex_decision") {
      const vad = trace.vadEnds.get(String(data.state || "unknown"));
      if (vad !== undefined) trace.vadLatency = Math.max(0, elapsed - vad);
      if (data.flag === "s2l") trace.lastS2L = elapsed;
      if (data.flag === "ks" && data.vad_end === true) {
        warnTrace(`ks-${data.unit}`, "warning", `Unit ${data.unit ?? "?"} kết thúc VAD nhưng vẫn KS.`);
      }
    }
    if (event === "response_started") trace.responses.set(String(data.generation), elapsed);
    if (event === "tts_first_audio") {
      const start = trace.responses.get(String(data.generation));
      trace.ttfa = numeric(data.ttfa) ?? (start === undefined ? null : elapsed - start);
    }
    if (event === "stop_audio" && trace.lastS2L !== null) {
      trace.cancelLatency = Math.max(0, elapsed - trace.lastS2L);
    }
    if (event === "paper_unit_replayed") {
      warnTrace("replay", "warning", `Unit ${data.unit ?? "?"} stale được replay sang ${data.current_state || "state mới"}.`);
    }
    if (event === "paper_units_coalesced") {
      warnTrace("coalesce", "info", `Ưu tiên Unit ${data.final_unit ?? "?"}, bỏ ${data.dropped_units?.length || 0} Unit rời.`);
    }
    if (event === "vad_segment_replaced") {
      warnTrace("segment-replaced", "warning", `Decision bận: bỏ Segment ${data.dropped_segment ?? "?"}, giữ Segment mới nhất ${data.latest_segment ?? "?"}.`);
    }
    if (event === "vad_segment_continue_timeout_fired") {
      warnTrace(
        "listen-timeout-" + (data.segment ?? "x"),
        "warning",
        "Classifier KL; fallback \u00e9p L2S sau "
          + (data.timeout_seconds ?? 2.5) + "s.",
      );
    }
    if (tone === "error" || event.includes("fallback")) {
      warnTrace(`${event}-${data.unit ?? data.generation ?? "x"}`, tone === "error" ? "error" : "warning", `${event}: ${data.message || data.reason || "xem timeline"}`);
    }
  }

  function renderMetrics() {
    const trace = state.trace;
    elements.traceIdentity.textContent = trace.traceId ? `trace ${trace.traceId.slice(0, 12)}` : "Chưa có trace";
    elements.traceIdentity.title = trace.tracePath || "";
    elements.traceState.textContent = trace.currentState;
    elements.traceTurnGeneration.textContent = `${trace.turn ?? "–"} / ${trace.generation ?? "–"}`;
    elements.traceDecision.textContent = trace.flag || "–";
    elements.traceDecisionMeta.textContent = `${trace.decisionSource || "–"} / ${trace.rawDecision || "–"}`;
    elements.traceDecisionMeta.title = `parse_valid=${trace.parseValid === null ? "–" : trace.parseValid}`;
    elements.traceDecisionQueue.textContent = `${trace.decisionQueue} / ${trace.maxDecisionQueue}`;
    elements.traceAsrQueue.textContent = `${trace.asrQueue} / ${trace.maxAsrQueue}`;
    elements.traceTtsQueue.textContent = `${trace.ttsQueue} / ${trace.maxTtsQueue}`;
    elements.traceDecisionLatency.textContent = seconds(trace.decisionLatency);
    elements.traceVadLatency.textContent = seconds(trace.vadLatency);
    elements.traceTtfa.textContent = seconds(trace.ttfa);
    elements.traceCancelLatency.textContent = seconds(trace.cancelLatency);
  }

  function renderAlerts() {
    elements.traceAlerts.replaceChildren();
    if (!state.trace.alerts.length) {
      const empty = document.createElement("p");
      empty.className = "trace-empty";
      empty.textContent = "Chưa phát hiện cảnh báo.";
      elements.traceAlerts.append(empty);
      return;
    }
    for (const alert of state.trace.alerts) {
      const row = document.createElement("p");
      row.className = `trace-alert ${alert.tone}`;
      row.textContent = `${alert.message}${alert.count > 1 ? ` · ×${alert.count}` : ""}`;
      elements.traceAlerts.append(row);
    }
  }

  function chip(key, value) {
    const node = document.createElement("span");
    node.className = `trace-chip${key === "flag" ? ` flag-${String(value).toLowerCase()}` : ""}`;
    node.textContent = `${key}=${value}`;
    return node;
  }

  function renderTraceTimeline() {
    if (state.trace.paused) return;
    const filter = elements.traceFilter.value;
    const events = state.trace.events.filter((row) => filter === "all" || row.category === filter).slice(-TRACE_VISIBLE_LIMIT);
    elements.traceTimeline.replaceChildren();
    if (!events.length) {
      const empty = document.createElement("p");
      empty.className = "trace-empty";
      empty.textContent = "Không có sự kiện phù hợp bộ lọc.";
      elements.traceTimeline.append(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    for (const entry of events) {
      const row = document.createElement("div");
      row.className = `trace-row ${entry.tone}`;
      const time = document.createElement("time");
      time.className = "trace-time";
      time.textContent = `${entry.elapsed.toFixed(3)}s`;
      const marker = document.createElement("span");
      marker.className = "trace-marker";
      const body = document.createElement("div");
      const title = document.createElement("div");
      title.className = "trace-event-title";
      const name = document.createElement("strong");
      name.textContent = entry.event;
      const summary = document.createElement("span");
      summary.textContent = entry.summary;
      title.append(name, summary);
      body.append(title);
      const chips = document.createElement("div");
      chips.className = "trace-chips";
      for (const key of ["state", "captured_state", "state_after", "turn", "captured_turn", "unit", "segment", "generation", "raw_decision", "parsed_decision", "parse_valid", "decision_source", "fallback_reason", "decision", "flag", "decision_queue_depth", "asr_frame_queue_depth", "tts_text_queue_depth", "infer_time", "ttfa", "audio_seconds", "segments", "current_asr_used", "available_from_segment", "stage", "reason"]) {
        const value = entry.data[key];
        if (value !== undefined && value !== null && value !== "") chips.append(chip(key, value));
      }
      if (chips.childElementCount) body.append(chips);
      if (entry.detail) {
        const detail = document.createElement("p");
        detail.className = "trace-detail";
        detail.textContent = entry.detail;
        body.append(detail);
      }
      row.append(time, marker, body);
      fragment.append(row);
    }
    elements.traceTimeline.append(fragment);
    elements.traceTimeline.scrollTop = elements.traceTimeline.scrollHeight;
  }

  function recordTraceEvent(event, data = {}) {
    const elapsed = eventElapsed(data);
    const tone = eventTone(event, data);
    updateTrace(event, data, elapsed, tone);
    updatePipeline(event, data, tone);
    if (event !== "assistant_delta") {
      state.trace.events.push({ event, data: { ...data }, elapsed, tone,
        category: ["error", "warning"].includes(tone)
          ? "error"
          : eventCategory(event),
        summary: eventSummary(event, data), detail: eventDetail(data) });
      if (state.trace.events.length > TRACE_EVENT_LIMIT) {
        state.trace.events.splice(0, state.trace.events.length - TRACE_EVENT_LIMIT);
      }
    }
    renderMetrics();
    renderAlerts();
    renderTraceTimeline();
  }

  function resetTraceDashboard(keepIdentity = false) {
    const previous = state.trace || newTraceState();
    const traceId = keepIdentity ? previous.traceId : null;
    const tracePath = keepIdentity ? previous.tracePath : null;
    const intervalMs = previous.intervalMs || 1000;
    state.trace = newTraceState();
    Object.assign(state.trace, { traceId, tracePath, intervalMs });
    elements.tracePause.classList.remove("active");
    elements.tracePause.textContent = "Tạm dừng UI";
    elements.traceFilter.value = "all";
    elements.traceStages.forEach((stage) => { stage.className = "trace-stage"; });
    renderMetrics();
    renderAlerts();
    renderTraceTimeline();
  }

  function toggleTracePause() {
    state.trace.paused = !state.trace.paused;
    elements.tracePause.classList.toggle("active", state.trace.paused);
    elements.tracePause.textContent = state.trace.paused ? "Tiếp tục UI" : "Tạm dừng UI";
    if (!state.trace.paused) renderTraceTimeline();
  }

  function exportTraceJson() {
    const trace = state.trace;
    const payload = { exported_at: new Date().toISOString(), trace_id: trace.traceId,
      trace_path: trace.tracePath,
      metrics: { state: trace.currentState, turn: trace.turn,
        generation: trace.generation, flag: trace.flag,
        decision_source: trace.decisionSource, raw_decision: trace.rawDecision,
        parse_valid: trace.parseValid,
        decision_queue_depth: trace.decisionQueue,
        max_decision_queue_depth: trace.maxDecisionQueue,
        asr_frame_queue_depth: trace.asrQueue,
        max_asr_frame_queue_depth: trace.maxAsrQueue,
        tts_text_queue_depth: trace.ttsQueue,
        max_tts_text_queue_depth: trace.maxTtsQueue,
        decision_latency: trace.decisionLatency,
        vad_to_decision: trace.vadLatency, ttfa: trace.ttfa,
        s2l_to_stop_audio: trace.cancelLatency },
      alerts: trace.alerts, events: trace.events };
    const link = document.createElement("a");
    link.href = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    link.download = `fd-badcat-trace-${trace.traceId?.slice(0, 12) || "browser"}.json`;
    document.body.append(link);
    link.click();
    URL.revokeObjectURL(link.href);
    link.remove();
  }

  function messageOrder(turn, role) {
    const numericTurn = Number(turn);
    const normalizedTurn = Number.isFinite(numericTurn)
      ? numericTurn
      : Number.MAX_SAFE_INTEGER;
    return normalizedTurn * 2 + (role === "user" ? 0 : 1);
  }

  function insertMessageInTurnOrder(message, role, turn) {
    const order = messageOrder(turn, role);
    message.dataset.messageOrder = String(order);
    const existing = Array.from(elements.messages.querySelectorAll(".message"));
    const next = existing.find(
      (item) => Number(item.dataset.messageOrder) > order,
    );
    if (next) elements.messages.insertBefore(message, next);
    else elements.messages.append(message);
  }

  function addMessage(role, content, turn) {
    if (!content) return;
    elements.empty.hidden = true;
    const key = `${role}-${turn}`;
    let message = elements.messages.querySelector(`[data-message-key="${key}"]`);
    if (!message) {
      message = document.createElement("div");
      message.className = `message ${role}`;
      message.dataset.messageKey = key;
      const label = document.createElement("span");
      label.className = "message-label";
      label.textContent = role === "user" ? "Bạn" : "Trợ lý";
      const body = document.createElement("span");
      body.className = "message-body";
      message.append(label, body);
      insertMessageInTurnOrder(message, role, turn);
    }
    message.querySelector(".message-body").textContent = content;
    message.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function appendAssistantDelta(data) {
    const generation = String(data.generation ?? data.turn ?? "current");
    const current = state.assistantDrafts.get(generation) || "";
    const content = current + String(data.content || "");
    state.assistantDrafts.set(generation, content);
    addMessage("assistant", content, data.turn);
    setActivity("MiniCPM đang sinh câu trả lời…");
  }

  function isUserFacingResponse(data) {
    if (data.purpose) return data.purpose === "response";
    const classifierValues = new Set(["continue", "switch"]);
    return !classifierValues.has(String(data.content || "").trim().toLowerCase());
  }

  function sendPlaybackAck(event, turn, generation, extra = {}) {
    const websocket = state.websocket;
    if (generation == null || websocket?.readyState !== WebSocket.OPEN) return;
    websocket.send(JSON.stringify({
      event,
      data: {
        turn, generation,
        client_time_ms: Math.round(performance.now()),
        audio_context_time: state.audioContext?.currentTime ?? null,
        ...extra,
      },
    }));
  }

  function schedulePlaybackStartedAck(stream, startAt) {
    const check = () => {
      const context = state.audioContext;
      if (
        state.pcmStream !== stream
        || stream.cancelled
        || stream.startedAck
      ) return;
      if (
        !context
        || context.state !== "running"
        || context.currentTime + 0.005 < startAt
      ) {
        const delayMs = context?.state === "running"
          ? Math.max(10, (startAt - context.currentTime) * 1000)
          : 50;
        stream.startAckTimer = setTimeout(check, delayMs);
        return;
      }
      markPlaybackStarted(stream);
    };
    const context = state.audioContext;
    const initialDelay = context
      ? Math.max(0, (startAt - context.currentTime) * 1000)
      : 50;
    stream.startAckTimer = setTimeout(check, initialDelay);
  }

  function markPlaybackStarted(stream) {
    if (!stream || stream.cancelled || stream.startedAck) return;
    stream.startedAck = true;
    sendPlaybackAck(
      "playback_started", stream.turn, stream.generation,
      { buffered_ms: Math.max(0, Math.round(
        (stream.nextStartTime - (state.audioContext?.currentTime || 0)) * 1000,
      )) },
    );
  }

  function stopPlayback(reason = "client_stop") {
    state.playbackQueue.length = 0;
    state.pendingTts = null;
    const pcmStream = state.pcmStream;
    state.pcmStream = null;
    if (pcmStream) {
      pcmStream.cancelled = true;
      if (pcmStream.startAckTimer) clearTimeout(pcmStream.startAckTimer);
      for (const source of pcmStream.sources) {
        try {
          source.onended = null;
          source.stop();
        } catch (_) {
          // The source may already have ended.
        }
        source.disconnect();
      }
      pcmStream.sources.clear();
      if (!pcmStream.terminalAck) {
        pcmStream.terminalAck = true;
        sendPlaybackAck(
          "playback_stopped", pcmStream.turn, pcmStream.generation,
          { reason, buffered_ms: 0 },
        );
      }
    }
    if (state.currentPlayback) {
      try {
        state.currentPlayback.onended = null;
        state.currentPlayback.stop();
      } catch (_) {
        // The source may already have ended.
      }
      state.currentPlayback.disconnect();
      state.currentPlayback = null;
    }
    state.playing = false;
  }

  async function playNext() {
    if (state.playing || !state.playbackQueue.length || !state.audioContext) return;
    state.playing = true;
    const item = state.playbackQueue.shift();
    try {
      const audioBuffer = await state.audioContext.decodeAudioData(item.bytes.slice(0));
      const source = state.audioContext.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(state.audioContext.destination);
      state.currentPlayback = source;
      source.onended = () => {
        source.disconnect();
        if (state.currentPlayback === source) state.currentPlayback = null;
        state.playing = false;
        sendPlaybackAck("playback_drained", item.turn, item.generation);
        setActivity("Đang nghe. Bạn có thể nói tiếp.");
        void playNext();
      };
      setActivity("Trợ lý đang nói. Bạn có thể nói chen.");
      source.start();
    } catch (error) {
      state.playing = false;
      sendPlaybackAck("playback_stopped", item.turn, item.generation, { reason: "decode_error" });
      logEvent("Không phát được WAV", { message: error.message });
      setActivity("Không phát được âm thanh trả lời.");
      void playNext();
    }
  }

  function queueAudio(bytes) {
    if (!state.pendingTts) {
      logEvent("Bỏ WAV không có generation đang chờ");
      return;
    }
    state.playbackQueue.push({ bytes, ...state.pendingTts });
    state.pendingTts = null;
    void playNext();
  }

  function finishPcmPlayback(stream) {
    if (
      state.pcmStream !== stream
      || stream.cancelled
      || !stream.endReceived
      || stream.sources.size
    ) {
      return;
    }
    state.pcmStream = null;
    state.playing = false;
    if (stream.startAckTimer) clearTimeout(stream.startAckTimer);
    markPlaybackStarted(stream);
    if (!stream.terminalAck) {
      stream.terminalAck = true;
      sendPlaybackAck(
        "playback_drained", stream.turn, stream.generation,
        { buffered_ms: 0, chunks: stream.chunkCount },
      );
    }
    setActivity("Đang nghe. Bạn có thể nói tiếp.");
  }

  function beginPcmStream(data) {
    stopPlayback("superseded_by_new_stream");
    if (data.format !== "pcm_s16le" || data.channels !== 1) {
      logEvent("Định dạng stream TTS không hỗ trợ", data);
      setActivity("Không phát được định dạng âm thanh trả lời.");
      return;
    }
    state.pcmStream = {
      turn: data.turn,
      generation: data.generation,
      sampleRate: Number(data.sample_rate) || TARGET_SAMPLE_RATE,
      nextStartTime: 0,
      chunkCount: 0,
      sources: new Set(),
      endReceived: false,
      cancelled: false,
      scheduledAck: false,
      startedAck: false,
      terminalAck: false,
      startAckTimer: null,
    };
    setActivity("Đang chờ chunk âm thanh đầu tiên…");
  }

  function pcm16AudioBuffer(bytes, sampleRate) {
    const sampleCount = Math.floor(bytes.byteLength / 2);
    const audioBuffer = state.audioContext.createBuffer(1, sampleCount, sampleRate);
    const channel = audioBuffer.getChannelData(0);
    const view = new DataView(bytes);
    for (let index = 0; index < sampleCount; index += 1) {
      channel[index] = view.getInt16(index * 2, true) / 32768;
    }
    return audioBuffer;
  }

  function queuePcmAudio(bytes) {
    const stream = state.pcmStream;
    const context = state.audioContext;
    if (!stream || !context) {
      logEvent("Bỏ PCM chunk không có stream đang chờ");
      return;
    }
    if (!bytes.byteLength || bytes.byteLength % 2) {
      logEvent("Bỏ PCM chunk sai kích thước", { bytes: bytes.byteLength });
      return;
    }

    const audioBuffer = pcm16AudioBuffer(bytes, stream.sampleRate);
    const source = context.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(context.destination);

    const lead = stream.chunkCount === 0
      ? STREAM_START_DELAY_SECONDS
      : STREAM_RECOVERY_DELAY_SECONDS;
    const startAt = Math.max(stream.nextStartTime, context.currentTime + lead);
    stream.nextStartTime = startAt + audioBuffer.duration;
    stream.chunkCount += 1;
    stream.sources.add(source);
    source.onended = () => {
      source.disconnect();
      stream.sources.delete(source);
      finishPcmPlayback(stream);
    };
    source.start(startAt);
    state.playing = true;
    if (stream.chunkCount === 1) {
      const bufferedMs = Math.max(0, Math.round(
        (stream.nextStartTime - context.currentTime) * 1000,
      ));
      stream.scheduledAck = true;
      sendPlaybackAck(
        "playback_scheduled", stream.turn, stream.generation,
        { buffered_ms: bufferedMs, scheduled_start: startAt },
      );
      schedulePlaybackStartedAck(stream, startAt);
      setActivity("Trợ lý đang nói. Bạn có thể nói chen.");
    }
  }

  function endPcmStream(data) {
    const stream = state.pcmStream;
    if (!stream) {
      logEvent("Bỏ kết thúc PCM không có stream đang chờ", data);
      return;
    }
    if (
      stream.turn !== data.turn
      || stream.generation !== data.generation
    ) {
      logEvent("Bỏ kết thúc PCM của generation cũ", data);
      return;
    }
    stream.endReceived = true;
    finishPcmPlayback(stream);
  }

  function handleControl(payload) {
    const event = payload.event;
    const data = payload.data || {};
    logEvent(event, data);
    recordTraceEvent(event, data);

    switch (event) {
      case "live_prefill_ready":
        setActivity("Live-prefill đã sẵn sàng. Hãy nói tự nhiên.");
        break;
      case "paper_unit_ready":
        if (data.native_prefill) {
          setActivity("Paper Unit + native KV-prefill đã sẵn sàng.");
        } else {
          setActivity("Paper Unit đang dùng buffered replay.");
        }
        break;
      case "vad_segment_ready":
        setActivity(`VAD Segment + streaming ASR sẵn sàng · endpoint ${data.endpoint_ms} ms.`);
        break;
      case "vad_segment_resumed":
        setActivity(`Tiếp tục thu Segment ${data.segment}; chưa gửi decision.`);
        break;
      case "vad_segment_finalized":
        setActivity(`Đã chốt Segment ${data.segment} (${data.audio_seconds}s); MiniCPM đang quyết định.`);
        break;
      case "vad_segment_tick":
        setActivity(`${String(data.flag || "").toUpperCase()} · Segment ${data.segment} · ASR hiện tại không dùng=${data.current_asr_used === false}.`);
        break;
      case "vad_segment_decision_fallback":
        setActivity(
          "Fallback " + String(data.flag || data.decision || "").toUpperCase()
          + " · " + (data.fallback_reason || "decision lỗi") + ".",
        );
        break;
      case "vad_segment_continue_timeout_armed":
        setActivity(
          "KL \u00b7 ch\u1edd ng\u01b0\u1eddi d\u00f9ng n\u00f3i ti\u1ebfp t\u1ed1i \u0111a "
          + (data.timeout_seconds ?? 2.5) + "s.",
        );
        break;
      case "vad_segment_continue_timeout_cancelled":
        setActivity("Ng\u01b0\u1eddi d\u00f9ng n\u00f3i ti\u1ebfp; \u0111\u00e3 h\u1ee7y fallback KL.");
        break;
      case "vad_segment_continue_timeout_skipped":
        setActivity(
          "B\u1ecf fallback KL c\u0169: "
          + (data.reason || "state \u0111\u00e3 \u0111\u1ed5i") + ".",
        );
        break;
      case "vad_segment_continue_timeout_fired":
        setActivity(
          "H\u1ebft " + (data.timeout_seconds ?? 2.5)
          + "s sau KL; fallback FD-BADCAT b\u1eaft \u0111\u1ea7u tr\u1ea3 l\u1eddi.",
        );
        break;
      case "vad_segment_replaced":
        setActivity(`Decision bận; giữ Segment mới nhất ${data.latest_segment}.`);
        break;
      case "vad_segment_superseded":
        setActivity(`Không áp dụng ${String(data.flag || "").toUpperCase()} của Segment ${data.segment}; đang xử lý segment mới hơn.`);
        break;
      case "vad_segment_replayed":
        setActivity(`Đánh giá lại Segment ${data.segment} theo state hiện tại.`);
        break;
      case "asr_stream_started":
        setActivity(`Streaming ASR bắt đầu cho Segment ${data.segment}.`);
        break;
      case "asr_partial":
        setActivity(`ASR partial: ${data.transcript || "…"}`);
        break;
      case "asr_stream_final":
        setActivity(`ASR final Segment ${data.segment}: ${data.transcript || "<rỗng>"}`);
        break;
      case "asr_context_cached":
        setActivity(`Transcript Segment ${data.segment} sẵn sàng cho decision kế tiếp.`);
        break;
      case "vad_segment_error":
      case "asr_stream_error":
        setActivity(`Lỗi VAD/ASR segment: ${data.message || "không xác định"}`);
        break;
      case "paper_barge_in_started":
        setActivity("Đang thu lời chen ngang…");
        break;
      case "paper_barge_in_merged":
        setActivity(`Đang ghép lời chen ngang · ${data.segments || 1} vùng VAD.`);
        break;
      case "paper_barge_in_finalized":
        setActivity(`Đã chốt ${data.audio_seconds || 0}s audio chen ngang.`);
        break;
      case "paper_units_coalesced":
        setActivity(`Đã ưu tiên Unit ${data.final_unit}; đang quyết định barge-in.`);
        break;
      case "paper_prefill_started":
        setActivity("Đã mở MiniCPM response KV-cache.");
        break;
      case "paper_prefill_tick":
        setActivity(`KV-prefill · Unit ${data.unit} · ${data.prefill_time}s`);
        break;
      case "paper_prefill_fallback":
        setActivity(`Native KV-prefill lỗi: ${data.message || "không xác định"}`);
        break;
      case "paper_decision_retry":
        setActivity(
          `Xét lại cuối vùng nói · ${String(data.decision || "").toUpperCase()}`,
        );
        break;
      case "paper_unit_tick":
        setActivity(`${String(data.flag || "").toUpperCase()} · Unit ${data.unit} đã xử lý.`);
        break;
      case "paper_unit_error":
        setActivity(`Lỗi Paper Unit: ${data.message || "không xác định"}`);
        break;
      case "paper_unit_replayed":
        setActivity(
          `Unit ${data.unit} thuộc response cũ đã được chuyển sang `
          + `${String(data.current_state || "LISTEN").toUpperCase()}.`,
        );
        break;
      case "duplex_decision":
        if (data.flag === "kl") {
          setActivity("KL · MiniCPM tiếp tục nghe.");
        } else if (data.flag === "l2s") {
          setActivity("L2S · MiniCPM bắt đầu trả lời.");
        } else if (data.flag === "ks") {
          setActivity("KS · Trợ lý tiếp tục nói.");
        } else if (data.flag === "s2l") {
          stopPlayback();
          setActivity("S2L · Đã nhận chen ngang, đang nghe bạn nói.");
        }
        break;
      case "live_prefill_fallback":
        setActivity("Live-prefill lỗi; đang dùng classifier continue/switch.");
        break;
      case "live_prefill_error":
        setActivity(`Lỗi live-prefill: ${data.message || "không xác định"}`);
        break;
      case "vad_start":
        setActivity(data.state === "SPEAK" ? "Đang nghe lời nói chen…" : "Đang nghe bạn nói…");
        break;
      case "vad_done":
      case "vad_640_done":
        setActivity("Đang xử lý lời nói…");
        break;
      case "asr_done":
        addMessage("user", data.content, data.turn);
        break;
      case "response_started":
        state.assistantDrafts.set(String(data.generation), "");
        setActivity("MiniCPM đang sinh câu trả lời…");
        break;
      case "assistant_delta":
        appendAssistantDelta(data);
        break;
      case "llm_done":
        if (isUserFacingResponse(data)) {
          state.assistantDrafts.delete(String(data.generation));
          addMessage("assistant", data.content, data.turn);
          setActivity("Đang tạo giọng nói…");
        }
        break;
      case "tts_done":
        state.pendingTts = {
          turn: data.turn,
          generation: data.generation,
        };
        setActivity("Đã tạo giọng nói, đang nhận WAV…");
        break;
      case "tts_stream_start":
        beginPcmStream(data);
        break;
      case "tts_phrase_ready":
        setActivity("Phrase " + (Number(data.phrase) + 1) + " đã sẵn sàng cho TTS.");
        break;
      case "tts_first_audio":
        setActivity("Đang nhận và phát trực tiếp âm thanh…");
        break;
      case "tts_segment_start":
        setActivity(`Đang tạo giọng nói đoạn ${Number(data.segment) + 1}…`);
        break;
      case "tts_segment_end":
        setActivity("Đang phát và chuẩn bị đoạn tiếp theo…");
        break;
      case "tts_stream_end":
        endPcmStream(data);
        break;
      case "tts_error":
        stopPlayback();
        setActivity(`Lỗi TTS: ${data.message || "không xác định"}`);
        break;
      case "stop_audio":
      case "shot_interrupt":
      case "long_interrupt":
        stopPlayback();
        setActivity("Đã ngắt câu trả lời. Đang nghe bạn nói…");
        break;
      case "generation_cancelled":
        state.assistantDrafts.delete(String(data.generation));
        stopPlayback();
        setActivity("Đã ngắt câu trả lời. Đang nghe bạn nói…");
        break;
      case "generation_error":
        stopPlayback();
        setActivity(`Lỗi pipeline: ${data.message || "không xác định"}`);
        break;
      case "response_server_complete":
        setActivity("Server đã gửi xong; browser vẫn đang phát audio…");
        break;
      case "playback_acknowledged":
        if (data.phase === "PLAYING") {
          setActivity("Browser đang phát audio. Bạn có thể nói chen.");
        }
        break;
      case "playback_ack_timeout":
        setActivity("Không nhận ACK phát hết; backend đã tự giải phóng state.");
        break;
      case "response_complete":
        setActivity("Đã trả lời xong. Đang nghe lượt tiếp theo…");
        break;
      case "no_interrupt":
        setActivity("Tiếp tục phát câu trả lời.");
        break;
      default:
        break;
    }
  }

  function sendSamples(samples) {
    const websocket = state.websocket;
    if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

    for (let index = 0; index < samples.length; index += 1) {
      state.frame[state.frameOffset] = samples[index];
      state.frameOffset += 1;
      if (state.frameOffset === FRAME_SAMPLES) {
        if (websocket.bufferedAmount < MAX_BUFFERED_BYTES) {
          websocket.send(state.frame.slice().buffer);
        }
        state.frameOffset = 0;
      }
    }
  }

  function waitForLivePrefillReady(websocket, timeoutMs = 60_000) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const finish = (error = null) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timer);
        websocket.removeEventListener("message", onMessage);
        websocket.removeEventListener("close", onClose);
        if (error) reject(error);
        else resolve();
      };
      const onMessage = (message) => {
        if (typeof message.data !== "string") return;
        try {
          const payload = JSON.parse(message.data);
          const event = payload.event;
          const data = payload.data || {};
          if (["live_prefill_ready", "paper_unit_ready", "vad_segment_ready"].includes(event)) {
            finish();
          } else if (event === "live_prefill_error") {
            finish(new Error(
              `Khởi tạo live-prefill lỗi: ${data.message || "không xác định"}`,
            ));
          } else if (event === "live_prefill_fallback") {
            finish(new Error(
              `MiniCPM chuyển sang fallback: ${data.reason || "không xác định"}`,
            ));
          }
        } catch (_) {
          // The regular message handler reports malformed control frames.
        }
      };
      const onClose = () => finish(new Error(
        "WebSocket đóng trước khi pipeline duplex sẵn sàng",
      ));
      const timer = window.setTimeout(() => finish(new Error(
        "Pipeline duplex khởi tạo quá 60 giây",
      )), timeoutMs);

      websocket.addEventListener("message", onMessage);
      websocket.addEventListener("close", onClose);
    });
  }

  async function openWebSocket() {
    const websocket = new WebSocket(websocketUrl());
    websocket.binaryType = "arraybuffer";
    state.websocket = websocket;

    await new Promise((resolve, reject) => {
      const timer = window.setTimeout(
        () => reject(new Error("WebSocket kết nối quá thời gian")),
        10_000,
      );
      websocket.addEventListener("open", () => {
        window.clearTimeout(timer);
        resolve();
      }, { once: true });
      websocket.addEventListener("error", () => {
        window.clearTimeout(timer);
        reject(new Error("Không kết nối được WebSocket backend"));
      }, { once: true });
    });

    const liveReady = waitForLivePrefillReady(websocket);

    websocket.addEventListener("message", (message) => {
      if (typeof message.data === "string") {
        try {
          handleControl(JSON.parse(message.data));
        } catch (error) {
          logEvent("JSON không hợp lệ", { message: error.message });
        }
      } else if (message.data instanceof ArrayBuffer) {
        if (state.pcmStream) queuePcmAudio(message.data);
        else queueAudio(message.data);
      }
    });

    websocket.addEventListener("close", () => {
      logEvent("WebSocket đã đóng");
      if (!state.expectedClose) {
        void stopConversation(false).then(() => {
          setConnection("Mất kết nối", "error");
          setActivity("Backend đã ngắt kết nối. Nhấn Bắt đầu để thử lại.");
        });
      }
    });

    websocket.send(JSON.stringify({
      event: "config",
      data: { exp: "live", lang: elements.language.value },
    }));
    return { liveReady };
  }

  async function startAudioCapture() {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass || !window.AudioWorkletNode) {
      throw new Error("Trình duyệt không hỗ trợ AudioWorklet");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("Microphone chỉ khả dụng trên localhost hoặc HTTPS");
    }

    state.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
      video: false,
    });

    state.audioContext = new AudioContextClass({ latencyHint: "interactive" });
    await state.audioContext.audioWorklet.addModule("/static/audio-worklet.js");
    await state.audioContext.resume();

    state.resampler = new StreamingResampler(
      state.audioContext.sampleRate,
      TARGET_SAMPLE_RATE,
    );
    state.frameOffset = 0;
    state.sourceNode = state.audioContext.createMediaStreamSource(state.mediaStream);
    state.captureNode = new AudioWorkletNode(state.audioContext, "pcm-capture");
    state.silentGain = state.audioContext.createGain();
    state.silentGain.gain.value = 0;

    state.captureNode.port.onmessage = ({ data }) => {
      const nativeSamples = new Float32Array(data);
      let energy = 0;
      for (const sample of nativeSamples) energy += sample * sample;
      const rms = Math.sqrt(energy / Math.max(1, nativeSamples.length));
      elements.meter.style.width = `${Math.min(100, Math.max(1, rms * 550))}%`;
      sendSamples(state.resampler.push(nativeSamples));
    };

    state.sourceNode
      .connect(state.captureNode)
      .connect(state.silentGain)
      .connect(state.audioContext.destination);
  }

  async function startConversation() {
    resetTraceDashboard(false);
    elements.start.disabled = true;
    elements.language.disabled = true;
    setConnection("Đang kết nối…");
    setActivity("Đang kết nối tới backend…");
    state.stopping = false;
    state.expectedClose = false;

    try {
      const { liveReady } = await openWebSocket();
      setActivity("Đang khởi tạo MiniCPM duplex…");
      await liveReady;
      setActivity("Đang xin quyền sử dụng microphone…");
      await startAudioCapture();
      elements.stop.disabled = false;
      setConnection("Đã kết nối", "connected");
      setActivity("Đang nghe. Hãy nói tự nhiên rồi dừng lại.");
      logEvent("Microphone đã mở", {
        inputSampleRate: state.audioContext.sampleRate,
        outputSampleRate: TARGET_SAMPLE_RATE,
      });
    } catch (error) {
      logEvent("Không khởi động được", { message: error.message });
      await stopConversation(false);
      setConnection("Không thể bắt đầu", "error");
      setActivity(error.message);
    }
  }

  async function stopConversation(notifyServer = true) {
    if (state.stopping) return;
    state.stopping = true;
    state.expectedClose = true;
    stopPlayback();

    if (state.captureNode) {
      state.captureNode.port.onmessage = null;
      state.captureNode.disconnect();
      state.captureNode = null;
    }
    if (state.sourceNode) {
      state.sourceNode.disconnect();
      state.sourceNode = null;
    }
    if (state.silentGain) {
      state.silentGain.disconnect();
      state.silentGain = null;
    }
    if (state.mediaStream) {
      for (const track of state.mediaStream.getTracks()) track.stop();
      state.mediaStream = null;
    }
    if (state.audioContext) {
      await state.audioContext.close();
      state.audioContext = null;
    }

    const websocket = state.websocket;
    state.websocket = null;
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      if (notifyServer) websocket.send(JSON.stringify({ event: "end" }));
      websocket.close(1000, "client stopped");
    } else if (websocket && websocket.readyState === WebSocket.CONNECTING) {
      websocket.close();
    }

    state.resampler = null;
    state.frameOffset = 0;
    state.pendingTts = null;
    state.assistantDrafts.clear();
    elements.meter.style.width = "1%";
    elements.start.disabled = false;
    elements.stop.disabled = true;
    elements.language.disabled = false;
    setConnection("Chưa kết nối");
    setActivity("Đã dừng. Nhấn Bắt đầu để trò chuyện lại.");
    state.stopping = false;
  }

  elements.start.addEventListener("click", () => void startConversation());
  elements.stop.addEventListener("click", () => void stopConversation());
  elements.traceFilter.addEventListener("change", renderTraceTimeline);
  elements.tracePause.addEventListener("click", toggleTracePause);
  elements.traceExport.addEventListener("click", exportTraceJson);
  elements.traceClear.addEventListener("click", () => resetTraceDashboard(true));
  elements.clear.addEventListener("click", () => {
    for (const message of elements.messages.querySelectorAll(".message")) {
      message.remove();
    }
    elements.empty.hidden = false;
    elements.log.textContent = "";
    state.assistantDrafts.clear();
    resetTraceDashboard(true);
  });
  resetTraceDashboard(false);
  window.addEventListener("beforeunload", () => {
    if (state.websocket?.readyState === WebSocket.OPEN) {
      state.websocket.send(JSON.stringify({ event: "end" }));
    }
  });
})();
