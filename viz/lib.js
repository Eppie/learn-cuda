// viz/lib.js — shared helpers for the learn-cuda visualization track.
// Vanilla JS, no dependencies. Exposed as `window.vizlib`.
(function (global) {
  "use strict";

  const SVG_NS = "http://www.w3.org/2000/svg";

  // 32-color palette for memory banks / lanes (high-contrast).
  const BANK_PALETTE = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#46f0f0","#f032e6",
    "#bcf60c","#fabebe","#008080","#e6beff","#9a6324","#fffac8","#800000","#aaffc3",
    "#808000","#ffd8b1","#000075","#808080","#a9a9a9","#ff6347","#7fffd4","#daa520",
    "#dda0dd","#5f9ea0","#b22222","#228b22","#dc143c","#00ced1","#ff1493","#1e90ff",
  ];

  // Status colors (coalescing efficiency, etc.).
  const STATUS = {
    good:    "#2ecc71",
    partial: "#f1c40f",
    bad:     "#e74c3c",
    neutral: "#95a5a6",
    accentA: "#3498db",
    accentB: "#9b59b6",
  };

  function el(tag, attrs, children) {
    const e = document.createElementNS(SVG_NS, tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (children) for (const c of children) e.appendChild(c);
    return e;
  }

  function html(tag, attrs, text) {
    const e = document.createElement(tag);
    if (attrs) for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }

  // Build a labeled slider input with a numeric readout.
  // opts: {min,max,step,value,label,format(v),onChange(v)}
  function slider(opts) {
    const wrap = html("div", { class: "ctrl" });
    const lab  = html("label", null, opts.label + ": ");
    const out  = html("span", { class: "readout" }, (opts.format ? opts.format(opts.value) : String(opts.value)));
    const inp  = html("input", { type: "range", min: opts.min, max: opts.max, step: opts.step || 1, value: opts.value });
    inp.style.width = "180px";
    inp.addEventListener("input", () => {
      const v = Number(inp.value);
      out.textContent = opts.format ? opts.format(v) : String(v);
      if (opts.onChange) opts.onChange(v);
    });
    lab.appendChild(out);
    wrap.appendChild(lab);
    wrap.appendChild(inp);
    return { node: wrap, input: inp, set value(v){ inp.value = v; out.textContent = opts.format ? opts.format(v) : String(v); }, get value(){ return Number(inp.value); } };
  }

  // Build a select dropdown.
  function dropdown(opts) {
    const wrap = html("div", { class: "ctrl" });
    const lab  = html("label", null, opts.label + ": ");
    const sel  = html("select");
    for (const o of opts.options) {
      const op = html("option", { value: String(o.value) }, o.text);
      sel.appendChild(op);
    }
    sel.value = String(opts.value);
    sel.addEventListener("change", () => { if (opts.onChange) opts.onChange(sel.value); });
    lab.appendChild(sel);
    wrap.appendChild(lab);
    return { node: wrap, select: sel };
  }

  // Build a button.
  function button(text, onClick) {
    const b = html("button", null, text);
    b.addEventListener("click", onClick);
    return b;
  }

  // Inject a small base stylesheet into the document.
  function injectBaseStyles() {
    if (document.getElementById("vizlib-base-css")) return;
    const css = `
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
             margin: 0; padding: 18px 24px; background: #fafafa; color: #222; }
      h1 { font-size: 20px; margin: 0 0 6px 0; }
      .intro { max-width: 900px; line-height: 1.45; color: #444; margin-bottom: 14px; font-size: 14px; }
      .controls { display: flex; flex-wrap: wrap; gap: 12px 18px; margin-bottom: 12px; align-items: center;
                  background: #fff; padding: 10px 14px; border: 1px solid #e0e0e0; border-radius: 6px; }
      .ctrl { display: flex; align-items: center; gap: 8px; font-size: 13px; }
      .ctrl label { display: flex; align-items: center; gap: 6px; }
      .readout { font-variant-numeric: tabular-nums; color: #555; min-width: 30px; display: inline-block; }
      button { font-size: 13px; padding: 4px 10px; border: 1px solid #bbb; background: #fff;
               border-radius: 4px; cursor: pointer; }
      button:hover { background: #f0f0f0; }
      .panel { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px; margin-bottom: 12px; }
      .legend { display: flex; flex-wrap: wrap; gap: 12px; font-size: 12px; color: #444; }
      .legend .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: middle; margin-right: 4px;
                        border: 1px solid #999; }
      svg { background: #fff; }
      code { background: #f3f3f3; padding: 1px 5px; border-radius: 3px; font-size: 12px; }
      .stats { font-size: 13px; line-height: 1.6; }
      .stats b { color: #111; }
    `;
    const s = document.createElement("style");
    s.id = "vizlib-base-css";
    s.textContent = css;
    document.head.appendChild(s);
  }

  // Convenience: make a header (h1 + intro).
  function header(parent, title, intro) {
    parent.appendChild(html("h1", null, title));
    parent.appendChild(html("div", { class: "intro" }, intro));
  }

  function legend(items) {
    const div = html("div", { class: "legend" });
    for (const it of items) {
      const s = html("span");
      s.innerHTML = `<span class="swatch" style="background:${it.color}"></span>${it.label}`;
      div.appendChild(s);
    }
    return div;
  }

  global.vizlib = {
    SVG_NS, el, html, slider, dropdown, button,
    injectBaseStyles, header, legend, BANK_PALETTE, STATUS,
  };
})(window);
