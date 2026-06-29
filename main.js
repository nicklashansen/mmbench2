/* ==========================================================================
   Interactive paper — interactions (dependency-free, progressive enhancement)
   ========================================================================== */
(function () {
	'use strict';
	var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

	function ready(fn) {
		if (document.readyState !== 'loading') fn();
		else document.addEventListener('DOMContentLoaded', fn);
	}

	ready(function () {
		initProgress();
		initTopbar();
		initReveal();
		initCounters();
		initDotnav();
		initScrolly();
		initRoundTrip();
		initActionTriptych();
		initBars();
		initFrameChart();
		initCalibration();
		initDetectionScrubber();
		initCompare();
		initTaskBelt();
		initLiveDemo();
		initLazyVideos();
		initFtMore();
		initBibCopy();
		if (reduceMotion) {  // hero clips have native autoplay; honor reduced-motion by holding them on a still
			document.querySelectorAll('.hero-mosaic video').forEach(function (v) {
				v.autoplay = false;
				v.addEventListener('play', function () { v.pause(); });
				v.pause();
			});
		} else {
			initHeroMosaic();
		}
	});

	/* ---- small DOM/SVG helpers + shared chart tooltip -------------------- */
	var SVGNS = 'http://www.w3.org/2000/svg';
	function el(ns, tag, attrs, parent) {
		var e = ns ? document.createElementNS(ns, tag) : document.createElement(tag);
		if (attrs) for (var k in attrs) e.setAttribute(k, attrs[k]);
		if (parent) parent.appendChild(e);
		return e;
	}
	var _tip;
	function showTip(html, x, y) {
		if (!_tip) { _tip = document.createElement('div'); _tip.className = 'chart-tip'; document.body.appendChild(_tip); }
		_tip.innerHTML = html; _tip.classList.add('on');
		var pad = 14, tw = _tip.offsetWidth, th = _tip.offsetHeight;
		var left = x + pad, top = y + pad;
		if (left + tw > window.innerWidth - 8) left = x - tw - pad;
		if (top + th > window.innerHeight - 8) top = y - th - pad;
		_tip.style.left = Math.max(8, left) + 'px';
		_tip.style.top = Math.max(8, top) + 'px';
	}
	function hideTip() { if (_tip) _tip.classList.remove('on'); }

	/* ---- per-task frame-count distribution (Fig. 2) --------------------- */
	function initFrameChart() {
		var host = document.getElementById('frameChart');
		if (!host || !window.MMB_FRAME) return;
		var d = window.MMB_FRAME, plot = host.querySelector('.dc-plot'), legend = host.querySelector('.dc-legend');
		if (!plot) return;
		if (!reduceMotion && 'IntersectionObserver' in window) host.classList.add('dc-anim');

		var W = 940, H = 300, m = { t: 16, r: 14, b: 30, l: 46 }, pw = W - m.l - m.r, ph = H - m.t - m.b;
		var L = Math.log10, llo = Math.floor(L(d.min_frames) - 0.05), lhi = Math.ceil(L(d.max_frames) + 0.05);
		var lo = Math.pow(10, llo), hi = Math.pow(10, lhi);
		function ly(v) { return m.t + ph - (L(v) - llo) / (lhi - llo) * ph; }
		function decLab(v) { return v >= 1e6 ? (v / 1e6) + 'M' : v >= 1e3 ? (v / 1e3) + 'k' : String(v); }

		var svg = el(SVGNS, 'svg', { viewBox: '0 0 ' + W + ' ' + H, 'class': 'dc-svg', preserveAspectRatio: 'xMidYMid meet', 'aria-hidden': 'true' });
		for (var e = Math.ceil(llo); e <= Math.floor(lhi); e++) {
			var v = Math.pow(10, e), gy = ly(v);
			el(SVGNS, 'line', { 'class': 'dc-grid', x1: m.l, y1: gy, x2: m.l + pw, y2: gy }, svg);
			el(SVGNS, 'text', { 'class': 'dc-ylab', x: m.l - 8, y: gy + 3.5, 'text-anchor': 'end' }, svg).textContent = decLab(v);
		}
		el(SVGNS, 'line', { 'class': 'dc-axis', x1: m.l, y1: m.t + ph, x2: m.l + pw, y2: m.t + ph }, svg);

		var n = d.tasks.length, bw = pw / n, base = m.t + ph, legendItems = [];
		var pinned = null;
		// Highlight one domain in the legend (null -> clear). Shared by bar hover and the pin click
		// so a pinned domain stays visibly marked exactly like a hovered one.
		function paintLegend(dm) {
			legendItems.forEach(function (l) {
				var on = l.getAttribute('data-domain') === dm;
				l.classList.toggle('hot', dm != null && on);
				l.classList.toggle('cool', dm != null && !on);
			});
		}
		d.tasks.forEach(function (t, i) {
			var x = m.l + i * bw, y = ly(t.frames);
			var r = el(SVGNS, 'rect', { 'class': 'dc-bar', x: x.toFixed(2), y: y.toFixed(2), width: Math.max(bw - 0.35, 0.5).toFixed(2), height: (base - y).toFixed(2), fill: t.color, 'data-domain': t.domain }, svg);
			function enter(ev) {
				showTip('<b>' + (t.name || t.task) + '</b><br><span class="dim">' + t.domain + '</span><br>' + t.frames.toLocaleString() + ' frames', ev.clientX, ev.clientY);
				paintLegend(t.domain);
			}
			function leave() { hideTip(); paintLegend(pinned); }  // restore the pinned highlight, if any
			r.addEventListener('mouseenter', enter); r.addEventListener('mousemove', enter); r.addEventListener('mouseleave', leave);
		});

		var my = ly(d.median_frames);
		el(SVGNS, 'line', { 'class': 'dc-median', x1: m.l, y1: my, x2: m.l + pw, y2: my }, svg);
		el(SVGNS, 'text', { 'class': 'dc-median-lab', x: m.l + pw, y: my - 5, 'text-anchor': 'end' }, svg).textContent = 'median = ' + d.median_frames.toLocaleString();
		el(SVGNS, 'text', { 'class': 'dc-xlab', x: m.l + pw / 2, y: H - 6, 'text-anchor': 'middle' }, svg).textContent = d.n_tasks + ' tasks, sorted by frame count';
		plot.appendChild(svg);

		if (legend) {
			// Hover previews a domain; tap/click pins it (so the highlight works on touch,
			// where there is no hover and pointer-leave never fires). Tapping again unpins.
			function paintDomain(dm) {  // dm null -> reset all bars
				svg.querySelectorAll('.dc-bar').forEach(function (b) {
					b.style.opacity = dm == null ? '' : (b.getAttribute('data-domain') === dm ? '1' : '0.15');
				});
			}
			d.domains.forEach(function (dm) {
				var lg = el(null, 'span', { 'class': 'lg', 'data-domain': dm.display, role: 'button', tabindex: '0', 'aria-pressed': 'false' }, legend);
				el(null, 'span', { 'class': 'sw' }, lg).style.background = dm.color;
				el(null, 'span', { 'class': 'lgt', 'data-label': dm.display }, lg).textContent = dm.display;
				legendItems.push(lg);
				lg.addEventListener('mouseenter', function () { if (!pinned) { paintDomain(dm.display); paintLegend(dm.display); } });
				lg.addEventListener('mouseleave', function () { if (!pinned) { paintDomain(null); paintLegend(null); } });
				lg.addEventListener('click', function () {
					pinned = (pinned === dm.display) ? null : dm.display;
					paintDomain(pinned);
					paintLegend(pinned);
					legendItems.forEach(function (l) { l.setAttribute('aria-pressed', l.getAttribute('data-domain') === pinned ? 'true' : 'false'); });
				});
				lg.addEventListener('keydown', function (e) {
					if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); lg.click(); }
				});
			});
		}

		if (!host.classList.contains('dc-anim')) { host.classList.add('in'); return; }
		var io = new IntersectionObserver(function (en) { en.forEach(function (x) { if (x.isIntersecting) { host.classList.add('in'); io.disconnect(); } }); }, { threshold: 0.25 });
		io.observe(host);
	}

	/* ---- predictor calibration scatter (Fig. 5) ------------------------- */
	function initCalibration() {
		var host = document.getElementById('calibChart');
		if (!host || !window.MMB_CALIB) return;
		var cfg = window.MMB_CALIB, wrap = host.querySelector('.calib-panels');
		if (!wrap) return;
		if (!reduceMotion && 'IntersectionObserver' in window) host.classList.add('calib-anim');
		var dpr = Math.max(1, Math.min(window.devicePixelRatio || 1, 2));
		var POINT = 'rgba(143,72,191,0.30)';
		var fieldOf = { u_r: 'ur', u_f: 'uf', u_s: 'us' };

		var panels = cfg.panels.map(function (p, idx) {
			var panel = el(null, 'div', { 'class': 'calib-panel' }, wrap);
			var head = el(null, 'div', { 'class': 'cp-head' }, panel);
			el(null, 'span', { 'class': 'cp-sym' }, head).innerHTML = 'u<sub>' + p.short.split('_')[1] + '</sub>';
			el(null, 'span', { 'class': 'cp-mode' }, head).textContent = p.mode;
			var plot = el(null, 'div', { 'class': 'cp-plot' }, panel);
			var canvas = el(null, 'canvas', null, plot);
			var svg = el(SVGNS, 'svg', { 'class': 'cp-svg' }, plot);
			el(null, 'div', { 'class': 'cp-xlabel' }, panel).textContent = p.label;
			return { p: p, plot: plot, canvas: canvas, svg: svg, field: fieldOf[p.short], first: idx === 0, idx: idx, hit: null };
		});

		function render(P) {
			var p = P.p, rect = P.plot.getBoundingClientRect(), cw = rect.width, ch = rect.height;
			if (cw < 2 || ch < 2) return;
			var m = { t: 10, r: 10, b: 26, l: 36 }, pw = cw - m.l - m.r, ph = ch - m.t - m.b;
			var x0 = p.xlim[0], x1 = p.xlim[1], y0 = cfg.ylim[0], y1 = cfg.ylim[1];
			function px(v) { return m.l + (v - x0) / (x1 - x0) * pw; }
			function py(v) { return m.t + ph - (v - y0) / (y1 - y0) * ph; }
			P.m = m; P.pw = pw; P.ph = ph; P.px = px; P.py = py; P.x0 = x0; P.x1 = x1;

			var cv = P.canvas; cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr);
			var ctx = cv.getContext('2d'); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, cw, ch);
			ctx.fillStyle = POINT;
			var pts = cfg.points, fld = P.field, hit = [];
			for (var i = 0; i < pts.length; i++) {
				var xv = pts[i][fld], yv = pts[i].y;
				if (xv < x0 || xv > x1 || yv < y0 || yv > y1) continue;
				var X = px(xv), Y = py(yv); ctx.fillRect(X - 1.05, Y - 1.05, 2.1, 2.1); hit.push([X, Y, i]);
			}
			P.hit = hit;

			var svg = P.svg; svg.setAttribute('viewBox', '0 0 ' + cw + ' ' + ch);
			while (svg.firstChild) svg.removeChild(svg.firstChild);
			cfg.yticks.forEach(function (t) {
				var Y = py(t); el(SVGNS, 'line', { 'class': 'cp-grid', x1: m.l, y1: Y, x2: m.l + pw, y2: Y }, svg);
				el(SVGNS, 'text', { 'class': 'cp-tick', x: m.l - 6, y: Y + 3.5, 'text-anchor': 'end' }, svg).textContent = t;
			});
			var thrY = py(cfg.threshold);
			el(SVGNS, 'line', { 'class': 'cp-thr', x1: m.l, y1: thrY, x2: m.l + pw, y2: thrY }, svg);
			el(SVGNS, 'line', { 'class': 'cp-axis', x1: m.l, y1: m.t, x2: m.l, y2: m.t + ph }, svg);
			el(SVGNS, 'line', { 'class': 'cp-axis', x1: m.l, y1: m.t + ph, x2: m.l + pw, y2: m.t + ph }, svg);
			p.xticks.forEach(function (t) { el(SVGNS, 'text', { 'class': 'cp-tick', x: px(t), y: m.t + ph + 15, 'text-anchor': 'middle' }, svg).textContent = t; });
			if (P.first) {
				var cy = m.t + ph / 2;
				el(SVGNS, 'text', { 'class': 'cp-axlabel', x: 11, y: cy, 'text-anchor': 'middle', transform: 'rotate(-90 11 ' + cy + ')' }, svg).textContent = 'Rollout ΔPSNR (dB)';
			}
			if (p.curve && p.curve.length) {
				// the median curve has bins beyond the axis limits — clip it to the plot area
				// (the SVG itself is overflow:visible so edge labels never clip)
				var defs = el(SVGNS, 'defs', null, svg);
				var clip = el(SVGNS, 'clipPath', { id: 'cp-clip-' + P.idx }, defs);
				el(SVGNS, 'rect', { x: m.l, y: m.t, width: pw, height: ph }, clip);
				var cg = el(SVGNS, 'g', { 'clip-path': 'url(#cp-clip-' + P.idx + ')' }, svg);
				var dp = p.curve.map(function (c) { return px(c[0]) + ',' + py(c[1]); }).join(' ');
				el(SVGNS, 'polyline', { 'class': 'cp-curve-halo', points: dp }, cg);
				el(SVGNS, 'polyline', { 'class': 'cp-curve', points: dp }, cg);
				p.curve.forEach(function (c) { el(SVGNS, 'circle', { 'class': 'cp-dot', cx: px(c[0]), cy: py(c[1]), r: 2.6 }, cg); });
			}
			el(SVGNS, 'text', { 'class': 'cp-rho', x: m.l + pw - 4, y: m.t + 13, 'text-anchor': 'end' }, svg).textContent = 'ρ = ' + (p.rho < 0 ? '−' : '+') + Math.abs(p.rho).toFixed(2);
			P.hi = el(SVGNS, 'g', { 'class': 'cp-hi' }, svg);   // brushing overlay, on top
		}

		function renderAll() { panels.forEach(render); }
		renderAll();

		// ---- linked cross-panel brushing: one point is one sequence in all 3 panels
		function pill(g, x, y, text, anchor, cls) {
			var w = text.length * 6.0 + 11, tx = anchor === 'end' ? x - w : (anchor === 'mid' ? x - w / 2 : x);
			el(SVGNS, 'rect', { 'class': 'cp-pill ' + (cls || ''), x: tx, y: y - 7.5, width: w, height: 15, rx: 4 }, g);
			el(SVGNS, 'text', { 'class': 'cp-pilltx', x: tx + w / 2, y: y + 3.5, 'text-anchor': 'middle' }, g).textContent = text;
		}
		function clearHi() {
			panels.forEach(function (Q) { if (Q.hi) while (Q.hi.firstChild) Q.hi.removeChild(Q.hi.firstChild); });
			host.classList.remove('brushing'); hideTip();
		}
		function brush(idx, clientX, clientY) {
			if (idx == null) { clearHi(); return; }
			host.classList.add('brushing');
			var pt = cfg.points[idx], yv = pt.y, diverging = yv < cfg.threshold;
			panels.forEach(function (Q) {
				var g = Q.hi; if (!g) return;
				while (g.firstChild) g.removeChild(g.firstChild);
				var QY = Q.py(yv), xv = pt[Q.field];
				// shared ΔPSNR guide — same height across all three panels (same sequence, same error)
				el(SVGNS, 'line', { 'class': 'cp-gy' + (diverging ? ' neg' : ''), x1: Q.m.l, y1: QY, x2: Q.m.l + Q.pw, y2: QY }, g);
				if (xv >= Q.x0 && xv <= Q.x1) {
					var QX = Q.px(xv);
					el(SVGNS, 'line', { 'class': 'cp-gx', x1: QX, y1: QY, x2: QX, y2: Q.m.t + Q.ph }, g);
					el(SVGNS, 'circle', { 'class': 'cp-ringhalo', cx: QX, cy: QY, r: 6 }, g);
					el(SVGNS, 'circle', { 'class': 'cp-ring', cx: QX, cy: QY, r: 4.5 }, g);
					pill(g, QX, Q.m.t + Q.ph - 9, xv.toFixed(2), 'mid');                 // this panel's predictor reading
				}
				if (Q.first) pill(g, Q.m.l + 3, QY, (yv >= 0 ? '+' : '') + yv.toFixed(1) + ' dB', 'start', diverging ? 'neg' : ''); // ΔPSNR
			});
			if (clientX != null) {
				var tname = (cfg.tasks && pt.t != null) ? cfg.tasks[pt.t] : null;
				var dpsnr = 'ΔPSNR = ' + (yv >= 0 ? '+' : '') + yv.toFixed(1) + ' dB';
				showTip((tname ? '<b>' + tname + '</b><br>' + dpsnr : '<b>' + dpsnr + '</b>') + (diverging ? ' <span class="dim">· diverging</span>' : '') +
					'<br><span class="dim">u<sub>r</sub> ' + pt.ur.toFixed(2) + ' &nbsp;·&nbsp; u<sub>f</sub> ' + pt.uf.toFixed(2) + ' &nbsp;·&nbsp; u<sub>s</sub> ' + pt.us.toFixed(2) + '</span>', clientX, clientY);
			}
		}
		function nearestIdx(P, ev) {
			if (!P.hit) return null;
			var rect = P.plot.getBoundingClientRect(), mx = ev.clientX - rect.left, my = ev.clientY - rect.top;
			var best = null, bd = 1e9;
			for (var i = 0; i < P.hit.length; i++) { var dx = P.hit[i][0] - mx, dy = P.hit[i][1] - my, dd = dx * dx + dy * dy; if (dd < bd) { bd = dd; best = P.hit[i][2]; } }
			return (best != null && bd <= 170) ? best : null;
		}
		panels.forEach(function (P) {
			P.plot.addEventListener('pointermove', function (ev) {
				var idx = nearestIdx(P, ev);
				if (idx != null) brush(idx, ev.clientX, ev.clientY); else clearHi();
			});
			P.plot.addEventListener('pointerdown', function (ev) {   // tap-to-brush on touch
				var idx = nearestIdx(P, ev); if (idx != null) brush(idx, ev.clientX, ev.clientY);
			});
			P.plot.addEventListener('pointerleave', clearHi);
		});

		var rz; window.addEventListener('resize', function () { clearTimeout(rz); clearHi(); rz = setTimeout(renderAll, 150); });

		if (!host.classList.contains('calib-anim')) { host.classList.add('in'); return; }
		var io = new IntersectionObserver(function (en) { en.forEach(function (x) { if (x.isIntersecting) { host.classList.add('in'); io.disconnect(); } }); }, { threshold: 0.2 });
		io.observe(host);
	}

	/* ---- "Detection in motion": per-frame predictor trace vs. a rollout ---- */
	function initDetectionScrubber() {
		var root = document.getElementById('detScrubber');
		if (!root || !window.MMB_SCRUB) return;
		var cfg = window.MMB_SCRUB, clips = cfg.clips, fps = cfg.fps || 15;
		var YMAX = 2.15;                                   // y-axis in units of "× threshold"
		var SIGS = [
			{ key: 'u_r', sym: 'u<sub>r</sub>', color: '#b072e0', name: 'round-trip residual' },
			{ key: 'u_f', sym: 'u<sub>f</sub>', color: '#3a86c8', name: 'flow instability' },
			{ key: 'u_s', sym: 'u<sub>s</sub>', color: '#e08a2e', name: 'inter-seed variance' }
		];

		var stage = root.querySelector('.scrub-stage');
		var vid = root.querySelector('.scrub-vid');
		var verdictEl = root.querySelector('.scrub-verdict .vtext');
		var roVal = root.querySelector('.ro-val');
		var roLbl = root.querySelector('.ro-lbl');
		var capEl = root.querySelector('.scrub-caption');
		var tl = root.querySelector('.scrub-timeline');
		var tlTrack = root.querySelector('.scrub-tl-track');
		var tlTime = root.querySelector('.scrub-tl-time');
		var plot = root.querySelector('.scrub-plot');
		var svg = plot.querySelector('svg');
		var legendEl = root.querySelector('.scrub-legend');

		var ci = 0, frame = 0, started = false, scrubbing = false, wasPlaying = false, loadHandler = null;
		var vis = { u_r: true, u_f: true, u_s: true };
		var G = null;                                      // current geometry + svg refs

		function clip() { return clips[ci]; }
		function frac(sig, f) { return clip()[sig][f] / cfg.thresh[sig]; }

		// ---- build the static chart for the current clip ----------------------
		function buildChart() {
			while (svg.firstChild) svg.removeChild(svg.firstChild);
			var rect = plot.getBoundingClientRect(), cw = rect.width, ch = rect.height;
			if (cw < 2 || ch < 2) return;
			svg.setAttribute('viewBox', '0 0 ' + cw + ' ' + ch);
			var m = { t: 14, r: 14, b: 30, l: 40 }, pw = cw - m.l - m.r, ph = ch - m.t - m.b;
			var n = clip().n;
			function X(f) { return m.l + (n < 2 ? 0 : f / (n - 1) * pw); }
			function Y(v) { return m.t + ph - Math.max(0, Math.min(YMAX, v)) / YMAX * ph; }
			G = { m: m, pw: pw, ph: ph, X: X, Y: Y, n: n };

			// red "hallucination" band above the threshold line (frac >= 1)
			el(SVGNS, 'rect', { 'class': 'sp-band', x: m.l, y: m.t, width: pw, height: Y(1) - m.t }, svg);
			// y grid + ticks (0, 1×, 2×)
			[0, 1, 2].forEach(function (t) {
				el(SVGNS, 'line', { 'class': 'sp-grid', x1: m.l, y1: Y(t), x2: m.l + pw, y2: Y(t) }, svg);
				el(SVGNS, 'text', { 'class': 'sp-tick', x: m.l - 7, y: Y(t) + 3.5, 'text-anchor': 'end' }, svg)
					.textContent = t === 0 ? '0' : t + '×';
			});
			el(SVGNS, 'line', { 'class': 'sp-thr', x1: m.l, y1: Y(1), x2: m.l + pw, y2: Y(1) }, svg);
			el(SVGNS, 'text', { 'class': 'sp-thr-lbl', x: m.l + pw - 2, y: Y(1) - 5, 'text-anchor': 'end' }, svg)
				.textContent = 'hallucination threshold';
			// axes
			el(SVGNS, 'line', { 'class': 'sp-axis', x1: m.l, y1: m.t, x2: m.l, y2: m.t + ph }, svg);
			el(SVGNS, 'line', { 'class': 'sp-axis', x1: m.l, y1: m.t + ph, x2: m.l + pw, y2: m.t + ph }, svg);
			// x ticks in seconds
			var secs = (n - 1) / fps;
			for (var s = 0; s <= secs + 1e-6; s++) {
				var f = s * fps;
				el(SVGNS, 'text', { 'class': 'sp-tick', x: X(f), y: m.t + ph + 15, 'text-anchor': 'middle' }, svg)
					.textContent = s + 's';
			}
			el(SVGNS, 'text', { 'class': 'sp-ylabel', x: 11, y: m.t + ph / 2, 'text-anchor': 'middle',
				transform: 'rotate(-90 11 ' + (m.t + ph / 2) + ')' }, svg).textContent = 'predictor × limit';

			// the three full predictor curves
			G.lines = {}; G.dots = {};
			SIGS.forEach(function (S) {
				var pts = [];
				for (var f = 0; f < n; f++) pts.push(X(f) + ',' + Y(frac(S.key, f)));
				var cls = 'sp-line' + (S.key === clip().lead ? ' lead' : ' dim') + (vis[S.key] ? '' : ' hidden');
				G.lines[S.key] = el(SVGNS, 'polyline', { 'class': cls, points: pts.join(' '), stroke: S.color }, svg);
			});
			// dynamic overlay (playhead + dots), drawn on top
			G.playhead = el(SVGNS, 'line', { 'class': 'sp-playhead', x1: X(0), y1: m.t, x2: X(0), y2: m.t + ph }, svg);
			SIGS.forEach(function (S) {
				G.dots[S.key] = el(SVGNS, 'circle', { 'class': 'sp-dot', r: 4.5, fill: S.color, cx: X(0), cy: Y(frac(S.key, 0)) }, svg);
			});
			updateFrame(frame, true);
		}

		// ---- per-frame UI update ---------------------------------------------
		function dominant(f) {       // visible signal with the highest fraction-of-threshold
			var best = null, bv = -1;
			SIGS.forEach(function (S) {
				if (!vis[S.key]) return;
				var v = frac(S.key, f);
				if (v > bv) { bv = v; best = S; }
			});
			return { S: best, v: bv };
		}
		function stateOf(v) { return v >= 1 ? 'red' : (v >= 0.75 ? 'yellow' : 'green'); }
		function verdictWord(st) { return st === 'red' ? 'Hallucinating' : (st === 'yellow' ? 'Drifting' : 'Stable'); }

		function updateFrame(f, force) {
			f = Math.max(0, Math.min(clip().n - 1, f | 0));
			if (!force && f === frame && G) return;
			frame = f;
			if (G) {
				var x = G.X(f);
				G.playhead.setAttribute('x1', x); G.playhead.setAttribute('x2', x);
				SIGS.forEach(function (S) {
					var d = G.dots[S.key];
					d.setAttribute('cx', x); d.setAttribute('cy', G.Y(frac(S.key, f)));
					d.style.display = vis[S.key] ? '' : 'none';
				});
			}
			var dom = dominant(f), st = stateOf(dom.v);
			stage.classList.remove('ustate-green', 'ustate-yellow', 'ustate-red');
			stage.classList.add('ustate-' + st);
			if (verdictEl) verdictEl.textContent = verdictWord(st);
			if (roVal) { roVal.textContent = dom.v.toFixed(2) + '×'; roVal.classList.toggle('over', dom.v >= 1); }
			if (roLbl && dom.S) roLbl.innerHTML = dom.S.sym + ' vs. limit';
			// timeline
			var t = clip().n < 2 ? 0 : f / (clip().n - 1);
			if (tl) tl.style.setProperty('--t', t);
			if (tlTime) tlTime.textContent = (f / fps).toFixed(1) + 's';
		}

		// ---- playback loop ----------------------------------------------------
		function tick() {
			if (!scrubbing && vid.duration) {
				var f = Math.round(vid.currentTime * fps);
				if (vid.currentTime >= vid.duration - 1e-3) f = clip().n - 1;
				updateFrame(f);
			}
			requestAnimationFrame(tick);
		}
		function paused() { return vid.paused; }
		function setPaused(p) { root.classList.toggle('is-paused', p); }
		function playVid() { var pr = vid.play(); if (pr && pr.catch) pr.catch(function () {}); setPaused(false); }
		function pauseVid() { vid.pause(); setPaused(true); }

		// ---- load a clip ------------------------------------------------------
		function loadClip(i, autoplay) {
			ci = i; frame = 0;
			var c = clip();
			root.querySelectorAll('.scrub-clip').forEach(function (b, k) {
				b.classList.toggle('is-active', k === i); b.setAttribute('aria-selected', k === i ? 'true' : 'false');
			});
			// reset solo so every clip starts showing all three (lead emphasised)
			vis = { u_r: true, u_f: true, u_s: true };
			syncLegend();
			if (capEl) capEl.innerHTML = '<b>' + c.name + '</b> · ' + c.domain + ' — ' + c.caption;
			var src = vid.querySelector('source');
			src.setAttribute('src', c.wm); vid.load();
			if (loadHandler) vid.removeEventListener('loadeddata', loadHandler);  // drop a stale pending load on rapid switch
			loadHandler = function () {
				vid.removeEventListener('loadeddata', loadHandler); loadHandler = null;
				buildChart();
				if (autoplay && !reduceMotion) playVid(); else { pauseVid(); updateFrame(0, true); }
			};
			vid.addEventListener('loadeddata', loadHandler);
		}

		// ---- legend (solo / toggle) ------------------------------------------
		function syncLegend() {
			root.querySelectorAll('.scrub-leg').forEach(function (b) {
				var k = b.getAttribute('data-sig');
				b.classList.toggle('is-off', !vis[k]);
				b.classList.toggle('is-lead', k === clip().lead);
				b.setAttribute('aria-pressed', vis[k] ? 'true' : 'false');
			});
		}
		function toggleSig(k) {
			var on = SIGS.filter(function (S) { return vis[S.key]; });
			if (vis[k] && on.length === 1) return;          // keep at least one visible
			vis[k] = !vis[k];
			if (G && G.lines[k]) G.lines[k].classList.toggle('hidden', !vis[k]);
			syncLegend(); updateFrame(frame, true);
		}

		// ---- build clip selector + legend DOM --------------------------------
		var clipHost = root.querySelector('.scrub-clips'), curGroup = null;
		clips.forEach(function (c, i) {
			if (c.group && c.group !== curGroup) {   // line-break + label when a review group starts
				var sep = el(null, 'div', { 'class': 'scrub-clip-sep' }, clipHost);
				sep.textContent = c.group;
				curGroup = c.group;
			}
			var b = el(null, 'button', { 'class': 'scrub-clip' + (i === 0 ? ' is-active' : '') + (c.candidate ? ' is-candidate' : ''),
				'role': 'tab', 'data-clip': i, 'aria-selected': i === 0 ? 'true' : 'false' }, clipHost);
			el(null, 'span', { 'class': 'sc-name' }, b).textContent = c.name;
			el(null, 'span', { 'class': 'sc-mode' }, b).textContent = c.mode;
			b.addEventListener('click', function () { loadClip(i, true); });
		});
		SIGS.forEach(function (S) {
			var b = el(null, 'button', { 'class': 'scrub-leg', 'data-sig': S.key, 'aria-pressed': 'true',
				'title': 'toggle ' + S.name }, legendEl);
			el(null, 'span', { 'class': 'lswatch' }, b).style.background = S.color;
			el(null, 'span', null, b).innerHTML = S.sym;
			b.addEventListener('click', function () { toggleSig(S.key); });
		});

		// ---- interactions -----------------------------------------------------
		stage.addEventListener('click', function () { paused() ? playVid() : pauseVid(); });

		function seekTo(clientX) {
			var r = tlTrack.getBoundingClientRect();
			var f = Math.round(Math.max(0, Math.min(1, (clientX - r.left) / r.width)) * (clip().n - 1));
			if (vid.duration) { try { vid.currentTime = f / fps; } catch (_) {} }
			updateFrame(f, true);
		}
		tl.addEventListener('pointerdown', function (e) {
			scrubbing = true; tl.classList.add('scrubbing'); wasPlaying = !paused();
			pauseVid();
			if (tl.setPointerCapture) { try { tl.setPointerCapture(e.pointerId); } catch (_) {} }
			seekTo(e.clientX); e.preventDefault();
		});
		tl.addEventListener('pointermove', function (e) { if (scrubbing) seekTo(e.clientX); });
		function endScrub() {
			if (!scrubbing) return;
			scrubbing = false; tl.classList.remove('scrubbing');
			if (wasPlaying && !reduceMotion) playVid();
		}
		tl.addEventListener('pointerup', endScrub);
		tl.addEventListener('pointercancel', endScrub);

		// also let the chart itself be scrubbed (drag anywhere on the plot)
		function seekFromPlot(clientX) {
			if (!G) return;
			var r = plot.getBoundingClientRect();
			var f = Math.round(Math.max(0, Math.min(1, (clientX - r.left - G.m.l) / G.pw)) * (clip().n - 1));
			if (vid.duration) { try { vid.currentTime = f / fps; } catch (_) {} }
			updateFrame(f, true);
		}
		plot.addEventListener('pointerdown', function (e) {
			scrubbing = true; wasPlaying = !paused(); pauseVid();
			if (plot.setPointerCapture) { try { plot.setPointerCapture(e.pointerId); } catch (_) {} }
			seekFromPlot(e.clientX); e.preventDefault();
		});
		plot.addEventListener('pointermove', function (e) { if (scrubbing) seekFromPlot(e.clientX); });
		plot.addEventListener('pointerup', endScrub);
		plot.addEventListener('pointercancel', endScrub);

		var rz; window.addEventListener('resize', function () { clearTimeout(rz); rz = setTimeout(buildChart, 150); });

		// ---- start when scrolled into view -----------------------------------
		function start() {
			if (started) return; started = true;
			loadClip(0, true);
			requestAnimationFrame(tick);
		}
		if ('IntersectionObserver' in window) {
			var io = new IntersectionObserver(function (en) {
				en.forEach(function (x) {
					if (x.isIntersecting) { start(); }
					else if (started && !reduceMotion) pauseVid();   // pause offscreen
				});
			}, { threshold: 0.25 });
			io.observe(root);
		} else { start(); }
	}

	/* ---- top reading-progress bar ---------------------------------------- */
	function initProgress() {
		var bar = document.querySelector('.progress');
		var cue = document.querySelector('.scrollcue');
		if (!bar) return;
		function update() {
			var h = document.documentElement;
			var max = h.scrollHeight - h.clientHeight;
			var st = h.scrollTop || document.body.scrollTop;
			var pct = max > 0 ? st / max * 100 : 0;
			bar.style.width = pct.toFixed(2) + '%';
			if (cue) cue.classList.toggle('cue-hide', st > 40);
		}
		window.addEventListener('scroll', update, { passive: true });
		window.addEventListener('resize', update);
		update();
	}

	/* ---- topbar title fade-in after hero --------------------------------- */
	function initTopbar() {
		var title = document.querySelector('.topbar-title');
		var bar = document.querySelector('.topbar');
		var hero = document.querySelector('.hero');
		if (!title || !hero || !bar || !('IntersectionObserver' in window)) return;
		var io = new IntersectionObserver(function (entries) {
			var pastHero = !entries[0].isIntersecting;
			title.style.opacity = pastHero ? '1' : '0';
			bar.classList.toggle('solid', pastHero);  // frosted plate keeps light links legible over white bands
		}, { threshold: 0.15 });
		io.observe(hero);
	}

	/* ---- scroll reveal ---------------------------------------------------- */
	function initReveal() {
		var els = document.querySelectorAll('.reveal');
		if (reduceMotion || !('IntersectionObserver' in window)) {
			els.forEach(function (el) { el.classList.add('is-visible'); });
			return;
		}
		var io = new IntersectionObserver(function (entries) {
			entries.forEach(function (e) {
				if (e.isIntersecting) { e.target.classList.add('is-visible'); io.unobserve(e.target); }
			});
		}, { threshold: 0.12, rootMargin: '0px 0px -8% 0px' });
		els.forEach(function (el) { io.observe(el); });
	}

	/* ---- animated number counters ---------------------------------------- */
	function initCounters() {
		var nums = document.querySelectorAll('.num[data-target]');
		if (!nums.length) return;
		function animate(el) {
			var target = parseFloat(el.getAttribute('data-target'));
			var dec = parseInt(el.getAttribute('data-decimals') || '0', 10);
			var suffix = el.getAttribute('data-suffix') || '';
			if (reduceMotion) { el.textContent = target.toFixed(dec) + suffix; return; }
			var dur = 1300, start = null;
			function tick(ts) {
				if (!start) start = ts;
				var p = Math.min((ts - start) / dur, 1);
				var eased = 1 - Math.pow(1 - p, 3);
				el.textContent = (target * eased).toFixed(dec) + suffix;
				if (p < 1) requestAnimationFrame(tick);
				else el.textContent = target.toFixed(dec) + suffix;
			}
			requestAnimationFrame(tick);
		}
		if (!('IntersectionObserver' in window)) { nums.forEach(animate); return; }
		var io = new IntersectionObserver(function (entries) {
			entries.forEach(function (e) {
				if (e.isIntersecting) { animate(e.target); io.unobserve(e.target); }
			});
		}, { threshold: 0.5 });
		nums.forEach(function (el) { io.observe(el); });
	}

	/* ---- section dot-nav (build + active state) -------------------------- */
	function initDotnav() {
		var nav = document.querySelector('.dotnav');
		if (!nav) return;
		var sections = Array.prototype.slice.call(document.querySelectorAll('[data-nav]'));
		sections.forEach(function (s) {
			var a = document.createElement('a');
			a.href = '#' + s.id;
			a.innerHTML = '<span>' + s.getAttribute('data-nav') + '</span>';
			a.setAttribute('aria-label', s.getAttribute('data-nav'));
			nav.appendChild(a);
		});
		var links = nav.querySelectorAll('a');
		var current = -1;
		function pick() {
			// Active = the section straddling the viewport's vertical center. A robust
			// scroll-spy: unlike an intersection-ratio threshold, it works for sections
			// taller than the viewport (the scrollytelling Acts never reach 50% visible).
			var centerY = window.innerHeight / 2, idx = -1;
			for (var i = 0; i < sections.length; i++) {
				if (sections[i].getBoundingClientRect().top <= centerY) idx = i; else break;
			}
			if (idx === current) return;
			current = idx;
			links.forEach(function (l, i) {
				var on = i === idx;
				l.classList.toggle('active', on);
				if (on) l.setAttribute('aria-current', 'true'); else l.removeAttribute('aria-current');
			});
			if (idx >= 0) {
				var onDark = sections[idx].classList.contains('act--dark');
				links.forEach(function (l) { l.classList.toggle('on-dark', onDark); });
			}
		}
		var ticking = false;
		function onScroll() {
			if (ticking) return;
			ticking = true;
			requestAnimationFrame(function () { ticking = false; pick(); });
		}
		window.addEventListener('scroll', onScroll, { passive: true });
		window.addEventListener('resize', onScroll);
		pick();
	}

	/* ---- scrollytelling: sticky diagram + stepped text (model & taxonomy) - */
	function initScrolly() {
		var blocks = Array.prototype.slice.call(document.querySelectorAll('.scrolly'));
		if (!blocks.length) return;
		blocks.forEach(function (scrolly) {
			var steps = Array.prototype.slice.call(scrolly.querySelectorAll('.step'));
			if (!steps.length) return;
			var figs = Array.prototype.slice.call(scrolly.querySelectorAll('.mode-fig'));
			var stages = Array.prototype.slice.call(scrolly.querySelectorAll('.stage-box'));

			function activate(idx) {
				steps.forEach(function (s, i) { s.classList.toggle('is-active', i === idx); });
				figs.forEach(function (f, i) { f.classList.toggle('is-active', i === idx); });
				var activeStages = (steps[idx].getAttribute('data-stage') || '').split(',');
				stages.forEach(function (st) {
					st.classList.toggle('is-active', activeStages.indexOf(st.getAttribute('data-stage-id')) !== -1);
				});
			}

			if (reduceMotion || !('IntersectionObserver' in window)) {
				steps.forEach(function (s) { s.classList.add('is-active'); });
				figs.forEach(function (f) { f.classList.add('is-active'); });
				stages.forEach(function (s) { s.classList.add('is-active'); });
				return;
			}

			activate(0);
			var io = new IntersectionObserver(function (entries) {
				entries.forEach(function (e) {
					if (e.isIntersecting) activate(steps.indexOf(e.target));
				});
			}, { rootMargin: '-45% 0px -45% 0px', threshold: 0 });
			steps.forEach(function (s) { io.observe(s); });
		});
	}

	/* ---- Act 4 (i): interactive tokenizer round-trip selector --------------- */
	function initRoundTrip() {
		document.querySelectorAll('.rt-panel').forEach(function (panel) {
			var tabs = Array.prototype.slice.call(panel.querySelectorAll('.rt-tab'));
			if (!tabs.length) return;
			var imgs = Array.prototype.slice.call(panel.querySelectorAll('.rt-img'));
			var verdicts = Array.prototype.slice.call(panel.querySelectorAll('.rt-verdict'));

			function select(ex) {
				panel.setAttribute('data-ex', ex);
				tabs.forEach(function (t) {
					var on = t.getAttribute('data-ex') === ex;
					t.classList.toggle('is-on', on);
					t.setAttribute('aria-selected', on ? 'true' : 'false');
					t.tabIndex = on ? 0 : -1;
				});
				imgs.forEach(function (m) { m.classList.toggle('is-on', m.getAttribute('data-ex') === ex); });
				verdicts.forEach(function (v) { v.classList.toggle('is-on', v.getAttribute('data-ex') === ex); });
			}

			tabs.forEach(function (t, i) {
				t.addEventListener('click', function () { select(t.getAttribute('data-ex')); });
				// roving-tabindex arrow navigation within the tablist
				t.addEventListener('keydown', function (e) {
					var d = (e.key === 'ArrowRight' || e.key === 'ArrowDown') ? 1
						: (e.key === 'ArrowLeft' || e.key === 'ArrowUp') ? -1 : 0;
					if (!d) return;
					e.preventDefault();
					var next = tabs[(i + d + tabs.length) % tabs.length];
					select(next.getAttribute('data-ex'));
					next.focus();
				});
			});

			var initial = tabs.filter(function (t) { return t.classList.contains('is-on'); })[0] || tabs[0];
			select(initial.getAttribute('data-ex'));
		});
	}

	/* ---- Act 4 (ii): action triptych — one task, real / zeroed / flipped side by side; tabs swap task */
	function initActionTriptych() {
		document.querySelectorAll('.am-panel').forEach(function (panel) {
			var tabs = Array.prototype.slice.call(panel.querySelectorAll('.rt-tab'));
			if (!tabs.length) return;
			var vids = Array.prototype.slice.call(panel.querySelectorAll('.am-vid'));
			var verdicts = Array.prototype.slice.call(panel.querySelectorAll('.rt-verdict'));

			function select(task) {
				panel.setAttribute('data-task', task);
				tabs.forEach(function (t) {
					var on = t.getAttribute('data-task') === task;
					t.classList.toggle('is-on', on);
					t.setAttribute('aria-selected', on ? 'true' : 'false');
					t.tabIndex = on ? 0 : -1;
				});
				vids.forEach(function (v) {
					var on = v.getAttribute('data-task') === task;
					v.classList.toggle('is-on', on);
					// Only the active task's 3 clips need to decode; the other 9 are opacity:0.
					// initLazyVideos autoplays every clip in view, so we actively hold the rest paused.
					if (on) { if (v.paused && !reduceMotion && v.dataset.loaded) { var p = v.play(); if (p && p.catch) p.catch(function () {}); } }
					else if (!v.paused) v.pause();
				});
				verdicts.forEach(function (vd) { vd.classList.toggle('is-on', vd.getAttribute('data-task') === task); });
			}

			tabs.forEach(function (t, i) {
				t.addEventListener('click', function () { select(t.getAttribute('data-task')); });
				// roving-tabindex arrow navigation within the tablist
				t.addEventListener('keydown', function (e) {
					var d = (e.key === 'ArrowRight' || e.key === 'ArrowDown') ? 1
						: (e.key === 'ArrowLeft' || e.key === 'ArrowUp') ? -1 : 0;
					if (!d) return;
					e.preventDefault();
					var next = tabs[(i + d + tabs.length) % tabs.length];
					select(next.getAttribute('data-task'));
					next.focus();
				});
			});

			// hold non-active-task clips paused even when initLazyVideos autoplays them on scroll-in
			vids.forEach(function (v) {
				v.addEventListener('play', function () {
					if (v.getAttribute('data-task') !== panel.getAttribute('data-task')) v.pause();
				});
			});

			// keep the three action variants of the active task frame-aligned to its real rollout
			panel.querySelectorAll('.am-vid[data-state="real"]').forEach(function (real) {
				real.addEventListener('timeupdate', function () {
					if (!real.classList.contains('is-on')) return;
					var task = real.getAttribute('data-task');
					vids.forEach(function (v) {
						if (v === real || v.getAttribute('data-task') !== task || v.readyState < 2) return;
						if (Math.abs(v.currentTime - real.currentTime) > 0.12) {
							try { v.currentTime = real.currentTime; } catch (_) {}
						}
					});
				});
			});

			// per-frame arrow-key animation for counterfactual tasks with an action trace
			// (Walker, MuJoCo Walker): map dims 0-1 of the real/flipped command to ← → / ↑ ↓,
			// synced to each column's video. The is-walker key boxes are shared; the active
			// task's trace + video are resolved at paint time.
			var TRACE = (window.AM_ACTIONS || {});
			var KEYTASKS = { walker: 1, mjwalker: 1 };
			var animKeys = Array.prototype.slice.call(panel.querySelectorAll('.am-keys.is-walker'));
			if (animKeys.length && Object.keys(TRACE).length) {
				var THR = 0.15, fps = TRACE.fps || 15;
				var rigs = animKeys.map(function (box) {
					var cells = {};
					box.querySelectorAll('.am-key[data-key]').forEach(function (c) { cells[c.getAttribute('data-key')] = c; });
					return { state: box.getAttribute('data-state'), cells: cells };
				});
				function paint() {
					var task = panel.getAttribute('data-task');
					if (KEYTASKS[task] && TRACE[task]) {
						rigs.forEach(function (r) {
							var vid = panel.querySelector('.am-vid[data-task="' + task + '"][data-state="' + r.state + '"]');
							var seq = (TRACE[task] || {})[r.state] || [];
							if (!vid || !seq.length) return;
							var f = Math.min(seq.length - 1, Math.max(0, Math.round(vid.currentTime * fps)));
							var a = seq[f] || [0, 0];
							r.cells.left.classList.toggle('is-held', a[0] < -THR);
							r.cells.right.classList.toggle('is-held', a[0] > THR);
							r.cells.up.classList.toggle('is-held', a[1] > THR);
							r.cells.down.classList.toggle('is-held', a[1] < -THR);
						});
					}
					requestAnimationFrame(paint);
				}
				requestAnimationFrame(paint);
			}

			var initial = tabs.filter(function (t) { return t.classList.contains('is-on'); })[0] || tabs[0];
			select(initial.getAttribute('data-task'));
		});
	}

	/* ---- animated bar charts --------------------------------------------- */
	function initBars() {
		var charts = document.querySelectorAll('.barchart');
		if (!charts.length) return;
		function fill(chart) {
			chart.querySelectorAll('.bar-fill').forEach(function (b) {
				var w = b.getAttribute('data-w');
				if (w) b.style.width = w;
				b.classList.add('in');
			});
			chart.querySelectorAll('.bar-row').forEach(function (r) { r.classList.add('in'); });
		}
		if (reduceMotion || !('IntersectionObserver' in window)) { charts.forEach(fill); return; }
		var io = new IntersectionObserver(function (entries) {
			entries.forEach(function (e) {
				if (e.isIntersecting) { fill(e.target); io.unobserve(e.target); }
			});
		}, { threshold: 0.35 });
		charts.forEach(function (c) { io.observe(c); });
	}

	/* ---- lazy-load <video> sources (perf) -------------------------------- */
	/* ---- GT vs. world-model side-by-side comparison (Act 1) ------------- */
	function initCompare() {
		var box = document.getElementById('rolloutCompare');
		if (!box) return;
		var gt = box.querySelector('.compare-gt');
		var wm = box.querySelector('.compare-wm');

		function playBoth() {
			if (gt) { var p1 = gt.play(); if (p1 && p1.catch) p1.catch(function () {}); }
			if (wm) { var p2 = wm.play(); if (p2 && p2.catch) p2.catch(function () {}); }
		}
		function pauseBoth() { if (gt) gt.pause(); if (wm) wm.pause(); }
		function toggle() { if (gt && gt.paused) playBoth(); else pauseBoth(); }

		// Play / pause: the button plus clicking or tabbing either rollout, all mirroring one state.
		var playBtn = document.getElementById('rolloutPlay');
		var cells = box.querySelectorAll('.duo-cell');
		function syncControls() {
			if (!gt) return;
			var paused = gt.paused;
			if (playBtn) {
				playBtn.classList.toggle('is-paused', paused);
				playBtn.setAttribute('aria-label', paused ? 'Play' : 'Pause');
			}
			cells.forEach(function (cell) {
				var tag = cell.querySelector('.duo-tag');
				cell.setAttribute('aria-label', (tag ? tag.textContent.trim() : 'rollout') + ', ' + (paused ? 'play' : 'pause'));
			});
		}
		if (playBtn) playBtn.addEventListener('click', toggle);
		cells.forEach(function (cell) {           // each rollout is a click/keyboard toggle, like the button
			cell.tabIndex = 0;
			cell.setAttribute('role', 'button');
			cell.addEventListener('click', toggle);
			cell.addEventListener('keydown', function (e) {
				if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); toggle(); }
			});
		});
		if (gt) {
			gt.addEventListener('play', syncControls);
			gt.addEventListener('pause', syncControls);
		}
		syncControls();

		// Keep the two rollouts frame-aligned so you compare the same timestep.
		if (gt && wm) {
			gt.addEventListener('timeupdate', function () {
				if (wm.readyState >= 2 && Math.abs(wm.currentTime - gt.currentTime) > 0.12) {
					try { wm.currentTime = gt.currentTime; } catch (_) {}
				}
			});
		}

		// Playback timeline: a scrubbable progress bar so you can read position in the loop
		// (the two rollouts agree at the start and drift apart toward the end).
		var timeline = box.querySelector('.compare-timeline');
		if (timeline && gt) {
			var track = timeline.querySelector('.compare-tl-track');
			var scrubbing = false, wasPlaying = false;
			function setT(f) { timeline.style.setProperty('--t', Math.max(0, Math.min(1, f || 0))); }
			(function tlTick() {
				if (!scrubbing && gt.duration) setT(gt.currentTime / gt.duration);  // scrub() owns --t while dragging
				requestAnimationFrame(tlTick);
			})();
			function seek(e) {
				var r = track.getBoundingClientRect();
				var f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
				setT(f);
				if (gt.duration) { try { gt.currentTime = f * gt.duration; if (wm) wm.currentTime = gt.currentTime; } catch (_) {} }
			}
			timeline.addEventListener('pointerdown', function (e) {
				scrubbing = true; timeline.classList.add('scrubbing');
				wasPlaying = !gt.paused;
				pauseBoth();
				if (timeline.setPointerCapture) { try { timeline.setPointerCapture(e.pointerId); } catch (_) {} }
				seek(e); e.preventDefault();
			});
			timeline.addEventListener('pointermove', function (e) { if (scrubbing) seek(e); });
			function endScrub() {
				if (!scrubbing) return;
				scrubbing = false; timeline.classList.remove('scrubbing');
				if (wasPlaying && !reduceMotion) playBoth();
			}
			timeline.addEventListener('pointerup', endScrub);
			timeline.addEventListener('pointercancel', endScrub);
		}
	}

	/* ---- task belt: two rotating, drag/swipe rows of WM rollouts --------- */
	function initTaskBelt() {
		var belt = document.getElementById('taskBelt');
		if (!belt) return;
		var rowEls = Array.prototype.slice.call(belt.querySelectorAll('.belt-row'));
		if (!rowEls.length) return;
		var GAP = 14, speed = 22, vel = 0;            // px gap, auto px/s, shared drag momentum
		var TILE = 138, W = belt.clientWidth;
		var dragging = false, hovering = false, inView = false, raf = 0, lastT = 0;

		var rows = rowEls.map(function (el, ri) {
			return { el: el, tiles: Array.prototype.slice.call(el.querySelectorAll('.belt-tile')),
			         offset: ri * 90, dir: ri % 2 === 0 ? 1 : -1, start: 0, STEP: 0, TOTAL: 0 };
		});
		function measure() {
			W = belt.clientWidth;
			TILE = rowEls[0].clientHeight || 138;
			rows.forEach(function (r) {
				r.STEP = TILE + GAP; r.TOTAL = r.tiles.length * r.STEP;
				r.tiles.forEach(function (t) { t.style.width = TILE + 'px'; t.style.height = TILE + 'px'; });
			});
		}
		function loadTile(t) {
			var v = t.querySelector('video');
			if (v && !v.dataset.loaded) {
				v.querySelectorAll('source[data-src]').forEach(function (s) { s.src = s.getAttribute('data-src'); });
				v.load(); v.dataset.loaded = '1';
			}
		}
		function layoutRow(r) {
			for (var i = 0; i < r.tiles.length; i++) {
				// wrap window is [-TILE, TOTAL-TILE) so tiles slide fully off-screen before recycling
				var x = (((i * r.STEP - r.offset) + TILE) % r.TOTAL + r.TOTAL) % r.TOTAL - TILE;
				r.tiles[i].style.transform = 'translate3d(' + x + 'px,0,0)';
				var v = r.tiles[i].querySelector('video');
				if (x > -TILE && x < W && inView) {
					loadTile(r.tiles[i]);
					if (v && v.dataset.loaded && v.paused && !reduceMotion) {
						var p = v.play(); if (p && p.catch) p.catch(function () {});
					}
				} else if (v && !v.paused) { v.pause(); }
			}
		}
		function tick(ts) {
			if (!lastT) lastT = ts;
			var dt = Math.min(0.05, (ts - lastT) / 1000); lastT = ts;
			var momentum = !dragging && Math.abs(vel) > 2;
			rows.forEach(function (r) {
				if (!dragging) {
					if (momentum) { r.offset -= vel * dt; }                       // flick carries both rows together
					else if (!hovering && !reduceMotion) { r.offset += speed * r.dir * dt; } // each row drifts its own way
				}
				r.offset = ((r.offset % r.TOTAL) + r.TOTAL) % r.TOTAL;
				layoutRow(r);
			});
			if (momentum) vel *= 0.93;
			raf = requestAnimationFrame(tick);
		}
		function start() { if (!raf) { lastT = 0; raf = requestAnimationFrame(tick); } }
		function stop() { if (raf) { cancelAnimationFrame(raf); raf = 0; } }

		var startX = 0, lastMX = 0, lastMT = 0;
		belt.addEventListener('pointerdown', function (e) {
			dragging = true; vel = 0; belt.classList.add('grabbing');
			startX = e.clientX; lastMX = e.clientX; lastMT = e.timeStamp;
			rows.forEach(function (r) { r.start = r.offset; });
			if (belt.setPointerCapture) { try { belt.setPointerCapture(e.pointerId); } catch (_) {} }
		});
		belt.addEventListener('pointermove', function (e) {
			if (!dragging) return;
			var d = e.clientX - startX;
			rows.forEach(function (r) { r.offset = r.start - d; });   // both rows follow the finger
			var dt = (e.timeStamp - lastMT) / 1000;
			if (dt > 0.001) { vel = (e.clientX - lastMX) / dt; lastMX = e.clientX; lastMT = e.timeStamp; }
			e.preventDefault();
		});
		function endDrag() { if (dragging) { dragging = false; belt.classList.remove('grabbing'); } }
		belt.addEventListener('pointerup', endDrag);
		belt.addEventListener('pointercancel', endDrag);
		belt.addEventListener('mouseenter', function () { hovering = true; });
		belt.addEventListener('mouseleave', function () { hovering = false; });
		belt.addEventListener('keydown', function (e) {
			if (e.key === 'ArrowLeft') { rows.forEach(function (r) { r.offset -= r.STEP; }); vel = 0; e.preventDefault(); }
			else if (e.key === 'ArrowRight') { rows.forEach(function (r) { r.offset += r.STEP; }); vel = 0; e.preventDefault(); }
		});

		measure();
		window.addEventListener('resize', measure);
		if ('IntersectionObserver' in window) {
			new IntersectionObserver(function (ents) {
				inView = ents[0].isIntersecting;
				if (inView) { start(); }
				else { stop(); rows.forEach(function (r) { r.tiles.forEach(function (t) { var v = t.querySelector('video'); if (v && !v.paused) v.pause(); }); }); }
			}, { threshold: 0.01 }).observe(belt);
		} else { inView = true; start(); }
	}

	/* ---- live demo (Act 8): embedded client for the self-hosted WM server.
	   Protocol (see ~/code/wm/dreamer4/interactive.py): JSON status/queue/
	   granted/full/end messages + binary JPEG frames over a websocket;
	   /status (CORS *) for availability. States: idle → connecting → queued →
	   live → ended, plus offline. Failover walks ENDPOINTS on "full" or on a
	   connection that dies before any server message arrives. ------------- */
	function initLiveDemo() {
		var root = document.getElementById('liveDemo');
		if (!root || !('WebSocket' in window)) return;

		// Featured tasks in display order, using the site's "Name (Domain)"
		// naming convention.
		var DEMO_TASKS = [
			['acrobot-swingup', 'Acrobot Swingup (DMControl)'],
			['cartpole-swingup', 'Cartpole Swingup (DMControl)'],
			['cup-catch', 'Cup Catch (DMControl)'],
			['finger-turn-easy', 'Finger Turn Easy (DMControl)'],
			['finger-turn-hard', 'Finger Turn Hard (DMControl)'],
			['reacher-easy', 'Reacher Easy (DMControl)'],
			['reacher-hard', 'Reacher Hard (DMControl)'],
			['mw-drawer-open', 'Drawer Open (Meta-World)'],
			['mw-pick-place', 'Pick Place (Meta-World)'],
			['mw-window-close', 'Window Close (Meta-World)'],
			['rd-open-slide', 'Open Slide (RoboDesk)'],
			['ms-pick-banana', 'Pick Banana (ManiSkill3)'],
			['ms-pick-cube', 'Pick Cube (ManiSkill3)'],
			['ms-poke-cube', 'Poke Cube (ManiSkill3)'],
			['og-point-arena', 'Point Arena (OGBench)'],
			['og-point-bottleneck', 'Point Bottleneck (OGBench)'],
			['og-point-circle', 'Point Circle (OGBench)'],
			['og-point-maze', 'Point Maze (OGBench)'],
			['og-point-var1', 'Point Maze Var 1 (OGBench)'],
			['og-point-var2', 'Point Maze Var 2 (OGBench)'],
			['og-point-spiral', 'Point Spiral (OGBench)'],
			['pygame-bird-attack', 'Bird Attack (MiniArcade)'],
			['pygame-coinrun', 'Coinrun (MiniArcade)'],
			['pygame-foraging', 'Foraging (MiniArcade)'],
			['pygame-point-maze-var1', 'Point Maze Var 1 (MiniArcade)'],
			['pygame-point-maze-var2', 'Point Maze Var 2 (MiniArcade)'],
			['pygame-point-maze-var3', 'Point Maze Var 3 (MiniArcade)'],
			['pygame-point-maze-var4', 'Point Maze Var 4 (MiniArcade)'],
			['pygame-pong', 'Pong (MiniArcade)'],
			['pygame-reacher-easy', 'Reacher (MiniArcade)'],
			['pygame-whirlpool', 'Whirlpool (MiniArcade)'],
		];

		var PROD = /(^|\.)nicklashansen\.com$/i.test(location.hostname);
		// Production WebSocket endpoints (with local fallbacks for development).
		var ENDPOINTS = PROD
			? ['wss://robertson-violation-univ-obtained.trycloudflare.com/ws']
			: ['ws://127.0.0.1:8861/ws'];

		var led = document.getElementById('demoLed');
		var ledLbl = document.getElementById('demoLedLbl');
		var select = document.getElementById('demoTask');
		var pill = document.getElementById('demoPill');
		var stage = document.getElementById('demoStage');
		var frame = document.getElementById('demoFrame');
		var overlay = document.getElementById('demoOverlay');
		var overlayMsg = document.getElementById('demoOverlayMsg');
		var btnLaunch = document.getElementById('demoLaunch');
		var pausedEl = document.getElementById('demoPaused');
		var btnReset = document.getElementById('demoResetBtn');
		var btnEnd = document.getElementById('demoEndBtn');
		var hudWasd = document.getElementById('hudWasd');

		// Touch devices have no keyboard, so swap the Space/R key hints for a note
		// that points to the touch equivalents: tap the on-screen keys to move, tap
			// the frame to pause (the reset button already works on touch).
		if (window.matchMedia && window.matchMedia('(pointer: coarse)').matches) {
			var keysEl = root.querySelector('.demo-keys');
			if (keysEl) keysEl.style.display = 'none';
			var noteEl = root.querySelector('.demo-touchnote');
			if (noteEl) noteEl.hidden = false;
		}

		/* in-frame key HUD: chips light on press; only the task's bindable
		   dims are shown. Dim per key pair mirrors the server's KEY_BINDINGS:
		   0 = left/right, 1 = up/down, 2 = A/D, 3 = W/S. */
		var hudKeys = {};
		Array.prototype.forEach.call(root.querySelectorAll('.hud-key'), function (k) {
			hudKeys[k.getAttribute('data-k')] = k;
		});
		var HUD_DIMS = { ArrowRight: 0, ArrowLeft: 0, ArrowUp: 1, ArrowDown: 1, d: 2, a: 2, w: 3, s: 3 };
		var lastActDim = -1;
		function hudPress(k, on) {
			var kk = (k.length === 1) ? k.toLowerCase() : k;
			if (hudKeys[kk]) hudKeys[kk].classList.toggle('pressed', on);
		}
		function hudClear() {
			for (var k in hudKeys) hudKeys[k].classList.remove('pressed');
		}
		function hudUpdate(actDim) {
			if (actDim === lastActDim) return;
			lastActDim = actDim;
			for (var k in hudKeys) hudKeys[k].classList.toggle('dim-off', HUD_DIMS[k] >= actDim);
			if (hudWasd) hudWasd.classList.toggle('dim-off', actDim <= 2);
		}

		var state = 'checking';        // checking|offline|idle|connecting|queued|live|ended
		// Endpoint order for this session: set at launch from the latest
		// /status poll so new sessions land on the least-loaded backend (without
		// this, early users pile onto ENDPOINTS[0] while the other idles).
		var order = ENDPOINTS.map(function (_, i) { return i; });
		var lastActive = ENDPOINTS.map(function () { return Infinity; });
		var ws = null, candIdx = 0, endReason = null, gotLive = false;
		var retryTimer = null, lastUrl = null, tasksFilled = false;
		var pendingTask = null, pendingTaskTimer = null, endedByUser = false;
		var USTATES = ['ustate-calibrating', 'ustate-green', 'ustate-yellow', 'ustate-red'];

		function inSession() { return state === 'connecting' || state === 'queued' || state === 'live'; }
		function showOverlay(msg, withLaunch, launchLabel) {
			overlayMsg.textContent = msg;
			btnLaunch.style.display = withLaunch ? 'inline-block' : 'none';
			if (launchLabel) btnLaunch.innerHTML = launchLabel;
			overlay.classList.add('show');
		}
		function hideOverlay() { overlay.classList.remove('show'); }
		// Verdict word mirrors the Act-5 scrubber: yellow → Drifting, red →
		// Hallucinating. The neutral/calibrating states show no label (the ring
		// already reads as "fine"), matching the scrubber's quiet "Stable".
		var verdictEl = document.getElementById('demoVerdict');
		var VERDICTS = { yellow: 'Drifting', red: 'Hallucinating' };
		function setUState(s) {
			USTATES.forEach(function (c) { stage.classList.remove(c); });
			if (s && s !== 'off') stage.classList.add('ustate-' + s);
			if (verdictEl) {
				var word = VERDICTS[s];
				verdictEl.hidden = !word;
				if (word) verdictEl.querySelector('.vtext').textContent = word;
			}
		}
		function setPill(msg) {
			// The 5-min cap is irrelevant to most casual sessions — surface the
			// countdown only once the last minute begins.
			var show = typeof msg.remaining === 'number' && msg.remaining <= 60;
			if (show) {
				var m = Math.floor(msg.remaining / 60), s = msg.remaining % 60;
				pill.textContent = m + ':' + String(s).padStart(2, '0') + ' left';
				pill.classList.toggle('warn', msg.remaining <= 30);
			}
			pill.style.display = show ? 'inline-block' : 'none';
		}
		function setButtons(on) {
			// Session controls (task picker + header icon buttons) only exist
			// while a session runs; the frame doubles as pause/resume.
			root.classList.toggle('is-live', on);
		}

		/* ---- availability ------------------------------------------------ */
		function statusBase(e) { return e.replace(/^ws/, 'http').replace(/\/ws$/, ''); }
		function fetchStatus(endpoint) {
			var ctl = ('AbortController' in window) ? new AbortController() : null;
			var t = ctl && setTimeout(function () { ctl.abort(); }, 2500);
			return fetch(statusBase(endpoint) + '/status', ctl ? { signal: ctl.signal } : {})
				.then(function (r) { return r.ok ? r.json() : null; })
				.catch(function () { return null; })
				.then(function (js) { if (t) clearTimeout(t); return js; });
		}
		function checkAvail() {
			if (!ENDPOINTS.length) { setAvail('offline', null); return; }
			Promise.all(ENDPOINTS.map(fetchStatus)).then(function (all) {
				var up = all.filter(function (x) { return !!x; });
				lastActive = all.map(function (s) {
					return s ? (s.active || 0) + 0.1 * (s.queued || 0) : Infinity;
				});
				if (!up.length) { setAvail('offline', null); return; }
				if (!tasksFilled && up[0].task_list) {
					tasksFilled = true;
					var served = {};
					up[0].task_list.forEach(function (t) { served[t] = true; });
					var labeled = {};
					DEMO_TASKS.forEach(function (pair) {       // curated order + labels
						if (!served[pair[0]]) return;
						labeled[pair[0]] = true;
						var o = document.createElement('option');
						o.value = pair[0]; o.textContent = pair[1];
						select.appendChild(o);
					});
					up[0].task_list.forEach(function (t) {     // served but unlabeled: raw id
						if (labeled[t]) return;
						var o = document.createElement('option');
						o.value = t; o.textContent = t;
						select.appendChild(o);
					});
					if (up[0].initial_task) select.value = up[0].initial_task;
				}
				setAvail('online', null);
			});
		}
		function setAvail(kind, info) {
			if (inSession()) return;  // session owns the header while active
			// Deliberately no capacity readout: visitors at a busy moment just
			// land in the queue and are told their position there.
			if (kind === 'online') {
				led.className = 'demo-led on';
				ledLbl.textContent = 'online';
				if (state === 'checking' || state === 'offline') {
					state = 'idle';
					showOverlay('', true, '&#9654;&#xFE0E;&ensp;Launch live demo');
				}
			} else {
				led.className = 'demo-led off';
				ledLbl.textContent = 'offline';
				if (state !== 'ended') {
					state = 'offline';
					showOverlay('The live demo is offline right now.\nPlease check back soon!', false);
				}
			}
		}
		var pollTimer = null;
		if ('IntersectionObserver' in window) {
			new IntersectionObserver(function (en) {
				if (en[0].isIntersecting) {
					checkAvail();
					if (!pollTimer) pollTimer = setInterval(function () { if (!inSession()) checkAvail(); }, 20000);
				} else if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
			}, { rootMargin: '400px 0px' }).observe(root);
		} else { checkAvail(); }

		/* ---- input capture: keyboard + on-screen keys (only while live) ---- */
		// The physical keyboard and the clickable/tappable HUD caps are two
		// independent sources that can hold the same key — or opposing keys — at
		// once. We track each source separately and emit a keydown/keyup to the
		// server only when a key's *combined* (union) held-state flips, so
		// releasing one source never clears a key the other still holds, and a
		// repeat from one source isn't sent twice. Opposing keys (e.g. Left +
		// Right) cancel server-side in build_action_from_keys, so forwarding the
		// true union is all that's needed for keyboard/button inputs to cancel.
		var kbHeld = {}, ptrHeld = {};
		function keyName(e) { return e.key === ' ' ? 'Space' : e.key; }
		// Canonical key id (and wire form): lowercase single chars so 'w'/'W' and
		// a 'w' cap collapse to one held entry. Server treats cases identically.
		function canon(k) { return k.length === 1 ? k.toLowerCase() : k; }
		function captured(k) {
			return ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
				'w', 'W', 'a', 'A', 's', 'S', 'd', 'D', 'Space', 'r', 'R', 'q', 'Q'].indexOf(k) !== -1;
		}
		function heldAnywhere(k) { return kbHeld[k] === true || ptrHeld[k] === true; }
		function applyKey(src, k, on) {
			var was = heldAnywhere(k);
			if (on) src[k] = true; else delete src[k];
			var now = heldAnywhere(k);
			if (now === was) return;     // key-repeat, or the other source still holds it
			hudPress(k, now);
			send({ type: now ? 'keydown' : 'keyup', key: k });
		}
		function releaseAll() {
			// Panic release of both sources (window blur / disconnect) so a key
			// can't stick server-side — it would never be resent. applyKey is
			// idempotent, so releasing from both dicts emits at most one keyup.
			var union = {};
			for (var a in kbHeld) union[a] = true;
			for (var b in ptrHeld) union[b] = true;
			for (var k in union) { applyKey(kbHeld, k, false); applyKey(ptrHeld, k, false); }
		}
		function onKeyDown(e) {
			var k = keyName(e);
			if (!captured(k)) return;
			e.preventDefault();
			applyKey(kbHeld, canon(k), true);
		}
		function onKeyUp(e) {
			var k = keyName(e);
			if (!captured(k)) return;
			e.preventDefault();
			applyKey(kbHeld, canon(k), false);
		}
		function onBlur() { releaseAll(); }
		var keysOn = false;
		function attachKeys() {
			if (keysOn) return;
			keysOn = true;
			window.addEventListener('keydown', onKeyDown, { passive: false });
			window.addEventListener('keyup', onKeyUp, { passive: false });
			window.addEventListener('blur', onBlur);
		}
		function detachKeys() {
			if (!keysOn) return;
			keysOn = false;
			window.removeEventListener('keydown', onKeyDown);
			window.removeEventListener('keyup', onKeyUp);
			window.removeEventListener('blur', onBlur);
			kbHeld = {}; ptrHeld = {};
		}

		// On-screen keys: a cap's data-k is already the canonical wire form
		// (lowercase letter / "Arrow…"), so clicking/tapping drives the exact
		// same keydown/keyup path as the physical key. Pointer events unify
		// mouse + touch; setPointerCapture keeps the release bound to the cap if
		// the finger/cursor drifts off it; stopPropagation stops the tap from
		// also reaching the stage's tap-to-pause handler.
		function capDown(e) {
			if (!keysOn) return;     // ignore unless a session is live
			var k = e.currentTarget.getAttribute('data-k');
			if (!k) return;
			e.preventDefault();
			e.stopPropagation();
			if (e.currentTarget.setPointerCapture && e.pointerId != null) {
				try { e.currentTarget.setPointerCapture(e.pointerId); } catch (_) {}
			}
			applyKey(ptrHeld, k, true);
		}
		function capUp(e) {
			var k = e.currentTarget.getAttribute('data-k');
			if (!k) return;
			e.stopPropagation();
			applyKey(ptrHeld, k, false);
		}
		Array.prototype.forEach.call(root.querySelectorAll('.hud-key'), function (cap) {
			cap.addEventListener('pointerdown', capDown);
			cap.addEventListener('pointerup', capUp);
			cap.addEventListener('pointercancel', capUp);
		});
		// Swallow clicks anywhere on the key-overlay box (keys AND the gaps between
		// them) so a near-miss can't reach the stage's tap-to-pause. Clicks on the
		// rest of the frame still pause, since .demo-hud is shrink-wrapped to the keys.
		var demoHudEl = root.querySelector('.demo-hud');
		if (demoHudEl) demoHudEl.addEventListener('click', function (e) { e.stopPropagation(); });

		/* ---- session ------------------------------------------------------ */
		function send(obj) {
			if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
		}
		function tryNext(steppingMsg, exhaustedMsg) {
			candIdx += 1;
			if (candIdx < order.length) {
				showOverlay(steppingMsg, false);
				connect(candIdx);
			} else {
				candIdx = 0;
				state = 'ended';
				showOverlay(exhaustedMsg, true, 'Try again');
				retryTimer = setTimeout(function () { connect(0); }, 15000);
			}
		}
		function onClosed() {
			detachKeys();
			hudClear();
			lastActDim = -1;
			setButtons(false);
			setUState('off');
			pill.style.display = 'none';
			pausedEl.classList.remove('show');
			if (endReason === 'full') {
				tryNext('Server full, trying another…',
					'Queue is full right now.\nRetrying in 15 seconds…');
				return;
			}
			if (!endReason && !gotLive && !endedByUser) {
				tryNext('Server unreachable, trying another…',
					"Unable to reach demo right now.\nRetrying in 15 seconds…");
				return;
			}
			state = 'ended';
			var msgs = {
				time: 'Session limit reached, thanks for playing!',
				idle: 'Disconnected after inactivity.',
				oom: 'We got interrupted, please try again.',
			};
			var m = endedByUser ? 'Session ended, thanks for playing!' : (msgs[endReason] || 'Disconnected.');
			showOverlay(m, true, 'Play again');
			checkAvail();
		}
		function connect(i) {
			if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
			candIdx = i;
			endReason = null;
			gotLive = false;
			endedByUser = false;
			pendingTask = null;
			state = 'connecting';
			showOverlay('connecting…', false);
			if (i === 0) {
				// Fresh launch: route to the least-loaded backend per the latest poll.
				order = ENDPOINTS.map(function (_, k) { return k; })
					.sort(function (a, b) { return lastActive[a] - lastActive[b]; });
			}
			ws = new WebSocket(ENDPOINTS[order[i]]);
			ws.binaryType = 'arraybuffer';
			ws.onclose = onClosed;
			ws.onerror = function () {};
			ws.onmessage = function (ev) {
				if (typeof ev.data === 'string') {
					var msg;
					try { msg = JSON.parse(ev.data); } catch (e) { return; }
					gotLive = true;
					if (msg.type === 'queue') {
						state = 'queued';
						showOverlay('In queue — position ' + msg.position + ' of ' + msg.of +
							'.\nA slot frees up within 5 minutes.', false);
					} else if (msg.type === 'granted') {
						showOverlay('Slot granted — starting session…', false);
					} else if (msg.type === 'full') {
						endReason = 'full';
					} else if (msg.type === 'end') {
						endReason = msg.reason || 'end';
					} else if (msg.type === 'status') {
						if (state !== 'live') {
							state = 'live';
							hideOverlay();
							attachKeys();
							setButtons(true);
							led.className = 'demo-led on';
							ledLbl.textContent = 'live';
						}
						var serverTask = (typeof msg.task === 'string') ? msg.task : null;
						if (pendingTask && serverTask === pendingTask) pendingTask = null;
						if (serverTask && !pendingTask && select.value !== serverTask) select.value = serverTask;
						if (typeof msg.paused === 'boolean') pausedEl.classList.toggle('show', msg.paused);
						if (typeof msg.u_state === 'string') setUState(msg.u_state);
						if (typeof msg.act_dim === 'number') hudUpdate(msg.act_dim);
						setPill(msg);
					}
					return;
				}
				var blob = new Blob([ev.data], { type: 'image/jpeg' });
				var url = URL.createObjectURL(blob);
				// Revoke the previous frame's URL now — it has already decoded by the time the
				// next frame arrives. Revoking in frame.onload leaked every frame whose load was
				// superseded by a newer src before it fired (routine at 10–15 fps).
				if (lastUrl) URL.revokeObjectURL(lastUrl);
				lastUrl = url;
				frame.src = url;
			};
		}

		btnLaunch.addEventListener('click', function () { connect(0); });
		stage.addEventListener('click', function () {
			if (state === 'live' && !overlay.classList.contains('show')) send({ type: 'toggle_pause' });
		});
		btnReset.addEventListener('click', function () { send({ type: 'reset' }); });
		btnEnd.addEventListener('click', function () {
			endedByUser = true;
			send({ type: 'disconnect' });
		});
		select.addEventListener('change', function () {
			var t = select.value;
			pendingTask = t;
			if (pendingTaskTimer) clearTimeout(pendingTaskTimer);
			pendingTaskTimer = setTimeout(function () { if (pendingTask === t) pendingTask = null; }, 10000);
			send({ type: 'set_task', task: t });
		});
	}

	function initLazyVideos() {
		var vids = document.querySelectorAll('video[data-lazy]');
		if (!vids.length) return;
		function load(v) {
			if (v.dataset.loaded) return;
			v.querySelectorAll('source[data-src]').forEach(function (s) { s.src = s.getAttribute('data-src'); });
			v.load();
			if (!reduceMotion) {  // honor prefers-reduced-motion: load the first frame but don't auto-play
				var p = v.play();
				if (p && p.catch) p.catch(function () {});
			}
			v.dataset.loaded = '1';
		}
		if (!('IntersectionObserver' in window)) { vids.forEach(load); return; }
		var io = new IntersectionObserver(function (entries) {
			entries.forEach(function (e) {
				if (e.isIntersecting) { load(e.target); io.unobserve(e.target); }
			});
		}, { rootMargin: '200px 0px' });
		vids.forEach(function (v) { io.observe(v); });
	}

	function initFtMore() {
		var btn = document.getElementById('ftMoreBtn'), more = document.getElementById('ftMore');
		if (!btn || !more) return;
		btn.addEventListener('click', function () {
			var open = more.classList.toggle('open');
			btn.setAttribute('aria-expanded', open ? 'true' : 'false');
			btn.querySelector('.ft-more-label').textContent = open ? 'Show fewer examples' : 'Show more examples';
			if (!open) return;
			more.querySelectorAll('video[data-lazy]').forEach(function (v) {
				if (v.dataset.loaded) return;
				v.querySelectorAll('source[data-src]').forEach(function (sc) { sc.src = sc.getAttribute('data-src'); });
				v.load();
				if (!reduceMotion) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
				v.dataset.loaded = '1';
			});
		});
	}

	function initBibCopy() {
		var btn = document.getElementById('bibCopyBtn');
		var src = document.getElementById('bibtex-text');
		if (!btn || !src) return;
		var label = btn.querySelector('.bib-copy-label');
		var resetTimer;
		btn.addEventListener('click', function () {
			var text = src.textContent.trim();
			function done() {
				btn.classList.add('copied');
				if (label) label.textContent = 'Copied';
				clearTimeout(resetTimer);
				resetTimer = setTimeout(function () {
					btn.classList.remove('copied');
					if (label) label.textContent = 'Copy';
				}, 1800);
			}
			if (navigator.clipboard && navigator.clipboard.writeText) {
				navigator.clipboard.writeText(text).then(done, fallback);
			} else { fallback(); }
			function fallback() {
				var ta = document.createElement('textarea');
				ta.value = text; ta.setAttribute('readonly', '');
				ta.style.position = 'absolute'; ta.style.left = '-9999px';
				document.body.appendChild(ta);
				ta.select();
				try { document.execCommand('copy'); done(); } catch (e) {}
				document.body.removeChild(ta);
			}
		});
	}

	function initHeroMosaic() {
		// The hero clips use native autoplay. Pause them once the hero scrolls out of view so
		// phones aren't decoding ~16 unseen videos for the whole article (battery / data / decoder
		// pressure). Tiles hidden on small screens (display:none) stay paused even while in view.
		var hero = document.querySelector('.hero');
		var vids = hero && hero.querySelectorAll('.hero-mosaic video');
		if (!hero || !vids || !vids.length || !('IntersectionObserver' in window)) return;
		new IntersectionObserver(function (ents) {
			var visible = ents[0].isIntersecting;
			vids.forEach(function (v) {
				var shouldPlay = visible && v.offsetParent !== null;
				if (shouldPlay) {
					if (v.paused) { var p = v.play(); if (p && p.catch) p.catch(function () {}); }
				} else if (!v.paused) { v.pause(); }
			});
		}, { threshold: 0 }).observe(hero);
	}
})();
