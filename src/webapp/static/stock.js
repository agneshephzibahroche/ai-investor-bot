// Stock analysis page
(() => {
  const { api, esc, price, pct, plainPct, compact, cls, ago, isNum, plotLayout, plot, css, SERIES, watchlist, toast, errorBox } = AI;
  const $ = (id) => document.getElementById(id);
  const { term } = AI;
  const SIGNAL_TERMS = { 'RSI (14)': 'rsi', MACD: 'macd', 'Price vs MA50': 'ma', 'Price vs MA200': 'ma', 'MA50 vs MA200': 'ma' };
  const page = $('page');
  const SYMBOL = page.dataset.symbol;
  const state = { range: '1Y', ctype: 'candle', ind: 'rsi', weeks: 4, years: 3, analysis: null, forecast: null, backtest: null };
  let currency = null, qtype = null;
  const money = (v) => price(v, currency, qtype);

  function seg(id, key, onChange) {
    $(id).addEventListener('click', (e) => {
      const b = e.target.closest('button'); if (!b) return;
      $(id).querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
      state[key] = isNaN(+b.dataset.v) ? b.dataset.v : +b.dataset.v;
      onChange();
    });
  }

  // ---------------------------------------------------------------- header
  function renderHead(a) {
    const q = a.quote || {};
    currency = a.currency || q.currency; qtype = q.type;
    document.title = `${SYMBOL} ${price(q.price, null)} · Smart Stocktopus`;
    const starred = watchlist.has(SYMBOL);
    const p = a.profile || {};
    $('head').innerHTML = `
      <div>
        <h1>${esc(q.name || SYMBOL)} <span class="muted num" style="font-size:18px">${esc(SYMBOL)}</span></h1>
        <div class="meta">${[q.exchange, p.sector, p.industry, currency].filter(Boolean).map(esc).join(' · ')}
          ${q.market_time ? `· updated ${ago(q.market_time)}` : ''}</div>
        <a class="verdict" id="verdict" href="#forecast-card"><span class="tag-ai">AI</span><span class="muted">Forecasting…</span></a>
      </div>
      <div style="display:flex;align-items:flex-end;gap:16px;flex-wrap:wrap">
        <div style="text-align:right">
          <div class="big-price num">${money(q.price)}</div>
          <div class="num ${cls(q.change)}" style="font-weight:600;margin-top:4px">${isNum(q.change) ? (q.change > 0 ? '+' : '') + q.change.toFixed(2) : ''} (${pct(q.change_pct)}) today</div>
        </div>
        <button class="btn ${starred ? 'starred' : ''}" id="star">${starred ? '★ Watching' : '☆ Watch'}</button>
        <a class="btn" href="/compare?symbols=${encodeURIComponent(SYMBOL + ',SPY')}">Compare</a>
      </div>`;
    $('star').addEventListener('click', () => {
      const on = watchlist.toggle(SYMBOL);
      $('star').classList.toggle('starred', on);
      $('star').textContent = on ? '★ Watching' : '☆ Watch';
      toast(on ? `${SYMBOL} added to your watchlist` : `${SYMBOL} removed from your watchlist`);
    });

    const r = a.risk || {};
    const lo = q.low_52w, hi = q.high_52w;
    const pos = isNum(lo) && isNum(hi) && hi > lo ? Math.min(100, Math.max(0, (q.price - lo) / (hi - lo) * 100)) : null;
    const stats = [
      ['Day range', `${money(q.day_low)} – ${money(q.day_high)}`],
      ['52-week range', `${money(lo)} – ${money(hi)}`, pos != null ? `<div class="range-bar"><i style="left:calc(${pos}% - 2px)"></i></div>` : ''],
      ['Volume', compact(q.volume)],
      [term('volatility', 'Volatility (1y)'), plainPct(r.volatility)],
      [term('sharpe', 'Sharpe (1y)'), isNum(r.sharpe) ? r.sharpe.toFixed(2) : '—'],
      [term('beta', 'Beta vs S&P'), isNum(r.beta) ? r.beta.toFixed(2) : '—'],
      [term('drawdown', 'Max drawdown (1y)'), plainPct(r.max_drawdown_1y)],
      [term('drawdown', 'Max drawdown (5y)'), plainPct(r.max_drawdown_5y)],
    ];
    $('stats').innerHTML = stats.map(([k, v, extra]) => `<div class="stat"><div class="k">${k}</div><div class="v num">${v}</div>${extra || ''}</div>`).join('');
    $('returns').innerHTML = Object.entries(a.returns).filter(([, v]) => v != null)
      .map(([k, v]) => `<span class="r"><b>${k}</b><span class="num ${cls(v)}">${pct(v, 1, true)}</span></span>`).join('');
  }

  // ---------------------------------------------------------------- price chart
  function rangeStart(dates) {
    const last = new Date(dates[dates.length - 1]);
    const d = new Date(last);
    const map = { '1M': () => d.setMonth(d.getMonth() - 1), '3M': () => d.setMonth(d.getMonth() - 3), '6M': () => d.setMonth(d.getMonth() - 6),
      '1Y': () => d.setFullYear(d.getFullYear() - 1), '5Y': () => d.setFullYear(d.getFullYear() - 5),
      YTD: () => { d.setMonth(0); d.setDate(1); } };
    map[state.range]();
    const iso = d.toISOString().slice(0, 10);
    const i = dates.findIndex((x) => x >= iso);
    return i < 0 ? 0 : i;
  }

  function renderPrice() {
    const h = state.analysis.history;
    const i0 = rangeStart(h.dates);
    const sl = (arr) => arr.slice(i0);
    const x = sl(h.dates);
    const up = css('--up'), down = css('--down');
    const close = sl(h.close);
    const traces = [];
    if (state.ctype === 'candle') {
      traces.push({ type: 'candlestick', x, open: sl(h.open), high: sl(h.high), low: sl(h.low), close, name: SYMBOL,
        increasing: { line: { color: up, width: 1 }, fillcolor: up }, decreasing: { line: { color: down, width: 1 }, fillcolor: down }, yaxis: 'y' });
    } else {
      traces.push({ type: 'scatter', mode: 'lines', x, y: close, name: SYMBOL, line: { color: SERIES[0], width: 2 },
        fill: 'tozeroy', fillcolor: 'rgba(124,140,255,0.08)', yaxis: 'y' });
    }
    traces.push({ type: 'scatter', mode: 'lines', x, y: sl(h.ma50), name: 'MA50', line: { color: SERIES[2], width: 1.4 }, yaxis: 'y' });
    traces.push({ type: 'scatter', mode: 'lines', x, y: sl(h.ma200), name: 'MA200', line: { color: SERIES[4], width: 1.4 }, yaxis: 'y' });

    const vols = sl(h.volume), opens = sl(h.open);
    traces.push({ type: 'bar', x, y: vols, name: 'Volume', yaxis: 'y2', showlegend: false, hovertemplate: '%{y:.3s}<extra>Volume</extra>',
      marker: { color: close.map((c, i) => (c >= opens[i] ? up : down)), opacity: 0.45 } });

    const shapes = [];
    if (state.ind === 'rsi') {
      traces.push({ type: 'scatter', mode: 'lines', x, y: sl(h.rsi), name: 'RSI', line: { color: SERIES[1], width: 1.4 }, yaxis: 'y3', showlegend: false });
      for (const lvl of [30, 70]) shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y3', y0: lvl, y1: lvl, line: { color: css('--faint'), width: 1, dash: 'dot' } });
    } else {
      const macd = sl(h.macd), sig = sl(h.signal);
      traces.push({ type: 'bar', x, y: macd.map((m, i) => (m == null || sig[i] == null ? null : m - sig[i])), name: 'Histogram', yaxis: 'y3', showlegend: false,
        marker: { color: macd.map((m, i) => (m - sig[i] >= 0 ? up : down)), opacity: 0.5 } });
      traces.push({ type: 'scatter', mode: 'lines', x, y: macd, name: 'MACD', line: { color: SERIES[1], width: 1.3 }, yaxis: 'y3', showlegend: false });
      traces.push({ type: 'scatter', mode: 'lines', x, y: sig, name: 'Signal', line: { color: SERIES[2], width: 1.3 }, yaxis: 'y3', showlegend: false });
    }

    const crypto = qtype === 'CRYPTOCURRENCY';
    plot('price-chart', traces, plotLayout({
      xaxis: { rangeslider: { visible: false }, rangebreaks: crypto ? [] : [{ bounds: ['sat', 'mon'] }], showspikes: true, spikemode: 'across', spikethickness: 1, spikecolor: css('--faint'), spikedash: 'dot' },
      yaxis: { domain: [0.42, 1], tickformat: ',.2f' },
      yaxis2: { domain: [0.27, 0.39], showgrid: false, tickformat: '.2s' },
      yaxis3: { domain: [0, 0.22], ...(state.ind === 'rsi' ? { range: [0, 100], tickvals: [30, 50, 70] } : {}) },
      shapes,
      annotations: [{ text: state.ind.toUpperCase(), xref: 'paper', yref: 'paper', x: 0, y: 0.22, showarrow: false, font: { size: 11, color: css('--muted') }, xanchor: 'left', yanchor: 'bottom' }],
    }));
  }

  // ---------------------------------------------------------------- scorecard
  function gauge(score) {
    const a = (score + 100) / 200 * Math.PI; // 0..π
    const r = 80, cx = 95, cy = 92;
    const nx = cx - r * 0.82 * Math.cos(a), ny = cy - r * 0.82 * Math.sin(a);
    const arc = (from, to, color) => {
      const x1 = cx - r * Math.cos(from), y1 = cy - r * Math.sin(from), x2 = cx - r * Math.cos(to), y2 = cy - r * Math.sin(to);
      return `<path d="M${x1} ${y1} A${r} ${r} 0 0 1 ${x2} ${y2}" stroke="${color}" stroke-width="14" fill="none"/>`;
    };
    const P = Math.PI;
    return `<svg class="gauge" viewBox="0 0 190 110" role="img" aria-label="Score ${score}">
      ${arc(0, P * 0.25, css('--down'))}${arc(P * 0.25 + 0.03, P * 0.425, 'rgba(242,85,90,.5)')}${arc(P * 0.425 + 0.03, P * 0.575, css('--border'))}
      ${arc(P * 0.575 + 0.03, P * 0.75, 'rgba(34,197,94,.5)')}${arc(P * 0.75 + 0.03, P, css('--up'))}
      <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="var(--text)" stroke-width="3" stroke-linecap="round"/>
      <circle cx="${cx}" cy="${cy}" r="6" fill="var(--text)"/>
      <text x="${cx}" y="${cy + 16}" text-anchor="middle" font-size="12" font-weight="700" font-family="JetBrains Mono, monospace">${score > 0 ? '+' : ''}${score}</text></svg>`;
  }

  function renderScorecard(sc) {
    if (!sc) { $('scorecard').innerHTML = '<div class="empty">Not enough history yet.</div>'; return; }
    const color = sc.score > 15 ? 'var(--up)' : sc.score < -15 ? 'var(--down)' : 'var(--warn)';
    $('scorecard').innerHTML = `<div class="gauge-wrap">
        <div style="text-align:center;margin:0 auto">${gauge(sc.score)}<div style="font-weight:750;font-size:20px;color:${color}">${esc(sc.rating)}</div></div>
        <div class="signals">${sc.signals.map((s) => {
          const v = s.vote > 0 ? ['pos', 'Bullish'] : s.vote < 0 ? ['neg', 'Bearish'] : ['neu', 'Neutral'];
          return `<div class="sig"><div><div>${SIGNAL_TERMS[s.name] ? term(SIGNAL_TERMS[s.name], esc(s.name)) : esc(s.name)}</div><div class="d tiny">${esc(s.detail)}</div></div><span class="vote ${v[0]}">${v[1]}</span></div>`;
        }).join('')}</div></div>`;
  }

  // ---------------------------------------------------------------- forecast
  let fcTimer = null, fcSeq = 0;
  async function loadForecast() {
    const mine = ++fcSeq;
    $('fc-summary').style.opacity = 0.5;
    try {
      const f = await api(`/api/forecast/${encodeURIComponent(SYMBOL)}?weeks=${state.weeks}`);
      if (mine !== fcSeq) return;
      state.forecast = f;
      renderForecast();
    } catch (e) {
      if (mine === fcSeq) errorBox($('fc-body'), e);
    } finally { $('fc-summary').style.opacity = 1; }
  }

  function renderForecast() {
    const f = state.forecast, h = state.analysis.history, mc = f.monte_carlo;
    const n = 90;
    const hx = h.dates.slice(-n), hy = h.close.slice(-n);
    const lastX = hx[hx.length - 1], lastY = hy[hy.length - 1];
    const fx = [lastX, ...f.dates];
    const band = (k) => [lastY, ...mc[k]];
    const traces = [
      { x: hx, y: hy, mode: 'lines', name: 'History', line: { color: css('--text'), width: 1.8 } },
      { x: fx, y: band('p95'), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: fx, y: band('p5'), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(61,214,208,0.10)', name: '90% range' },
      { x: fx, y: band('p75'), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: fx, y: band('p25'), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(61,214,208,0.20)', name: '50% range' },
      { x: fx, y: band('p50'), mode: 'lines', name: 'Monte Carlo median', line: { color: SERIES[1], width: 1.5, dash: 'dot' } },
    ];
    if (f.lstm) traces.push({ x: fx, y: [lastY, ...f.lstm.path], mode: 'lines', name: 'LSTM forecast', line: { color: SERIES[0], width: 2.6 } });
    plot('fc-chart', traces, plotLayout({
      xaxis: { rangebreaks: qtype === 'CRYPTOCURRENCY' ? [] : [{ bounds: ['sat', 'mon'] }] },
      yaxis: { tickformat: ',.2f' },
      shapes: [{ type: 'line', x0: lastX, x1: lastX, yref: 'paper', y0: 0, y1: 1, line: { color: css('--faint'), width: 1, dash: 'dot' } }],
      annotations: [{ x: lastX, y: 1, yref: 'paper', text: 'today', showarrow: false, xanchor: 'right', font: { size: 11, color: css('--muted') } }],
    }));

    $('fc-model').textContent = f.lstm ? 'LSTM + Monte Carlo' : 'Monte Carlo';
    $('verdict').innerHTML = `<span class="tag-ai">AI</span><b class="${f.signal}">${f.signal}</b>
      <span class="muted">· target <span class="num">${money(f.target)}</span>
      <span class="num ${cls(f.pct_change)}">(${pct(f.pct_change, 1, true)})</span> in ${f.weeks} wk${f.weeks > 1 ? 's' : ''}</span>`;
    const lo = mc.p5[mc.p5.length - 1], hi = mc.p95[mc.p95.length - 1];
    const sent = f.sentiment;
    $('fc-summary').innerHTML = `
      <div style="display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-top:8px">
        <span class="signal-badge ${f.signal}">${f.signal}</span>
        <div class="muted small" style="flex:1;min-width:220px">${f.model === 'lstm' ? 'LSTM' : 'Monte Carlo'} target in ${f.weeks} week${f.weeks > 1 ? 's' : ''}:
          <b class="num" style="color:var(--text)">${money(f.target)}</b> <span class="num ${cls(f.pct_change)}">(${pct(f.pct_change, 2, true)})</span>.
          Moves within ±2% are called Hold.</div>
      </div>
      <div class="kv">
        <div><div class="k">${term('montecarlo', 'Chance of finishing higher')}</div><div class="v num">${plainPct(mc.prob_up, 0)}</div></div>
        <div><div class="k">${term('montecarlo', '90% range at horizon')}</div><div class="v num" style="font-size:15px">${money(lo)} – ${money(hi)}</div></div>
        <div><div class="k">Monte Carlo median</div><div class="v num">${money(mc.p50[mc.p50.length - 1])}</div></div>
        <div><div class="k">${term('sentiment', 'News sentiment')}</div><div class="v num ${cls(sent)}">${isNum(sent) ? (sent > 0 ? '+' : '') + sent.toFixed(2) : 'n/a'}</div></div>
      </div>
      <div class="advice">${esc(f.advice)}</div>`;

    const tr = f.lstm && f.lstm.track_record;
    $('track-wrap').classList.toggle('hidden', !tr);
    if (tr) {
      plot('track-chart', [
        { x: tr.dates, y: tr.actual, mode: 'lines', name: 'Actual close', line: { color: css('--text'), width: 1.6 } },
        { x: tr.dates, y: tr.predicted, mode: 'lines', name: 'LSTM next-day call', line: { color: SERIES[0], width: 1.4 } },
      ], plotLayout({ yaxis: { tickformat: ',.2f' }, xaxis: { rangebreaks: qtype === 'CRYPTOCURRENCY' ? [] : [{ bounds: ['sat', 'mon'] }] } }));
      const m = f.lstm.meta;
      $('track-stats').innerHTML = `
        <div><div class="k">${term('direction', 'Direction right')}</div><div class="v num">${plainPct(tr.direction_acc, 0)}</div></div>
        <div><div class="k">${term('mape', 'Avg error (MAPE)')}</div><div class="v num">${plainPct(tr.mape, 2)}</div></div>
        <div><div class="k">${term('naive', 'Naive "no change" error')}</div><div class="v num">${plainPct(tr.naive_mape, 2)}</div></div>
        <div><div class="k">${f.lstm.in_training_set ? 'In training set' : 'Unseen ticker'}</div><div class="v small">${f.lstm.in_training_set ? 'Yes (older data)' : 'Model generalises'}</div></div>
        <div style="grid-column:1/-1" class="tiny muted">Universal model trained ${esc(m.trained_on || '')} on ${m.tickers.length} tickers.
          On its held-out test set it called direction right ${plainPct(m.test_direction_acc)} of the time.</div>`;
    }
  }

  // ---------------------------------------------------------------- backtest
  async function loadBacktest() {
    try {
      state.backtest = await api(`/api/backtest/${encodeURIComponent(SYMBOL)}?years=${state.years}`);
      renderBacktest();
    } catch (e) { errorBox($('bt-chart'), e); $('bt-table').innerHTML = ''; }
  }

  function renderBacktest() {
    const b = state.backtest;
    const best = b.strategies.reduce((m, s) => (s.total_return > m.total_return ? s : m));
    plot('bt-chart', b.strategies.map((s, i) => ({
      x: b.dates, y: s.equity.map((v) => v * 10000), mode: 'lines', name: s.name,
      line: { color: i === 0 ? css('--text') : SERIES[(i - 1) % SERIES.length], width: s === best ? 2.6 : 1.5 },
    })), plotLayout({ yaxis: { tickprefix: '$', tickformat: ',.0f' }, hovermode: 'x unified' }));

    $('bt-table').innerHTML = `<thead><tr><th>Strategy</th><th>Final value</th><th>Return</th><th>${term('cagr', 'CAGR')}</th><th>${term('drawdown', 'Max DD')}</th><th>${term('sharpe', 'Sharpe')}</th><th>Trades</th><th>Now</th></tr></thead><tbody>
      ${b.strategies.map((s, i) => `<tr title="${esc(s.description)}">
        <td><span class="swatch" style="background:${i === 0 ? css('--text') : SERIES[(i - 1) % SERIES.length]}"></span>${esc(s.name)}${s === best ? ' 🏆' : ''}
          <div class="tiny muted" style="margin-left:18px;white-space:normal">${esc(s.description)}</div></td>
        <td class="num">${price(s.equity[s.equity.length - 1] * 10000, 'USD')}</td>
        <td class="num ${cls(s.total_return)}">${pct(s.total_return, 1, true)}</td>
        <td class="num">${pct(s.cagr, 1, true)}</td>
        <td class="num down">${plainPct(s.max_drawdown)}</td>
        <td class="num">${isNum(s.sharpe) ? s.sharpe.toFixed(2) : '—'}</td>
        <td class="num">${s.trades ?? '—'}</td>
        <td>${s.key === 'buy_hold' ? '' : `<span class="vote ${s.in_market ? 'pos' : 'neu'}">${s.in_market ? 'In' : 'Cash'}</span>`}</td></tr>`).join('')}
      </tbody>`;
  }

  // ---------------------------------------------------------------- news
  async function loadNews() {
    try {
      const n = await api(`/api/news/${encodeURIComponent(SYMBOL)}`);
      if (!n.items.length) { $('news').innerHTML = '<div class="empty">No recent headlines.</div>'; return; }
      const s = n.sentiment ?? 0;
      const label = s > 0.2 ? 'Positive' : s < -0.2 ? 'Negative' : s > 0.05 ? 'Slightly positive' : s < -0.05 ? 'Slightly negative' : 'Neutral';
      $('news').innerHTML = `
        <div class="small" style="display:flex;justify-content:space-between"><span class="muted">Average of ${n.items.length} headlines</span><b class="${cls(s)}">${label} (${s > 0 ? '+' : ''}${s.toFixed(2)})</b></div>
        <div class="sent-bar"><i style="left:${(s + 1) / 2 * 100}%"></i></div>
        <div class="news">${n.items.map((it) => {
          const v = it.sentiment > 0.05 ? 'pos' : it.sentiment < -0.05 ? 'neg' : 'neu';
          return `<a href="${esc(it.link)}" target="_blank" rel="noopener noreferrer">
            ${it.thumbnail ? `<img src="${esc(it.thumbnail)}" alt="" loading="lazy" referrerpolicy="no-referrer" />` : ''}
            <div><div class="ttl">${esc(it.title)}</div><div class="src">${esc(it.publisher || '')} · ${it.time ? ago(it.time) : ''}
              <span class="vote ${v}">${it.sentiment > 0 ? '+' : ''}${it.sentiment.toFixed(2)}</span></div></div></a>`;
        }).join('')}</div>`;
    } catch (e) { errorBox($('news'), e); }
  }

  // ---------------------------------------------------------------- boot
  async function init() {
    seg('range', 'range', renderPrice);
    seg('ctype', 'ctype', renderPrice);
    seg('ind', 'ind', renderPrice);
    seg('bt-years', 'years', loadBacktest);
    $('weeks').addEventListener('input', (e) => {
      state.weeks = +e.target.value;
      $('weeks-label').textContent = `${state.weeks} week${state.weeks > 1 ? 's' : ''}`;
      clearTimeout(fcTimer);
      fcTimer = setTimeout(loadForecast, 250);
    });
    // The track-record chart may have been drawn while collapsed
    $('track-wrap').addEventListener('toggle', () => { if ($('track-wrap').open) Plotly.Plots.resize('track-chart'); });
    document.addEventListener('themechange', () => {
      if (state.analysis) { renderPrice(); renderScorecard(state.analysis.scorecard); }
      if (state.forecast) renderForecast();
      if (state.backtest) renderBacktest();
    });

    loadNews();
    try {
      state.analysis = await api(`/api/analysis/${encodeURIComponent(SYMBOL)}`);
    } catch (e) {
      page.innerHTML = `<div class="card" style="max-width:560px;margin:40px auto;text-align:center">
        <h2 style="margin-top:0">Couldn't load ${esc(SYMBOL)}</h2><p class="muted">${esc(e.message)}</p>
        <a class="btn primary" href="/">Back to markets</a></div>`;
      return;
    }
    renderHead(state.analysis);
    renderPrice();
    renderScorecard(state.analysis.scorecard);
    loadForecast();
    loadBacktest();
  }

  init();
})();
