// Portfolio tracker: holdings live in localStorage, prices and history come from the API.
(() => {
  const { api, esc, price, pct, plainPct, cls, isNum, store, plotLayout, plot, SERIES, attachSearch, errorBox, stockUrl, toast, css } = AI;
  const $ = (id) => document.getElementById(id);
  const SAMPLE = [
    { symbol: 'AAPL', shares: 15, cost: 190 }, { symbol: 'NVDA', shares: 25, cost: 120 },
    { symbol: 'MSFT', shares: 8, cost: 410 }, { symbol: 'SPY', shares: 10, cost: 560 },
    { symbol: 'BTC-USD', shares: 0.05, cost: 60000 },
  ];
  let holdings = store.get('portfolio', []);
  let range = store.get('portfolio-range', '1y');
  if (!['6mo', 'ytd', '1y', '2y', '5y'].includes(range)) range = '1y';
  let quotes = {};
  let history = null;

  const save = () => store.set('portfolio', holdings);
  const usd = (v) => price(v, 'USD');
  const signed = (v) => (isNum(v) ? (v >= 0 ? '+' : '−') + usd(Math.abs(v)) : '—');

  // ---------------------------------------------------------------- data
  async function loadQuotes() {
    if (!holdings.length) return;
    const { quotes: list } = await api('/api/quotes?symbols=' + encodeURIComponent(holdings.map((h) => h.symbol).join(',')));
    quotes = Object.fromEntries(list.map((q) => [q.symbol, q]));
  }

  async function loadHistory() {
    if (!holdings.length) return;
    const param = holdings.map((h) => `${h.symbol}:${h.shares}`).join(',');
    try {
      history = await api(`/api/portfolio?holdings=${encodeURIComponent(param)}&range=${range}`);
      if (history.missing.length) toast(`No price history for ${history.missing.join(', ')}`);
      renderPerformance();
    } catch (e) { errorBox($('perf-chart'), e); }
  }

  // ---------------------------------------------------------------- render
  function rows() {
    return holdings.map((h) => {
      const q = quotes[h.symbol] || {};
      const value = isNum(q.price) ? q.price * h.shares : null;
      const dayChange = isNum(q.change) ? q.change * h.shares : null;
      const basis = isNum(h.cost) ? h.cost * h.shares : null;
      return { ...h, q, value, dayChange, basis, gain: value != null && basis != null ? value - basis : null };
    });
  }

  function renderSummary() {
    const r = rows();
    const sum = (k) => r.reduce((a, x) => a + (x[k] ?? 0), 0);
    const total = sum('value'), day = sum('dayChange');
    const withCost = r.filter((x) => x.gain != null);
    const gain = withCost.reduce((a, x) => a + x.gain, 0), basis = withCost.reduce((a, x) => a + x.basis, 0);
    const s = history?.stats || {};
    const cards = [
      ['Total value', `<span class="v lg num">${usd(total)}</span>`],
      ["Today's change", `<span class="v num ${cls(day)}">${signed(day)} (${pct(total - day ? day / (total - day) * 100 : null)})</span>`],
      ['Total gain / loss', withCost.length ? `<span class="v num ${cls(gain)}">${signed(gain)} (${pct(basis ? gain / basis * 100 : null)})</span>` : '<span class="v muted small">Add avg costs to see this</span>'],
      [`Return (${range.toUpperCase()}) vs S&P`, isNum(s.return) ? `<span class="v num"><span class="${cls(s.return)}">${pct(s.return, 1, true)}</span> <span class="muted small">vs ${pct(s.benchmark_return, 1, true)}</span></span>` : '<span class="v">—</span>'],
    ];
    $('summary').innerHTML = cards.map(([k, v]) => `<div class="stat"><div class="k">${k}</div>${v}</div>`).join('');
  }

  function renderHoldings() {
    const r = rows();
    const total = r.reduce((a, x) => a + (x.value ?? 0), 0);
    $('holdings').innerHTML = `<thead><tr><th>Holding</th><th>Shares</th><th>Price</th><th>Today</th><th>Value</th><th>Weight</th><th>Avg cost</th><th>Gain / loss</th><th></th></tr></thead><tbody>
      ${r.sort((a, b) => (b.value ?? 0) - (a.value ?? 0)).map((x) => `<tr>
        <td><a href="${stockUrl(x.symbol)}" class="num" style="font-weight:600">${esc(x.symbol)}</a>
          <div class="tiny muted" style="max-width:200px;overflow:hidden;text-overflow:ellipsis">${esc(x.q.name || '')}</div></td>
        <td class="num">${x.shares.toLocaleString('en-US', { maximumFractionDigits: 6 })}</td>
        <td class="num">${price(x.q.price, x.q.currency, x.q.type)}</td>
        <td class="num ${cls(x.q.change_pct)}">${pct(x.q.change_pct)}</td>
        <td class="num">${usd(x.value)}</td>
        <td class="num">${total && x.value != null ? plainPct(x.value / total) : '—'}</td>
        <td class="num">${isNum(x.cost) ? usd(x.cost) : '—'}</td>
        <td class="num ${cls(x.gain)}">${x.gain != null ? `${signed(x.gain)}<div class="tiny">${pct(x.gain / x.basis * 100)}</div>` : '—'}</td>
        <td><button class="icon-btn" style="width:26px;height:26px" data-remove="${esc(x.symbol)}" title="Remove ${esc(x.symbol)}">×</button></td></tr>`).join('')}
      </tbody>`;

    plot('alloc-chart', [{
      type: 'pie', hole: 0.55, sort: true, textinfo: 'label+percent', textposition: 'inside', insidetextorientation: 'horizontal',
      labels: r.map((x) => x.symbol), values: r.map((x) => x.value ?? 0),
      marker: { colors: r.map((_, i) => SERIES[i % SERIES.length]), line: { color: css('--panel'), width: 2 } },
      hovertemplate: '%{label}: $%{value:,.2f} (%{percent})<extra></extra>',
    }], plotLayout({ showlegend: false, margin: { l: 10, r: 10, t: 10, b: 10 } }));
  }

  function renderPerformance() {
    if (!history || !history.dates.length) return;
    const traces = [{ x: history.dates, y: history.value, mode: 'lines', name: 'Portfolio', line: { color: SERIES[0], width: 2.4 },
      fill: 'tozeroy', fillcolor: 'rgba(124,140,255,0.08)' }];
    if (history.benchmark.length) traces.push({ x: history.dates, y: history.benchmark, mode: 'lines', name: 'S&P 500 (same start)', line: { color: css('--muted'), width: 1.5, dash: 'dot' } });
    const vals = history.value.concat(history.benchmark);
    plot('perf-chart', traces, plotLayout({ yaxis: { tickprefix: '$', tickformat: ',.0f', range: [Math.min(...vals) * 0.97, Math.max(...vals) * 1.02] } }));

    const s = history.stats;
    const div = s.avg_correlation == null ? '' : s.avg_correlation > 0.7 ? 'low diversification' : s.avg_correlation > 0.4 ? 'moderate diversification' : 'well diversified';
    $('risk-note').innerHTML = [
      `${AI.term('volatility', 'Volatility')} ${plainPct(s.volatility)}`, `${AI.term('sharpe', 'Sharpe')} ${isNum(s.sharpe) ? s.sharpe.toFixed(2) : '—'}`,
      `${AI.term('beta', 'Beta')} ${isNum(s.beta) ? s.beta.toFixed(2) : '—'}`, `${AI.term('drawdown', 'Max drawdown')} ${plainPct(s.max_drawdown_1y)}`,
      s.avg_correlation != null ? `${AI.term('correlation', 'Avg correlation')} ${s.avg_correlation.toFixed(2)} (${div})` : '',
    ].filter(Boolean).join(' · ');
    renderSummary();
  }

  async function refresh(withHistory = true) {
    const empty = !holdings.length;
    $('empty').classList.toggle('hidden', !empty);
    $('content').classList.toggle('hidden', empty);
    if (empty) return;
    try {
      await loadQuotes();
      renderSummary();
      renderHoldings();
    } catch (e) { errorBox($('holdings'), e); }
    if (withHistory) loadHistory();
  }

  // ---------------------------------------------------------------- events
  attachSearch($('pf-search'), (sym, input) => { input.value = sym; $('pf-shares').focus(); });

  $('add-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const symbol = $('pf-symbol').value.trim().toUpperCase();
    const shares = parseFloat($('pf-shares').value);
    const cost = parseFloat($('pf-cost').value);
    if (!/^[A-Z0-9^=.\-]{1,20}$/.test(symbol)) return toast('Enter a valid ticker');
    if (!(shares > 0)) return toast('Shares must be more than zero');
    const existing = holdings.find((h) => h.symbol === symbol);
    if (existing) {
      // Merge into one position with a share-weighted average cost
      const both = isNum(existing.cost) && isNum(cost);
      existing.cost = both ? (existing.cost * existing.shares + cost * shares) / (existing.shares + shares) : existing.cost ?? (isNum(cost) ? cost : undefined);
      existing.shares += shares;
    } else {
      holdings.push({ symbol, shares, ...(isNum(cost) ? { cost } : {}) });
    }
    save();
    e.target.reset();
    toast(`${symbol} added`);
    refresh();
  });

  $('holdings').addEventListener('click', (e) => {
    const b = e.target.closest('[data-remove]');
    if (!b) return;
    holdings = holdings.filter((h) => h.symbol !== b.dataset.remove);
    save();
    refresh();
  });

  $('sample').addEventListener('click', () => { holdings = SAMPLE.map((h) => ({ ...h })); save(); refresh(); });

  $('range').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === range));
  $('range').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    $('range').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    range = b.dataset.v;
    store.set('portfolio-range', range);
    loadHistory();
  });

  document.addEventListener('themechange', () => { if (holdings.length) { renderHoldings(); renderPerformance(); } });

  refresh();
  setInterval(() => refresh(false), 60000);
})();
