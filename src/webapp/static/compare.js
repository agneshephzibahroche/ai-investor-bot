// Compare page
(() => {
  const { api, esc, pct, plainPct, cls, isNum, plotLayout, plot, SERIES, attachSearch, errorBox, stockUrl } = AI;
  const $ = (id) => document.getElementById(id);
  const MAX = 6;
  const params = new URLSearchParams(location.search);
  let symbols = (params.get('symbols') || 'AAPL,MSFT,NVDA,SPY').split(',').map((s) => s.trim().toUpperCase()).filter(Boolean).slice(0, MAX);
  let range = params.get('range') || '1y';
  let data = null;
  const color = (s) => SERIES[symbols.indexOf(s) % SERIES.length];

  function syncUrl() {
    history.replaceState(null, '', `/compare?symbols=${encodeURIComponent(symbols.join(','))}&range=${range}`);
  }

  function renderTags() {
    $('tags').querySelectorAll('.tag').forEach((t) => t.remove());
    const form = $('cmp-search');
    symbols.forEach((s) => {
      const t = document.createElement('span');
      t.className = 'tag'; t.style.background = color(s);
      t.innerHTML = `${esc(s)} <button aria-label="Remove ${esc(s)}">×</button>`;
      t.querySelector('button').addEventListener('click', () => { symbols = symbols.filter((x) => x !== s); refresh(); });
      $('tags').insertBefore(t, form);
    });
    form.classList.toggle('hidden', symbols.length >= MAX);
  }

  function corrColor(v) {
    if (!isNum(v)) return 'transparent';
    return v >= 0 ? `rgba(124,140,255,${0.15 + v * 0.75})` : `rgba(242,85,90,${0.15 + -v * 0.75})`;
  }

  function render() {
    const syms = Object.keys(data.series);
    plot('cmp-chart', syms.map((s) => ({
      x: data.dates, y: data.series[s], mode: 'lines', name: s, line: { color: color(s), width: 2 },
      hovertemplate: `${s}: %{y:.1f}<extra></extra>`,
    })), plotLayout({ yaxis: { tickformat: ',.0f' }, shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 100, y1: 100, line: { color: AI.css('--faint'), dash: 'dot', width: 1 } }] }));

    const ranked = syms.slice().sort((a, b) => data.stats[b].return - data.stats[a].return);
    $('cmp-stats').innerHTML = `<thead><tr><th>Ticker</th><th>Return</th><th>${AI.term('volatility', 'Volatility')}</th><th>${AI.term('sharpe', 'Sharpe')}</th><th>${AI.term('drawdown', 'Max drawdown')}</th></tr></thead><tbody>
      ${ranked.map((s, i) => { const st = data.stats[s]; return `<tr>
        <td><span class="swatch" style="background:${color(s)}"></span><a href="${stockUrl(s)}" class="num">${esc(s)}</a>${i === 0 ? ' 🏆' : ''}</td>
        <td class="num ${cls(st.return)}">${pct(st.return, 1, true)}</td><td class="num">${plainPct(st.volatility)}</td>
        <td class="num">${isNum(st.sharpe) ? st.sharpe.toFixed(2) : '—'}</td><td class="num down">${plainPct(st.max_drawdown)}</td></tr>`; }).join('')}</tbody>`;

    $('cmp-corr').innerHTML = `<thead><tr><th></th>${syms.map((s) => `<th style="text-align:center">${esc(s)}</th>`).join('')}</tr></thead><tbody>
      ${syms.map((r) => `<tr><td class="num">${esc(r)}</td>${syms.map((c) => { const v = data.correlation[r][c];
        const ink = isNum(v) && Math.abs(v) > 0.5 ? '#fff' : 'var(--text)';
        return `<td style="background:${corrColor(v)};color:${ink}">${isNum(v) ? v.toFixed(2) : '—'}</td>`; }).join('')}</tr>`).join('')}</tbody>`;
  }

  async function refresh() {
    renderTags(); syncUrl();
    if (symbols.length < 2) {
      $('cmp-chart').innerHTML = '<div class="empty" style="padding-top:120px">Add at least two tickers to compare.</div>';
      $('cmp-stats').innerHTML = ''; $('cmp-corr').innerHTML = '';
      return;
    }
    try {
      data = await api(`/api/compare?symbols=${encodeURIComponent(symbols.join(','))}&range=${range}`);
      if (data.missing.length) AI.toast(`No data for ${data.missing.join(', ')}`);
      if (!data.dates.length) throw new Error('These tickers have no overlapping history in this range.');
      render();
    } catch (e) { errorBox($('cmp-chart'), e); }
  }

  attachSearch($('cmp-search'), (sym, input) => {
    input.value = '';
    if (!symbols.includes(sym) && symbols.length < MAX) { symbols.push(sym); refresh(); }
  });
  $('presets').addEventListener('click', (e) => {
    const b = e.target.closest('[data-p]'); if (!b) return;
    symbols = b.dataset.p.split(','); refresh();
  });
  $('range').querySelectorAll('button').forEach((b) => b.classList.toggle('on', b.dataset.v === range));
  $('range').addEventListener('click', (e) => {
    const b = e.target.closest('button'); if (!b) return;
    $('range').querySelectorAll('button').forEach((x) => x.classList.toggle('on', x === b));
    range = b.dataset.v; refresh();
  });
  document.addEventListener('themechange', () => data && render());
  refresh();
})();
