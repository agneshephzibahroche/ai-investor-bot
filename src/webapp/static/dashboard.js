// Markets dashboard
(() => {
  const { api, esc, price, pct, cls, sparkline, stockUrl, watchlist, plainPct, errorBox } = AI;
  const $ = (id) => document.getElementById(id);

  function marketStatus() {
    // NYSE regular hours in New York time (holidays not accounted for)
    const ny = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }));
    const day = ny.getDay(), mins = ny.getHours() * 60 + ny.getMinutes();
    const open = day > 0 && day < 6 && mins >= 570 && mins < 960;
    $('mkt-dot').classList.toggle('open', open);
    $('mkt-status').textContent = open ? 'US market open' : 'US market closed';
  }

  const row = (q, extra = '') => `<a class="row" href="${stockUrl(q.symbol)}">
      <span class="sym">${esc(q.symbol)}</span><span class="nm">${esc(q.name || '')}</span>
      <span class="px num">${price(q.price, q.currency, q.type)}</span>
      <span class="ch num ${cls(q.change_pct)}">${pct(q.change_pct)}</span>${extra}</a>`;

  function heatColor(p) {
    const a = Math.min(Math.abs(p) / 3, 1);
    return p >= 0 ? `rgba(22,163,74,${0.55 + a * 0.45})` : `rgba(220,52,57,${0.55 + a * 0.45})`;
  }

  function renderMarket(m) {
    $('overview').innerHTML = m.overview.map((q) => `<a class="tile" href="${stockUrl(q.symbol)}">
        <div class="t-label"><span>${esc(q.label)}</span><span class="num ${cls(q.change_pct)}">${pct(q.change_pct)}</span></div>
        <div class="t-price num">${price(q.price, null)}</div>
        ${sparkline(q.spark, { base: q.prev_close })}</a>`).join('');

    $('sectors').innerHTML = m.sectors.map((q) => `<a class="heat" href="${stockUrl(q.symbol)}" style="background:${heatColor(q.change_pct || 0)}">
        <b>${esc(q.label)}</b><span>${pct(q.change_pct)}</span></a>`).join('');

    $('gainers').innerHTML = m.gainers.map((q) => row(q)).join('') || '<div class="empty">No data</div>';
    $('losers').innerHTML = m.losers.map((q) => row(q)).join('') || '<div class="empty">No data</div>';
    $('trending').innerHTML = m.trending.length
      ? m.trending.map((t, i) => `<a class="row" href="${stockUrl(t.symbol)}"><span class="muted num">#${i + 1}</span>
          <span class="sym">${esc(t.symbol)}</span><span class="nm"></span><span class="ch muted small">${t.count} forecast${t.count > 1 ? 's' : ''}</span></a>`).join('')
      : '<div class="empty">Run a forecast on any stock page and it shows up here.</div>';

    const u = m.models.find((x) => x.name === 'universal') || m.models[0];
    if (!u) { $('model-card').classList.add('hidden'); return; }
    const beats = u.test_mape != null && u.naive_mape != null ? u.test_mape <= u.naive_mape : null;
    $('model-info').innerHTML = `
      <div><p class="muted" style="margin-top:0">A two-layer LSTM reads the last 60 trading days of price, volume, moving averages, MACD and RSI,
        normalised to the latest close, and predicts the next day's return. Forecasts roll that forward day by day.
        It was trained on 10 years of ${u.tickers.length} large caps and ETFs, so it works on any ticker.</p>
        <div class="chips" style="justify-content:flex-start">${u.tickers.map((t) => `<a class="chip" href="${stockUrl(t)}">${esc(t)}</a>`).join('')}</div></div>
      <div>
        <div class="kv" style="margin-top:0">
          <div><div class="k">${AI.term('direction', 'Direction accuracy')}</div><div class="v num">${plainPct(u.test_direction_acc)}</div></div>
          <div><div class="k">${AI.term('mape', 'Next-day error (MAPE)')}</div><div class="v num">${plainPct(u.test_mape, 2)}</div></div>
          <div><div class="k">${AI.term('naive', 'Naive baseline error')}</div><div class="v num">${plainPct(u.naive_mape, 2)}</div></div>
          <div><div class="k">Test samples</div><div class="v num">${AI.compact(u.test_samples)}</div></div>
        </div>
        <div class="tiny muted">Scored on the most recent 15% of each ticker's history, which the model never saw in training. Trained ${esc(u.trained_on || '')}.</div>
      </div>
      <div class="notice"><b>Read this before trusting it.</b> Daily stock moves are mostly noise. A direction hit rate a few points above 50% is
        typical for models like this${beats === false ? ', and it does not beat the naive "tomorrow equals today" guess on price error' : ''}.
        That's why every forecast here also shows a Monte Carlo range and the model's live track record.</div>`;
  }

  async function renderWatchlist() {
    const list = watchlist.all();
    const el = $('watch-list');
    if (!list.length) { el.innerHTML = '<div class="empty">Tap ☆ on any stock page to add it here.</div>'; return; }
    try {
      const { quotes } = await api('/api/quotes?symbols=' + encodeURIComponent(list.join(',')));
      el.innerHTML = quotes.map((q) => row(q, `<button class="icon-btn" data-remove="${esc(q.symbol)}" title="Remove">×</button>`)).join('');
    } catch (e) { errorBox(el, e); }
  }

  $('watch-list').addEventListener('click', (e) => {
    const b = e.target.closest('[data-remove]');
    if (!b) return;
    e.preventDefault();
    watchlist.remove(b.dataset.remove);
    renderWatchlist();
  });

  async function load() {
    try { renderMarket(await api('/api/market')); }
    catch (e) { errorBox($('overview'), e); }
  }

  marketStatus();
  load();
  renderWatchlist();
  setInterval(() => { load(); renderWatchlist(); marketStatus(); }, 60000);
})();
