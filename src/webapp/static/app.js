// Shared helpers: formatting, API calls, search, watchlist, theme and the ticker tape.
const AI = (() => {
  // ---------------------------------------------------------------- storage
  const store = {
    get(key, fallback) {
      try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); }
      catch (e) { return fallback; }
    },
    set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch (e) {} },
  };

  // ---------------------------------------------------------------- formatting
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const isNum = (v) => typeof v === 'number' && isFinite(v);

  function price(v, currency, type) {
    if (!isNum(v)) return '—';
    const digits = Math.abs(v) >= 1000 ? 2 : Math.abs(v) >= 1 ? 2 : 4;
    const opts = { minimumFractionDigits: digits, maximumFractionDigits: digits };
    if (currency && type !== 'INDEX' && currency.length === 3) {
      try { return new Intl.NumberFormat('en-US', { style: 'currency', currency, ...opts }).format(v); } catch (e) {}
    }
    return new Intl.NumberFormat('en-US', opts).format(v);
  }
  const pct = (v, digits = 2, isFraction = false) => {
    if (!isNum(v)) return '—';
    const x = isFraction ? v * 100 : v;
    return (x > 0 ? '+' : '') + x.toFixed(digits) + '%';
  };
  const plainPct = (v, digits = 1) => (isNum(v) ? (v * 100).toFixed(digits) + '%' : '—');
  const compact = (v) => (isNum(v) ? new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(v) : '—');
  const cls = (v) => (!isNum(v) || v === 0 ? '' : v > 0 ? 'up' : 'down');
  const ago = (unix) => {
    const s = Date.now() / 1000 - unix;
    if (s < 3600) return Math.max(1, Math.round(s / 60)) + 'm ago';
    if (s < 86400) return Math.round(s / 3600) + 'h ago';
    return Math.round(s / 86400) + 'd ago';
  };
  const stockUrl = (sym) => '/stock/' + encodeURIComponent(sym);

  // ---------------------------------------------------------------- api
  async function api(url) {
    const res = await fetch(url, { headers: { Accept: 'application/json' } });
    let body = null;
    try { body = await res.json(); } catch (e) {}
    if (!res.ok) throw new Error((body && body.error) || `Request failed (${res.status})`);
    return body;
  }

  // ---------------------------------------------------------------- visuals
  function sparkline(values, opts = {}) {
    const v = (values || []).filter(isNum);
    const w = opts.width || 160, h = opts.height || 34;
    if (v.length < 2) return `<svg viewBox="0 0 ${w} ${h}"></svg>`;
    const min = Math.min(...v), max = Math.max(...v), span = max - min || 1;
    const base = opts.base ?? v[0];
    const color = v[v.length - 1] >= base ? 'var(--up)' : 'var(--down)';
    const pts = v.map((y, i) => `${(i / (v.length - 1) * w).toFixed(1)},${(h - 2 - (y - min) / span * (h - 4)).toFixed(1)}`);
    const id = 'g' + Math.random().toString(36).slice(2, 8);
    return `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none" aria-hidden="true">
      <defs><linearGradient id="${id}" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stop-color="${color}" stop-opacity=".25"/><stop offset="1" stop-color="${color}" stop-opacity="0"/></linearGradient></defs>
      <polygon points="0,${h} ${pts.join(' ')} ${w},${h}" fill="url(#${id})"/>
      <polyline points="${pts.join(' ')}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>`;
  }

  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const SERIES = ['#7c8cff', '#3dd6d0', '#f5b84a', '#f2555a', '#b57cff', '#5ad16b', '#ff8ac2', '#8fb3ff'];

  function plotLayout(extra = {}) {
    const grid = css('--border'), text = css('--muted');
    const axis = { gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickfont: { color: text, size: 11 }, automargin: true };
    const out = {
      paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'Inter, system-ui, sans-serif', color: css('--text'), size: 12 },
      margin: { l: 16, r: 16, t: 10, b: 30 },
      hovermode: 'x unified',
      hoverlabel: { bgcolor: css('--panel'), bordercolor: grid, font: { color: css('--text') } },
      legend: { orientation: 'h', y: 1.1, x: 0, font: { color: text } },
      xaxis: { ...axis },
      yaxis: { ...axis, side: 'right' },
    };
    // Every x/y axis gets the themed defaults; callers override per axis
    for (const [k, v] of Object.entries(extra)) {
      out[k] = /^[xy]axis\d*$/.test(k) ? { ...axis, ...(k.startsWith('y') ? { side: 'right' } : {}), ...v } : v;
    }
    return out;
  }
  const plotConfig = { responsive: true, displaylogo: false, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d', 'toggleSpikelines'] };

  function plot(id, traces, layout) {
    const el = document.getElementById(id);
    // A placeholder or error message replaced the plot: start from a clean element
    if (el.querySelector(':scope > .skeleton, :scope > .error-box, :scope > .empty')) {
      Plotly.purge(el);
      el.innerHTML = '';
    }
    return Plotly.react(el, traces, layout, plotConfig);
  }

  function toast(msg) {
    const el = document.createElement('div');
    el.className = 'toast'; el.textContent = msg;
    document.body.appendChild(el);
    setTimeout(() => el.remove(), 2200);
  }

  function errorBox(el, err) {
    el.innerHTML = `<div class="error-box">${esc(err.message || err)}</div>`;
  }

  // ---------------------------------------------------------------- glossary tooltips
  const GLOSSARY = {
    volatility: 'How much the price swings, as a yearly %. Higher means a bumpier ride.',
    sharpe: 'Return earned per unit of risk, after the risk-free rate. Above 1 is good, below 0 lost to cash.',
    sortino: 'Like Sharpe, but only counts downside swings as risk.',
    beta: 'How strongly it moves with the S&P 500. 1 = in step, 2 = twice as much, 0.5 = half.',
    drawdown: 'The biggest fall from a peak to a later low. Shows the worst loss you could have sat through.',
    rsi: 'Relative Strength Index (0–100). Above 70 may be overbought, below 30 oversold.',
    macd: 'Momentum indicator: the gap between 12- and 26-day averages. Above its signal line is bullish.',
    ma: 'Average closing price over the last 50 or 200 days. Price above it suggests an uptrend.',
    mape: 'Average forecast error as a % of price. Lower is better.',
    naive: "Simply guessing tomorrow's price equals today's. A model must beat this to add value.",
    direction: 'How often the model correctly called whether the next day closed up or down. 50% is a coin flip.',
    montecarlo: 'Thousands of random price paths based on past volatility. The bands show where the price may land.',
    correlation: 'How closely two assets move together, from −1 (opposite) to 1 (identical). Lower means better diversification.',
    cagr: 'Compound annual growth rate: the steady yearly return that gives the same result.',
    sentiment: 'Tone of recent headlines from −1 (very negative) to +1 (very positive).',
  };
  const term = (key, label) => `<span class="term" tabindex="0" data-tip="${esc(GLOSSARY[key] || '')}">${label}</span>`;

  function initTips() {
    const tip = document.createElement('div');
    tip.className = 'tip'; tip.setAttribute('role', 'tooltip');
    document.body.appendChild(tip);
    let current = null;
    const show = (el) => {
      current = el;
      tip.textContent = el.dataset.tip;
      tip.classList.add('on');
      const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
      const left = Math.min(Math.max(8, r.left + r.width / 2 - t.width / 2), innerWidth - t.width - 8);
      const top = r.top - t.height - 8 < 8 ? r.bottom + 8 : r.top - t.height - 8;
      tip.style.left = left + 'px'; tip.style.top = top + 'px';
    };
    const hide = () => { current = null; tip.classList.remove('on'); };
    document.addEventListener('mouseover', (e) => { const el = e.target.closest('.term'); if (el) show(el); });
    document.addEventListener('mouseout', (e) => { if (e.target.closest('.term')) hide(); });
    document.addEventListener('focusin', (e) => { const el = e.target.closest('.term'); if (el) show(el); });
    document.addEventListener('focusout', (e) => { if (e.target.closest('.term')) hide(); });
    // Touch: tap a term to open (taps also fire mouseover, so never toggle), tap elsewhere to close
    document.addEventListener('click', (e) => {
      const el = e.target.closest('.term');
      if (el) { e.preventDefault(); show(el); } else if (current) hide();
    });
    addEventListener('scroll', hide, { passive: true });
  }

  // ---------------------------------------------------------------- watchlist
  const DEFAULT_WATCH = ['AAPL', 'NVDA', 'MSFT', 'TSLA', 'BTC-USD', 'SPY'];
  const watchlist = {
    all: () => store.get('watchlist', DEFAULT_WATCH),
    has: (s) => watchlist.all().includes(s),
    toggle(s) {
      const list = watchlist.all();
      const next = list.includes(s) ? list.filter((x) => x !== s) : [...list, s];
      store.set('watchlist', next);
      return next.includes(s);
    },
    remove(s) { store.set('watchlist', watchlist.all().filter((x) => x !== s)); },
  };

  // ---------------------------------------------------------------- search
  function attachSearch(form, onPick) {
    const input = form.querySelector('input');
    let box = null, items = [], sel = -1, timer = null, seq = 0;

    const close = () => { box?.remove(); box = null; sel = -1; };
    const pick = (sym) => {
      sym = (sym || '').trim().toUpperCase();
      if (!sym) return;
      close();
      if (onPick) onPick(sym, input); else location.href = stockUrl(sym);
    };
    const render = () => {
      close();
      if (!items.length) return;
      box = document.createElement('div');
      box.className = 'suggest';
      box.innerHTML = items.map((r, i) => `<a href="${stockUrl(r.symbol)}" data-sym="${esc(r.symbol)}" class="${i === sel ? 'sel' : ''}">
        <span class="sym">${esc(r.symbol)}</span><span class="nm">${esc(r.name)}</span><span class="tp">${esc(r.type || '')} · ${esc(r.exchange || '')}</span></a>`).join('');
      box.addEventListener('mousedown', (e) => {
        const a = e.target.closest('a'); if (!a) return;
        e.preventDefault(); pick(a.dataset.sym);
      });
      form.appendChild(box);
    };

    input.addEventListener('input', () => {
      clearTimeout(timer);
      const q = input.value.trim();
      if (!q) { items = []; close(); return; }
      timer = setTimeout(async () => {
        const mine = ++seq;
        try {
          const data = await api('/api/search?q=' + encodeURIComponent(q));
          if (mine !== seq) return;
          items = data.results; sel = -1; render();
        } catch (e) { items = []; close(); }
      }, 180);
    });
    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        if (!items.length) return;
        e.preventDefault();
        sel = (sel + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
        render();
      } else if (e.key === 'Escape') { close(); input.blur(); }
    });
    input.addEventListener('blur', () => setTimeout(close, 120));
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      pick(sel >= 0 ? items[sel].symbol : input.value);
    });
  }

  // ---------------------------------------------------------------- tape
  const TAPE = [['^GSPC', 'S&P 500'], ['^IXIC', 'Nasdaq'], ['^DJI', 'Dow'], ['^VIX', 'VIX'], ['BTC-USD', 'Bitcoin'],
    ['ETH-USD', 'Ether'], ['GC=F', 'Gold'], ['CL=F', 'Oil'], ['AAPL', 'AAPL'], ['MSFT', 'MSFT'], ['NVDA', 'NVDA'],
    ['AMZN', 'AMZN'], ['GOOGL', 'GOOGL'], ['META', 'META'], ['TSLA', 'TSLA'], ['JPM', 'JPM']];

  async function loadTape() {
    const track = document.querySelector('#tape .tape-track');
    if (!track) return;
    try {
      const { quotes } = await api('/api/quotes?symbols=' + encodeURIComponent(TAPE.map((t) => t[0]).join(',')));
      const label = Object.fromEntries(TAPE);
      const html = quotes.map((q) => `<a class="tape-item" href="${stockUrl(q.symbol)}"><b>${esc(label[q.symbol] || q.symbol)}</b>
        <span class="num">${price(q.price, null)}</span><span class="num ${cls(q.change_pct)}">${pct(q.change_pct)}</span></a>`).join('');
      track.innerHTML = html + html; // duplicated for a seamless loop
    } catch (e) { document.getElementById('tape').classList.add('hidden'); }
  }

  // ---------------------------------------------------------------- boot
  function initTheme() {
    document.getElementById('theme-toggle')?.addEventListener('click', () => {
      const root = document.documentElement;
      const next = root.dataset.theme === 'light' ? 'dark' : 'light';
      root.dataset.theme = next;
      try { localStorage.setItem('theme', next); } catch (e) {}
      document.dispatchEvent(new Event('themechange'));
    });
  }

  document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    initTips();
    document.querySelectorAll('form[data-search]').forEach((f) => {
      if (f.dataset.search !== 'custom') attachSearch(f);
    });
    document.addEventListener('keydown', (e) => {
      if (e.key === '/' && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement.tagName)) {
        e.preventDefault();
        document.querySelector('form[data-search] input')?.focus();
      }
    });
    loadTape();
    setInterval(loadTape, 60000);
  });

  return { store, esc, isNum, price, pct, plainPct, compact, cls, ago, stockUrl, api, sparkline, css, SERIES,
           plotLayout, plotConfig, plot, term, toast, errorBox, watchlist, attachSearch };
})();
