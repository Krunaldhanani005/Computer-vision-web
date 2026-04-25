/* ============================================================
   COMPUTER VISION PLATFORM — platform.js
   ============================================================ */

document.addEventListener('DOMContentLoaded', () => {

  /* ── Active sidebar highlight ────────────────────────────── */
  const path = window.location.pathname.replace(/\/+$/, '') || '/';

  const map = {
    '/':                  'home',
    '/live-demo':         'demo',
    '/face-recognition':  'demo',
    '/emotion-detection': 'demo',
    '/object-detection':  'demo',
    '/restricted-area':   'demo',
    '/report':            'report',
    '/details':           'details',
    '/license':           'license',
  };

  const activeKey = map[path];
  if (activeKey) {
    document.querySelectorAll('.sb-item').forEach(el => {
      if (el.dataset.page === activeKey) el.classList.add('active');
      else el.classList.remove('active');
    });
  }

  /* ── Intersection Observer — animate cards on scroll ──────── */
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.style.opacity = '1';
          e.target.style.transform = 'translateY(0)';
          observer.unobserve(e.target);
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll('.feat-card, .service-card, .stat-card, .info-card').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity .45s ease, transform .45s ease';
    observer.observe(el);
  });

  /* ── Report page: load alerts on page load ─────────────────── */
  if (path === '/report') {
    loadAlerts();
    loadAlertStats();

    const refreshBtn = document.getElementById('report-refresh-btn');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => {
        loadAlerts();
        loadAlertStats();
      });
    }
  }

  function loadAlerts() {
    const tbody = document.getElementById('report-alerts-body');
    const countEl = document.getElementById('report-alerts-count');
    if (!tbody) return;

    fetch('/get_alerts')
      .then(r => r.json())
      .then(alerts => {
        if (countEl) countEl.textContent = alerts.length + ' records';
        if (!alerts.length) {
          tbody.innerHTML = `
            <tr>
              <td colspan="3">
                <div class="p-empty">
                  <svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                  <p class="p-empty-text">No intrusion alerts recorded yet.</p>
                </div>
              </td>
            </tr>`;
          return;
        }
        tbody.innerHTML = alerts.map(a => `
          <tr>
            <td>${escHtml(a.timestamp || '—')}</td>
            <td>
              <span class="p-badge p-badge--danger">
                &#9888; ${escHtml(a.status || 'INTRUDER')}
              </span>
            </td>
            <td>
              ${a.image_path
                ? `<a href="/${escHtml(a.image_path)}" target="_blank" style="color:var(--p-accent);font-weight:600;font-size:.8rem;">View Snapshot</a>`
                : '<span style="color:var(--p-text3);">—</span>'}
            </td>
          </tr>`).join('');

        /* update stat card */
        const alertStatEl = document.getElementById('stat-alerts-val');
        if (alertStatEl) alertStatEl.textContent = alerts.length;
      })
      .catch(() => {
        if (tbody) tbody.innerHTML = `<tr><td colspan="3"><div class="p-empty"><p class="p-empty-text">Could not load alerts.</p></div></td></tr>`;
      });
  }

  function loadAlertStats() {
    /* derive stats from alerts */
    fetch('/get_alerts')
      .then(r => r.json())
      .then(alerts => {
        const today = new Date().toDateString();
        const todayCount = alerts.filter(a => {
          try { return new Date(a.timestamp).toDateString() === today; } catch { return false; }
        }).length;

        const todayEl = document.getElementById('stat-today-val');
        if (todayEl) todayEl.textContent = todayCount;
      })
      .catch(() => {});
  }

  function escHtml(str) {
    const d = document.createElement('div');
    d.appendChild(document.createTextNode(str));
    return d.innerHTML;
  }

});
