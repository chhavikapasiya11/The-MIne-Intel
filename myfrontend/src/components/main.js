// main.js — plain JavaScript (no JSX)
// Updated: Homepage-only build. Chat widget and microphone removed; footer added.

import React from 'react';
import './main.css';

/* tiny helper to create elements without JSX */
function el(type, props) {
  var args = Array.prototype.slice.call(arguments, 2);
  return React.createElement.apply(null, [type, props].concat(args));
}

/* ---------------- Home (large attractive cards) ---------------- */
function Home() {
  function card(title, desc, onClick, emoji) {
    return el('div', {
      className: 'card home-card',
      style: {
        minWidth: 340,
        maxWidth: 720,
        padding: 28,
        borderRadius: 16,
        cursor: 'pointer',
        transition: 'transform .18s cubic-bezier(.2,.9,.2,1), box-shadow .18s ease, background .18s ease',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'space-between',
        position: 'relative',
        overflow: 'hidden',
        background: 'linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.06))',
        border: '1px solid rgba(255,255,255,0.04)',
        boxShadow: '0 10px 30px rgba(2,6,23,0.45)'
      },
      onMouseOver: function (e) {
        e.currentTarget.style.transform = 'translateY(-10px) scale(1.02)';
        e.currentTarget.style.boxShadow = '0 22px 48px rgba(91,227,255,0.08)';
        var stripe = e.currentTarget.querySelector('.home-accent'); if (stripe) stripe.style.opacity = '1';
      },
      onMouseOut: function (e) {
        e.currentTarget.style.transform = 'translateY(0) scale(1)';
        e.currentTarget.style.boxShadow = '0 10px 30px rgba(2,6,23,0.45)';
        var stripe = e.currentTarget.querySelector('.home-accent'); if (stripe) stripe.style.opacity = '0.16';
      },
      onClick: onClick
    },
      el('div', { className: 'home-accent', style: { position: 'absolute', left: 0, top: 0, bottom: 0, width: 10, borderRadius: '0 0 8px 0', background: 'linear-gradient(180deg, var(--accent), var(--accent-2))', opacity: 0.16, transition: 'opacity .18s ease' } }),
      el('div', { style: { display: 'flex', gap: 16, alignItems: 'flex-start' } },
        el('div', { className: 'card-icon', style: { width: 72, height: 72, borderRadius: 14, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 32, background: 'linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))', border: '1px solid rgba(255,255,255,0.03)', boxShadow: 'inset 0 -6px 18px rgba(0,0,0,0.25)' } }, emoji),
        el('div', null,
          el('h3', { style: { margin: '0 0 8px', fontSize: 20, letterSpacing: 0.2 } }, title),
          el('div', { className: 'muted', style: { fontSize: 15, lineHeight: 1.35 } }, desc)
        )
      ),
      el('div', { style: { marginTop: 18, display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12 } },
        el('div', { style: { display: 'flex', gap: 8, alignItems: 'center' } }, el('div', { className: 'small muted' }, 'Quick access')),
        el('div', null, el('button', { className: 'btn btn-primary', style: { padding: '10px 16px', borderRadius: 12, fontWeight: 700, boxShadow: '0 6px 18px rgba(91,227,255,0.06)' } }, 'Open'))
      )
    );
  }

  // onClick handlers are no-op navigations — replace with router if desired
  return el('div', { className: 'page page-enter' },
    el('h2', { className: 'page-title', style: { textAlign: 'center', fontSize: '28px', padding: '0px' } }, 'Roof Fate Rate — Quick Actions'),
    el('div', { className: 'home-grid', style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 22, alignItems: 'start', marginTop: 8 } },
      card(
        'Predict Rate',
        'Run model predictions quickly. Upload data on your dedicated predict page or trigger a run programmatically.',
        function () { window.location.href = '/predict'; },
        '⚡'
      ),

      card(
        'Impactful Parameters',
        'Explore feature importance and recommended parameter settings to reduce roof fate rate in the field.',
        function () { window.location.href = '/params'; },
        '🛠️'
      ),

      card(
        'Graph Analysis',
        'View interactive charts, residuals and time trends. Swap the placeholder with Chart.js where you like.',
        function () { window.location.href = '/graphs'; },
        '📈'
      )

    )
  );
}

/* ---------------- App (homepage-only) ---------------- */
export default function App() {
  var bgImagePath = '/assets/assets.jpeg'; // ensure this exists in public/assets or change path

  return el('div', null,
    el('div', { className: 'site-bg', style: { backgroundImage: "url('" + bgImagePath + "')" } }, el('div', { className: 'techlines' })),
    el('div', { className: 'container' }, el(Home)),

    // Footer (removed chat widget & mic; added a small informative footer)
    el('footer', { style: { width: '100%', padding: '18px 12px', boxSizing: 'border-box', position: 'fixed', left: 0, bottom: 0, display: 'flex', justifyContent: 'center', alignItems: 'center', gap: 12, background: 'linear-gradient(180deg, rgba(0,0,0,0.25), rgba(0,0,0,0.35))', borderTop: '1px solid rgba(255,255,255,0.03)' } },
      el('div', { style: { color: 'var(--muted)', fontSize: 13, textAlign: 'center' } },
        '© ' + new Date().getFullYear() + ' Roof Fate — Built for safe mining decisions. ',
  
      )
    )
  );
}