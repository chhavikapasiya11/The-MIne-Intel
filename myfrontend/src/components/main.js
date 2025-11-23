// main.js — plain JavaScript (no JSX)
// Updated: Homepage-only build. Heavy implementations for PredictRate, BestParameters and GraphAnalysis have been removed
// as requested — you will handle routing and full-page implementations elsewhere.

import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
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

  // onClick handlers are empty/no-op — you said you'll manage routing
  return el('div', { className: 'page page-enter' },
    el('h2', { className: 'page-title',style: { textAlign: 'center', fontSize: '28px',padding:'0px' } }, 'Roof Fate Rate — Quick Actions'),
    el('div', { className: 'home-grid', style: { display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 22, alignItems: 'start', marginTop: 8 } },
     card(
  'Predict Rate',
  'Run model predictions quickly. Upload data on your dedicated predict page or trigger a run programmatically.',
  function () {
    window.location.href = '/predict';
  },
  '⚡'
),

card(
  'Impactful Parameters',
  'Explore feature importance and recommended parameter settings to reduce roof fate rate in the field.',
  function () {
    window.location.href = '/params';
  },
  '🛠️'
),

card(
  'Graph Analysis',
  'View interactive charts, residuals and time trends. Swap the placeholder with Chart.js where you like.',
  function () {
    window.location.href = '/graphs';
  },
  '📈'
)

    )
  );
}

/* ---------------- Minimal Chat Widget with Mic ---------------- */
function ChatWidget() {
  var _useState = useState(false), open = _useState[0], setOpen = _useState[1];
  var _useState2 = useState([{ from: 'bot', text: 'Hi — I can help run predictions, explain parameters or show charts. Ask me anything about roof fate rate.' }]), messages = _useState2[0], setMessages = _useState2[1];
  var _useState3 = useState(''), value = _useState3[0], setValue = _useState3[1];
  var _useState4 = useState(false), listening = _useState4[0], setListening = _useState4[1];
  var recognitionRef = useRef(null);

  useEffect(function () {
    var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    var rec = new SpeechRecognition(); rec.lang = 'en-US'; rec.interimResults = true; rec.maxAlternatives = 1;

    rec.onresult = function (ev) {
      var interim = ''; var final = '';
      for (var i = 0; i < ev.results.length; i++) { var res = ev.results[i]; if (res.isFinal) final += res[0].transcript; else interim += res[0].transcript; }
      setValue(function (prev) { return final ? final : interim; });
    };
    rec.onend = function () { setListening(false); };
    rec.onerror = function (e) { console.warn('Speech recognition error', e); setListening(false); };

    recognitionRef.current = rec;
    return function () { try { rec.stop(); } catch (e) { } };
  }, []);

  function toggleListening() {
    var rec = recognitionRef.current; if (!rec) { alert('Speech recognition not supported in this browser.'); return; }
    if (!listening) { try { rec.start(); setListening(true); } catch (e) { console.warn('start error', e); } } else { rec.stop(); setListening(false); }
  }

  function send(txt) {
    if (!txt) return; setMessages(function (m) { return m.concat({ from: 'user', text: txt }); }); setValue('');
    setTimeout(function () { setMessages(function (m) { return m.concat({ from: 'bot', text: 'I received: "' + txt + '" — POST data to /api/predict and return { rate, confidence }.' }); }); }, 600);
  }

  return el('div', { className: 'chat-widget' },
    el('div', { style: { display: 'flex', alignItems: 'center', gap: 8 } },
      el('button', { 'aria-label': 'Voice input', title: listening ? 'Stop listening' : 'Start voice input', onClick: toggleListening, style: { background: 'transparent', border: 'none', color: 'var(--accent)', fontSize: 20, cursor: 'pointer', padding: 8, borderRadius: 8 } }, listening ? '🎤⏺' : '🎤'),
      el('button', { 'aria-label': 'Open chat', className: 'chat-toggle', onClick: function () { setOpen(!open); } }, open ? 'Close Chat' : 'Chat')
    ),
    open && el('div', { className: 'chat-panel card', style: { marginTop: 10 } },
      el('div', { className: 'chat-header' }, 'Assistant'),
      el('div', { className: 'chat-log', role: 'log', 'aria-live': 'polite', style: { maxHeight: 260 } },
        messages.map(function (m, i) { return el('div', { key: i, className: 'chat-msg ' + (m.from === 'user' ? 'user' : 'bot') }, m.text); })
      ),
      el('form', { onSubmit: function (e) { e.preventDefault(); send(value); } },
        el('input', { placeholder: 'Ask about predictions or parameters...', value: value, onChange: function (e) { setValue(e.target.value); } }),
        el('button', { type: 'submit', className: 'btn btn-primary' }, 'Send')
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
    el(ChatWidget),
    el('div', { style: { position: 'fixed', left: 18, bottom: 18, color: 'var(--muted)', fontSize: 12 } }, )
  );
}
