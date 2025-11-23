// Predict.js — Premium dark UI (no JSX)
import React, { useState, useRef, useEffect } from 'react';
import { extractFeaturesFromText } from './nlp';
import './main.css';
import toolsImg from './tools.jpg';

function el(type, props) {
  var args = Array.prototype.slice.call(arguments, 2);
  return React.createElement.apply(null, [type, props].concat(args));
}

export default function PredictPage() {
  const [form, setForm] = useState({
    cmrr: '50',
    prsup: '40',
    depthOfCover: '200',
    intersectionDiagonal: '5',
    miningHeight: '2.5',
  });

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);

  function update(key, val) {
    setForm((p) => ({ ...p, [key]: val }));
  }

  function resetForm() {
    setForm({
      cmrr: '50',
      prsup: '40',
      depthOfCover: '200',
      intersectionDiagonal: '5',
      miningHeight: '2.5',
    });
    setResult(null);
  }

  function validate() {
    const nums = ['cmrr', 'prsup', 'depthOfCover', 'intersectionDiagonal', 'miningHeight'];
    for (const k of nums) {
      if (form[k] === '' || isNaN(Number(form[k]))) return { ok: false, msg: `${k} must be a number` };
    }
    return { ok: true };
  }

  async function handlePredict() {
    const v = validate();
    if (!v.ok) {
      alert(v.msg);
      return;
    }

    setLoading(true);
    setResult(null);

    try {
      // Backend expects keys: CMRR, PRSUP, depth_of_ cover, intersection_diagonal, mining_hight
      const payload = {
        CMRR: Number(form.cmrr),
        PRSUP: Number(form.prsup),
        depth_of_cover: Number(form.depthOfCover),
        intersection_diagonal: Number(form.intersectionDiagonal),
        mining_hight: Number(form.miningHeight),
      };

      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!response.ok) throw new Error('Prediction API returned an error');

      const data = await response.json();

      setResult({
        rate: data.prediction.toFixed(2) + '%',
        confidence: 'N/A',
        confidenceRaw: 0
      });

    } catch (err) {
      console.error(err);
      alert('Prediction failed: ' + err.message);
    } finally {
      setLoading(false);
    }
  }

  function LabeledInput({ label, keyName, placeholder, helper, type = 'number', step, full = false }) {
    return el('div', {
      style: {
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        gridColumn: full ? '1 / -1' : 'auto'
      }
    },
      el('label', { style: { fontWeight: 800, fontSize: 14, color: 'var(--muted)' } }, label),
      el('input', {
        type,
        step,
        value: form[keyName],
        placeholder,
        onChange: (e) => update(keyName, e.target.value),
        style: {
          padding: '14px 16px',
          borderRadius: 10,
          border: '1px solid rgba(255,255,255,0.04)',
          background: 'linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.06))',
          color: 'inherit',
          fontSize: 15,
          outline: 'none',
          boxShadow: 'inset 0 -6px 14px rgba(0,0,0,0.35)'
        }
      }),
      helper ? el('div', { style: { fontSize: 13, color: 'var(--muted)', marginTop: 4 } }, helper) : null
    );
  }

  return el('div', { className: 'page page-enter', style: { padding: '28px 18px' } },

  // main container: left card + right chat
  el('div', {
    style: {
      display: 'flex',
      gap: 20,
      maxWidth: 1200,
      margin: '12px auto',
      alignItems: 'flex-start'
    }
  },

    // left: card
    el('div', { style: { flex: 2 } },
      el('div', {
        className: 'card',
        style: {
          padding: 28,
          display: 'flex',
          flexDirection: 'column',
          gap: 20,
          borderRadius: 14,
          background: 'linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.06))',
          border: '1px solid rgba(255,255,255,0.03)',
          boxShadow: '0 18px 40px rgba(2,6,23,0.6)'
        }
      },

        // header row
        el('div', { style: { display: 'flex', alignItems: 'center', gap: 16 } },
          el('div', {
            style: {
              width: 64,
              height: 64,
              borderRadius: 10,
              overflow: 'hidden',
              flexShrink: 0,
              border: '1px solid rgba(255,255,255,0.04)'
            }
          },
            el('img', { src: toolsImg, alt: 'thumbnail', style: { width: '100%', height: '100%', objectFit: 'cover', display: 'block' } })
          ),
          el('div', null,
            el('h2', { style: { margin: 0, fontSize: 24, fontWeight: 800, color: 'var(--fg)' } }, 'Mining Parameters'),
            el('div', { style: { marginTop: 4, color: 'var(--muted)', fontSize: 14 } }, 'Enter values (use units/notes below) and click Predict.')
          )
        ),

        // grid inputs
        el('div', {
          style: { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 18, marginTop: 6 }
        },
          el(LabeledInput, { label: 'CMRR', keyName: 'cmrr', placeholder: '50', helper: 'Coal Mine Roof Rating (0–100)', step: '1' }),
          el(LabeledInput, { label: 'PRSUP', keyName: 'prsup', placeholder: '40', helper: 'Percentage of roof support load (0–100)', step: '1' }),
          el(LabeledInput, { label: 'Depth Of Cover (m)', keyName: 'depthOfCover', placeholder: '200', helper: 'Depth of cover in meters', step: '0.1' }),
          el(LabeledInput, { label: 'Intersection Diagonal (m)', keyName: 'intersectionDiagonal', placeholder: '5', helper: 'Intersection diagonal in meters', step: '0.1' }),
          el(LabeledInput, { label: 'Mining Height (m)', keyName: 'miningHeight', placeholder: '2.5', helper: 'Mining height in meters', step: '0.01', full: true })
        ),

        // action buttons
        el('div', { style: { display: 'flex', gap: 12, alignItems: 'center', marginTop: 6 } },
          el('button', {
            className: 'btn btn-primary',
            onClick: handlePredict,
            disabled: loading,
            style: {
              flex: 1,
              padding: '14px 18px',
              fontSize: 16,
              fontWeight: 800,
              borderRadius: 12,
              border: 'none',
              cursor: 'pointer',
              background: 'linear-gradient(90deg, #5EE7FF 0%, #C58CFF 100%)',
              color: '#042028',
              boxShadow: '0 10px 30px rgba(97,200,255,0.08)',
              transition: 'transform .15s ease'
            }
          }, loading ? 'Predicting…' : 'Predict roof fall rate'),
          el('button', {
            onClick: resetForm,
            style: {
              padding: '12px 14px',
              borderRadius: 10,
              border: '1px solid rgba(255,255,255,0.04)',
              background: 'transparent',
              color: 'var(--muted)',
              cursor: 'pointer',
              fontWeight: 700
            }
          }, 'Reset')
        ),

        // results area
        result && el('div', {
          className: 'card result-card',
          style: {
            marginTop: 10,
            padding: 18,
            borderRadius: 12,
            background: 'linear-gradient(180deg, rgba(0,0,0,0.35), rgba(255,255,255,0.01))',
            border: '1px solid rgba(255,255,255,0.03)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 12
          }
        },
          el('div', null,
            el('div', { style: { fontSize: 18, fontWeight: 800, color: 'var(--accent)' } }, 'Predicted Roof Fate Rate:'),
            el('div', { style: { marginTop: 6, fontSize: 22, fontWeight: 900, color: 'white' } }, result.rate),
            el('div', { style: { marginTop: 8, color: 'var(--muted)' } }, 'Model confidence: ' + result.confidence)
          ),
          el('div', { style: { flexBasis: 320, flexGrow: 0 } },
            el('div', { style: { height: 12, borderRadius: 8, background: 'rgba(255,255,255,0.04)', overflow: 'hidden' } },
              el('div', {
                style: {
                  width: (result.confidenceRaw ? Math.max(0, Math.min(100, result.confidenceRaw)) : 0) + '%',
                  height: '100%',
                  background: 'linear-gradient(90deg,#38F9D7,#0EA5E9)',
                  transition: 'width .5s ease'
                }
              })
            ),
            el('div', { style: { marginTop: 8, fontSize: 13, color: 'var(--muted)', textAlign: 'right' } }, result.confidence)
          )
        )
      )
    ),

    // right: chat
    el('div', { style: { flex: 1, minWidth: 320 } },
      el(ChatWidget, { updateForm: update })
    )
  )
);


}

/* ---------------- Chat Widget (Homepage style) ---------------- */
function ChatWidget({ updateForm }) {
  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState([{ from: 'bot', text: 'Hi — I can help run predictions, explain parameters or show charts. Ask me anything about roof fate rate.' }]);
  const [value, setValue] = useState('');
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);
  const chatLogRef = useRef(null);

  // auto-scroll when messages change
  useEffect(() => {
    if (chatLogRef.current) {
      chatLogRef.current.scrollTop = chatLogRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) return;
    const rec = new SpeechRecognition();
    rec.lang = 'en-US'; rec.interimResults = true; rec.maxAlternatives = 1;
    rec.onresult = (ev) => {
      let interim = '', final = '';
      for (let i = 0; i < ev.results.length; i++) { let res = ev.results[i]; if (res.isFinal) final += res[0].transcript; else interim += res[0].transcript; }
      setValue(final || interim);
    };
    rec.onend = () => setListening(false);
    rec.onerror = (e) => { console.warn('Speech recognition error', e); setListening(false); };
    recognitionRef.current = rec;
    return () => { try { rec.stop(); } catch (e) { } };
  }, []);

  function toggleListening() {
    const rec = recognitionRef.current;
    if (!rec) { alert('Speech recognition not supported'); return; }
    if (!listening) { try { rec.start(); setListening(true); } catch (e) { console.warn(e); } }
    else { rec.stop(); setListening(false); }
  }

  function send(txt) {
    if (!txt) return;
    setMessages(m => m.concat({ from: 'user', text: txt }));
    setValue('');
    const features = extractFeaturesFromText(txt);

    // Map extracted features to form keys and update
    if (updateForm) {
      if (features.CMRR !== null) updateForm('cmrr', features.CMRR);
      if (features.PRSUP !== null) updateForm('prsup', features.PRSUP);
      if (features.depth_of_cover !== null) updateForm('depthOfCover', features.depth_of_cover);
      if (features.intersection_diagonal !== null) updateForm('intersectionDiagonal', features.intersection_diagonal);
      if (features.mining_height !== null) updateForm('miningHeight', features.mining_height);
    }
    setTimeout(() => {
      let botText = 'I received: "' + txt + '"';
      const extractedText = Object.entries(features)
        .filter(([_, v]) => v !== null)
        .map(([k, v]) => `${k}: ${v}`)
        .join(', ');

      if (extractedText) botText += ` — updated parameters: ${extractedText}`;

      setMessages(m => m.concat({ from: 'bot', text: botText }));
    }, 600);
   
  }

  return el('div', { className: 'chat-widget', style: { position: 'fixed', bottom: 20, right: 20, width: 300, zIndex: 999 } },

  // top bar
  el('div', { style: { display: 'flex', gap: 8 } },
    el('button', {
      onClick: toggleListening,
      'aria-label': 'Voice input',
      title: listening ? 'Stop listening' : 'Start voice input',
      style: { background: 'transparent', border: 'none', color: 'var(--accent)', fontSize: 20, cursor: 'pointer', padding: 8, borderRadius: 8 }
    }, listening ? '🎤⏺' : '🎤'),
    el('button', {
      onClick: () => setOpen(!open),
      className: 'chat-toggle',
      style: { flex: 1, padding: 6, cursor: 'pointer' }
    }, open ? 'Close Chat' : 'Chat')
  ),

  // panel
  open && el('div', {
    className: 'chat-panel card',
    style: { marginTop: 10, padding: 10, maxHeight: 360, overflowY: 'auto', background: 'rgba(0,0,0,0.85)', color: 'white' }
  },

    el('div', { className: 'chat-header', style: { fontWeight: 700, marginBottom: 6 } }, 'Assistant'),

    el('div', { className: 'chat-log', role: 'log', 'aria-live': 'polite' },
      // map messages safely
      ...messages.map((m, i) =>
        el('div', {
          key: i,
          className: 'chat-msg ' + (m.from === 'user' ? 'user' : 'bot'),
          style: {
            margin: '6px 0',
            color: m.from === 'bot' ? '#C5F0FF' : '#FFFFFF',
            background: m.from === 'bot' ? 'rgba(0,0,0,0.3)' : 'rgba(255,255,255,0.1)',
            padding: '6px 10px',
            borderRadius: 8,
            fontSize: 14,
            lineHeight: '1.4'
          }
        }, m.text)
      )
    ),

    // input form
    el('form', { onSubmit: e => { e.preventDefault(); send(value); } },
      el('input', {
        placeholder: 'Ask about predictions or parameters...',
        value,
        onChange: e => setValue(e.target.value),
        style: { width: 'calc(100% - 60px)', marginRight: 6, padding: 6, borderRadius: 6, border: '1px solid rgba(255,255,255,0.2)', background: 'rgba(0,0,0,0.3)', color: 'white' }
      }),
      el('button', { type: 'submit', className: 'btn btn-primary', style: { padding: '6px 10px' } }, 'Send')
    )
  )
);

}