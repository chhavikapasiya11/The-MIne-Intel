import React, { useState, useEffect, useRef } from "react";

// Import all images from the same folder
import cmrrImg from "./cmrr.jpeg";
import prsupImg from "./prsup.jpeg";
import depthImg from "./mh.jpeg";
import shapSummaryImg from "./shap_summary_plot.png";
import intersectionImg from "./ic.jpeg";
import depthCoverImg from "./doc.jpeg";

export default function ShapGraphDarkUpdated() {
  const [modalOpen, setModalOpen] = useState(false);
  const [activeSrc, setActiveSrc] = useState(null);
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const dragging = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });
  const imgWrapperRef = useRef(null);

  useEffect(() => {
    function onKey(e) {
      if (e.key === "Escape") closeModal();
      if ((e.key === "+" || e.key === "=") && modalOpen) setScale(s => Math.min(4, +(s + 0.2).toFixed(2)));
      if (e.key === "-" && modalOpen) setScale(s => Math.max(0.5, +(s - 0.2).toFixed(2)));
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [modalOpen]);

  useEffect(() => {
    if (!modalOpen) {
      setScale(1);
      setTranslate({ x: 0, y: 0 });
    }
  }, [modalOpen]);

  function openModal(src) {
    setActiveSrc(src);
    setModalOpen(true);
  }
  function closeModal() {
    setModalOpen(false);
    setActiveSrc(null);
  }

  function onWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setScale(s => Math.min(6, Math.max(0.5, +(s + delta).toFixed(2))));
  }

  function onMouseDown(e) {
    if (scale <= 1) return;
    dragging.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
  }
  function onMouseMove(e) {
    if (!dragging.current) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    lastPos.current = { x: e.clientX, y: e.clientY };
    setTranslate(t => ({ x: +(t.x + dx).toFixed(2), y: +(t.y + dy).toFixed(2) }));
  }
  function onMouseUp() {
    dragging.current = false;
  }

  function resetView() {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }

  const container = {
    maxWidth: 1200,
    margin: "28px auto",
    padding: 20,
    color: "#E6EEF3",
    fontFamily: "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial",
    display: "flex",
    flexDirection: "column",
    gap: 28
  };

  const glass = {
    borderRadius: 16,
    padding: 20,
    background: "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.32))",
    border: "1px solid rgba(255,255,255,0.06)",
    boxShadow: "0 12px 36px rgba(2,6,23,0.6)",
    backdropFilter: "blur(6px)"
  };

  const sectionTitle = (accent) => ({
    fontSize: 20,
    fontWeight: 800,
    letterSpacing: 0.6,
    color: accent
  });

  const placeholder = (label) => (
    <div
      style={{
        height: 220,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 10,
        background: "linear-gradient(180deg, rgba(255,255,255,0.02), rgba(0,0,0,0.2))",
        border: "1px dashed rgba(255,255,255,0.08)",
        color: "rgba(200,210,220,0.5)",
        fontSize: 13
      }}
    >
      {label} — Add path
    </div>
  );

  return (
    <div style={container}>
      <style>{`
        .frame {
          width: 100%;
          height: 280px;
          border-radius: 14px;
          overflow: hidden;
          transition: transform .22s ease, box-shadow .22s ease;
          box-shadow: inset 0 -20px 40px rgba(0,0,0,0.45);
          display: flex;
          align-items: center;
          justify-content: center;
        }
        .frame:hover {
          transform: translateY(-5px) scale(1.01);
          box-shadow: 0 18px 40px rgba(12,40,80,0.36);
        }
        .frame img { width: 100%; height: 100%; object-fit: contain; }

        .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }
        .grid-2 { display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 20px; }

        .thumb { height: 180px; border-radius: 12px; overflow: hidden; transition: transform .18s ease, box-shadow .18s ease; display: flex; align-items: center; justify-content: center; background: linear-gradient(180deg, rgba(255,255,255,0.01), rgba(0,0,0,0.22)); cursor: zoom-in; }
        .thumb:hover { transform: translateY(-6px) scale(1.02); box-shadow: 0 14px 34px rgba(0,0,0,0.6); }
        .thumb img { width: 100%; height: 100%; object-fit: contain; object-position: center; }

        .accent-line { height: 6px; border-radius: 6px; background: linear-gradient(90deg, #6EE7F5, #7C3AED); box-shadow: 0 6px 30px rgba(108,99,255,0.12); margin-top: 12px; }

        .insights-list li { margin: 8px 0; color: rgba(230,238,243,0.82); }

        /* Modal styles */
        .modal-overlay { position: fixed; inset: 0; background: rgba(3,6,12,0.78); display: flex; align-items: center; justify-content: center; z-index: 2000; }
        .modal-card { max-width: 92vw; max-height: 92vh; border-radius: 10px; overflow: hidden; position: relative; display: flex; align-items: center; justify-content: center; }
        .modal-img-wrapper { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; cursor: grab; }
        .modal-img-wrapper:active { cursor: grabbing; }
        .modal-img { max-width: 100%; max-height: 100%; user-select: none; pointer-events: none; transform-origin: center center; transition: transform 120ms ease; }

        .modal-controls { position: absolute; top: 12px; right: 12px; display: flex; gap: 8px; }
        .btn { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08); color: #e6eef3; padding: 8px 10px; border-radius: 8px; cursor: pointer; font-weight: 600; }
      `}</style>

      {/* Header */}
      <section style={glass}>
        <div style={{ textAlign: "center" }}>
          <div style={sectionTitle("#7DD3FC")}>GRAPH ANALYSIS (SHAP)</div>
        </div>
      </section>

      {/* SHAP Summary */}
      <section style={{ ...glass, display: "flex", flexDirection: "column", gap: 14 }}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <div style={sectionTitle("#C084FC")}>SHAP SUMMARY PLOT</div>
          <div style={{ color: "rgba(230,238,243,0.6)", fontSize: 12 }}>Global impact ranking</div>
        </div>

        <div className="frame" style={{ height: 320 }} onClick={() => openModal(shapSummaryImg)}>
          {shapSummaryImg ? (
            <img src={shapSummaryImg} alt="SHAP Summary Plot" style={{ width: "100%", height: "100%", objectFit: "contain" }} />
          ) : (
            placeholder("SHAP SUMMARY")
          )}
        </div>

        <div className="accent-line" style={{ width: 160 }} />
      </section>

      {/* 3 Main Dependence Graphs */}
      <section style={glass}>
        <div style={{ display: "flex", justifyContent: "space-between" }}>
          <div style={sectionTitle("#93C5FD")}>MAIN DEPENDENCE PLOTS</div>
          <div style={{ color: "rgba(230,238,243,0.6)", fontSize: 12 }}>(Top 3 contributors)</div>
        </div>

        <div className="grid-3" style={{ marginTop: 16 }}>
          <div>
            <div style={{ textAlign: "center", fontSize: 14, marginBottom: 8 }}>CMRR</div>
            <div className="thumb" onClick={() => openModal(cmrrImg)}><img src={cmrrImg} alt="CMRR SHAP" /></div>
          </div>

          <div>
            <div style={{ textAlign: "center", fontSize: 14, marginBottom: 8 }}>PRSUP</div>
            <div className="thumb" onClick={() => openModal(prsupImg)}><img src={prsupImg} alt="PRSUP SHAP" /></div>
          </div>

          <div>
            <div style={{ textAlign: "center", fontSize: 14, marginBottom: 8 }}>DEPTH OF COVER</div>
            <div className="thumb" onClick={() => openModal(depthImg)}><img src={depthImg} alt="Depth SHAP" /></div>
          </div>
        </div>
      </section>

      {/* Additional Relationship Plots */}
      <section style={glass}>
        <div style={sectionTitle("#A5B4FC")}>ADDITIONAL RELATIONSHIP PLOTS</div>

        <div className="grid-2">
          <div>
            <div style={{ textAlign: "center", fontSize: 14, marginBottom: 8 }}>INTERSECTION DIAGONAL</div>
            <div className="thumb" onClick={() => openModal(intersectionImg)}><img src={intersectionImg} alt="Intersection Plot" /></div>
          </div>

          <div>
            <div style={{ textAlign: "center", fontSize: 14, marginBottom: 8 }}>DEPTH COVER RELATION</div>
            <div className="thumb" onClick={() => openModal(depthCoverImg)}><img src={depthCoverImg} alt="Depth Cover Relation" /></div>
          </div>
        </div>
      </section>

      {/* Insights Summary */}
      <section style={glass}>
        <div style={sectionTitle("#F59E0B")}>INSIGHTS SUMMARY</div>
        <ul className="insights-list" style={{ marginTop: 12 }}>
          <li>CMRR and PRSUP consistently dominate SHAP importance rankings.</li>
          <li>Depth of cover shows smooth but non-linear influence on failure probability.</li>
          <li>Diagonal intersection plot highlights 2-way interaction zones.</li>
          <li>Depth-cover relationship suggests combined mechanical weakening patterns.</li>
        </ul>
      </section>

      {/* Zoom Modal */}
      {modalOpen && (
        <div className="modal-overlay" onMouseUp={onMouseUp} onMouseMove={onMouseMove} onWheel={onWheel} onMouseDown={onMouseDown} onDoubleClick={() => setScale(s => (s === 1 ? 2.2 : 1))}>
          <div className="modal-card" onClick={e => e.stopPropagation()} ref={imgWrapperRef}>
            <div className="modal-img-wrapper" style={{ transform: `translate(${translate.x}px, ${translate.y}px)` }}>
              <img
                src={activeSrc}
                alt="Zoom"
                className="modal-img"
                style={{ transform: `scale(${scale})` }}
                draggable={false}
                onDragStart={e => e.preventDefault()}
              />
            </div>

            <div className="modal-controls">
              <button className="btn" onClick={() => setScale(s => Math.min(6, +(s + 0.2).toFixed(2)))}>+</button>
              <button className="btn" onClick={() => setScale(s => Math.max(0.5, +(s - 0.2).toFixed(2)))}>-</button>
              <button className="btn" onClick={resetView}>Reset</button>
              <button className="btn" onClick={closeModal}>Close</button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}

