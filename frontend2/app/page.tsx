'use client';

import Link from 'next/link';
import '../globals.css';
import type { CSSProperties } from 'react';

export default function Home() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-black">
      
      {/* Background */}
      <div
        className="absolute inset-0 bg-cover bg-center opacity-20 blur-[2px]"
        
        style={{ backgroundImage: "url('/assets.jpeg')" }}
      />

      {/* Gradients */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-gradient-to-br from-transparent via-black/40 to-black/70"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_20%_20%,rgba(167,139,250,0.10),transparent_30%)]"></div>
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_80%,rgba(91,227,255,0.08),transparent_35%)]"></div>
      </div>

      {/* ---- MAIN CONTAINER ---- */}
      <div className="relative container mx-auto px-4 py-12">

        <h2 className="text-center text-white text-3xl font-bold mb-6">
          Roof Fate Rate — Quick Actions
        </h2>

        {/* GRID */}
        <div
          className="grid"
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
            gap: 22,
            alignItems: "start",
            marginTop: 8
          }}
        >

          {/* -------- Card 1 -------- */}
          <div
            className="home-card"
            style={cardStyle}
            onMouseEnter={(e) => hoverCard(e, true)}
            onMouseLeave={(e) => hoverCard(e, false)}
          >
            <div className="home-accent" style={accentStyle}></div>

            <div style={rowStyle}>
              <div style={iconStyle}>⚡</div>
              <div>
                <h3 style={titleStyle}>Predict Rate</h3>
                <div style={descStyle}>
                  Run model predictions quickly. Upload data on your dedicated predict page.
                </div>
              </div>
            </div>

            <div style={footerRowStyle}>
              <div className="small muted">Quick access</div>

              <Link href="/predict">
                <button className="btn btn-primary" style={buttonStyle}>
                  Open
                </button>
              </Link>
            </div>
          </div>

          {/* -------- Card 2 -------- */}
          <div
            className="home-card"
            style={cardStyle}
            onMouseEnter={(e) => hoverCard(e, true)}
            onMouseLeave={(e) => hoverCard(e, false)}
          >
            <div className="home-accent" style={accentStyle}></div>

            <div style={rowStyle}>
              <div style={iconStyle}>🛠️</div>
              <div>
                <h3 style={titleStyle}>Impactful Parameters</h3>
                <div style={descStyle}>
                  Explore feature importance and recommendations to reduce roof fate rate.
                </div>
              </div>
            </div>

            <div style={footerRowStyle}>
              <div className="small muted">Quick access</div>

              <Link href="/params">
                <button className="btn btn-primary" style={buttonStyle}>
                  Open
                </button>
              </Link>
            </div>
          </div>

          {/* -------- Card 3 -------- */}
          <div
            className="home-card"
            style={cardStyle}
            onMouseEnter={(e) => hoverCard(e, true)}
            onMouseLeave={(e) => hoverCard(e, false)}
          >
            <div className="home-accent" style={accentStyle}></div>

            <div style={rowStyle}>
              <div style={iconStyle}>📈</div>
              <div>
                <h3 style={titleStyle}>Graph Analysis</h3>
                <div style={descStyle}>
                  View interactive charts, residuals and trend analysis.
                </div>
              </div>
            </div>

            <div style={footerRowStyle}>
              <div className="small muted">Quick access</div>

              <Link href="/graphs">
                <button className="btn btn-primary" style={buttonStyle}>
                  Open
                </button>
              </Link>
            </div>
          </div>

        </div>
      </div>
    </div>
  );
}

/* ----------------------------------------------------
   VALID TYPED STYLE OBJECTS (NO TS ERROR ANYMORE)
---------------------------------------------------- */

const cardStyle: CSSProperties = {
  minWidth: 340,
  maxWidth: 720,
  padding: 28,
  borderRadius: 16,
  cursor: "pointer",
  transition:
    "transform .18s cubic-bezier(.2,.9,.2,1), box-shadow .18s ease, background .18s ease",
  display: "flex",
  flexDirection: "column",
  justifyContent: "space-between",
  position: "relative",
  overflow: "hidden",
  background: "linear-gradient(180deg, rgba(255,255,255,0.03), rgba(0,0,0,0.06))",
  border: "1px solid rgba(255,255,255,0.04)",
  boxShadow: "0 10px 30px rgba(2,6,23,0.45)"
};

const accentStyle: CSSProperties = {
  position: "absolute",
  left: 0,
  top: 0,
  bottom: 0,
  width: 10,
  borderRadius: "0 0 8px 0",
  background: "linear-gradient(180deg, var(--accent), var(--accent-2))",
  opacity: 0.16,
  transition: "opacity .18s ease"
};

const rowStyle: CSSProperties = {
  display: "flex",
  gap: 16,
  alignItems: "flex-start"
};

const iconStyle: CSSProperties = {
  width: 72,
  height: 72,
  borderRadius: 14,
  flexShrink: 0,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  fontSize: 32,
  background: "linear-gradient(135deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01))",
  border: "1px solid rgba(255,255,255,0.03)",
  boxShadow: "inset 0 -6px 18px rgba(0,0,0,0.25)"
};

const titleStyle: CSSProperties = {
  margin: "0 0 8px",
  fontSize: 20,
  letterSpacing: 0.2
};

const descStyle: CSSProperties = {
  fontSize: 15,
  lineHeight: 1.35
};

const footerRowStyle: CSSProperties = {
  marginTop: 18,
  display: "flex",
  justifyContent: "space-between",
  alignItems: "center",
  gap: 12
};

const buttonStyle: CSSProperties = {
  padding: "10px 16px",
  borderRadius: 12,
  fontWeight: 700,
  boxShadow: "0 6px 18px rgba(91,227,255,0.06)"
};

/* -------- Hover Animation -------- */
function hoverCard(
  e: React.MouseEvent<HTMLDivElement>,
  hover: boolean
) {
  const card = e.currentTarget as HTMLDivElement;
  const accent = card.querySelector(".home-accent") as HTMLElement;

  if (hover) {
    card.style.transform = "translateY(-10px) scale(1.02)";
    card.style.boxShadow = "0 22px 48px rgba(91,227,255,0.08)";
    if (accent) accent.style.opacity = "1";
  } else {
    card.style.transform = "translateY(0) scale(1)";
    card.style.boxShadow = "0 10px 30px rgba(2,6,23,0.45)";
    if (accent) accent.style.opacity = "0.16";
  }
}
