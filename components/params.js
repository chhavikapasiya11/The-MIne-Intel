// MiningImpacts.js — Upgraded visual design
import React from "react";

const MINING_IMPACTS = [
  { name: "PRSUP", key: "prsup", impact: 0.91 },
  { name: "Intersection Diagonal", key: "intersectionDiagonal", impact: 0.83},
  { name: "CMRR", key: "cmrr", impact: 0.72},
  { name: "Depth of Cover", key: "depthOfCover", impact: 0.62 },
  { name: "Mining Height", key: "miningHeight", impact: 0.54 }
];


export default function MiningImpacts() {
  return (
    <div
      style={{
      
        padding: "26px 28px",
        borderRadius: 18,
        background: "rgba(12,16,24,0.65)",
        backdropFilter: "blur(14px)",
        border: "1px solid rgba(255,255,255,0.06)",
        boxShadow: "0 18px 42px rgba(0,0,0,0.55)"
      }}
    >
      <h2
        style={{
          margin: 0,
          textAlign: "center",
          color: "#E2E8F0",
          fontWeight: 800,
          fontSize: 24,
          letterSpacing: 0.5
        }}
      >
        Most Impactful Mining Parameters Now
      </h2>

      <div
        style={{
          marginTop: 26,
          display: "flex",
          flexDirection: "column",
          gap: 22
        }}
      >
        {MINING_IMPACTS.map((p) => (
          <div key={p.key} style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {/* Label row */}
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                color: "#A0AEC0",
                fontSize: 15,
                fontWeight: 600
              }}
            >
              <span>{p.name}</span>
              <span style={{ color: "#38F9D7" }}>{p.impact}</span>
            </div>

            {/* Track */}
            <div
              style={{
                height: 12,
                borderRadius: 10,
                background: "linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))",
                overflow: "hidden",
                boxShadow: "inset 0 0 6px rgba(0,0,0,0.6)"
              }}
            >
              {/* Fill */}
              <div
                style={{
                  width: `${p.impact * 100}%`,
                  height: "100%",
                  background: "linear-gradient(90deg, #3CFEC6, #0EA5E9)",
                  transition: "width 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
                  boxShadow: "0 0 12px rgba(56,249,215,0.8)"
                }}
              ></div>
            </div>
          </div>
        ))}
      </div>

      <div
        style={{
          marginTop: 20,
          color: "rgba(255,255,255,0.35)",
          fontSize: 13,
          textAlign: "center",
          fontStyle: "italic"
        }}
      >
      </div>
    </div>
  );
}
