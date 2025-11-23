// MiningImpacts.js — Upgraded visual design
import React from "react";

const MINING_IMPACTS = [
  {
    name: "Intersection Diagonal",
    key: "intersectionDiagonal",
    impact: 0.27486998,
    display: "27.49%",
  },
  {
    name: "PRSUP",
    key: "prsup",
    impact: 0.26765858,
    display: "26.77%",
  },
  {
    name: "CMRR",
    key: "cmrr",
    impact: 0.19668302,
    display: "19.67%",
  },
  {
    name: "Depth of Cover",
    key: "depthOfCover",
    impact: 0.18008264,
    display: "18.01%",
  },
  {
    name: "Mining Height",
    key: "miningHeight",
    impact: 0.08070578,
    display: "8.07%",
  },
];

export default function MiningImpacts() {
  return (
    <div style={{ padding: "0 32px" }}>
      {/* MAIN IMPACT CARD */}
      <div
        style={{
          padding: "26px 28px",
          borderRadius: 18,
          background: "rgba(12,16,24,0.65)",
          backdropFilter: "blur(14px)",
          border: "1px solid rgba(255,255,255,0.06)",
          boxShadow: "0 18px 42px rgba(0,0,0,0.55)",
        }}
      >
        <h2
          style={{
            margin: 0,
            textAlign: "center",
            color: "#E2E8F0",
            fontWeight: 800,
            fontSize: 24,
            letterSpacing: 0.5,
          }}
        >
          Most Impactful Mining Parameters Now
        </h2>

        <div
          style={{
            marginTop: 26,
            display: "flex",
            flexDirection: "column",
            gap: 22,
          }}
        >
          {MINING_IMPACTS.map((p) => (
            <div
              key={p.key}
              style={{ display: "flex", flexDirection: "column", gap: 8 }}
            >
              {/* Label row */}
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  color: "#A0AEC0",
                  fontSize: 15,
                  fontWeight: 600,
                }}
              >
                <span>{p.name}</span>
                <span style={{ color: "#38F9D7" }}>{p.display}</span>
              </div>

              {/* Track */}
              <div
                style={{
                  height: 12,
                  borderRadius: 10,
                  background:
                    "linear-gradient(90deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02))",
                  overflow: "hidden",
                  boxShadow: "inset 0 0 6px rgba(0,0,0,0.6)",
                }}
              >
                <div
                  style={{
                    width: `${p.impact * 100}%`,
                    height: "100%",
                    background: "linear-gradient(90deg, #3CFEC6, #0EA5E9)",
                    transition: "width 0.6s cubic-bezier(0.4, 0, 0.2, 1)",
                    boxShadow: "0 0 12px rgba(56,249,215,0.8)",
                  }}
                ></div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ================= SUMMARY CARD ================= */}
      <div
        style={{
          marginTop: 32,
          padding: "24px 28px",
          borderRadius: 18,
          background: "rgba(12,16,24,0.65)",
          backdropFilter: "blur(14px)",
          border: "1px solid rgba(255,255,255,0.06)",
          boxShadow: "0 18px 42px rgba(0,0,0,0.55)",
        }}
      >
        <h2
          style={{
            margin: 0,
            textAlign: "center",
            color: "#FDE68A",
            fontWeight: 800,
            fontSize: 22,
            letterSpacing: 0.5,
          }}
        >
          Model Interpretation Summary
        </h2>

        <p
          style={{
            marginTop: 16,
            color: "#CBD5E1",
            fontSize: 15,
            lineHeight: "22px",
          }}
        >
          <li>
            The model highlights <strong>Intersection Diagonal</strong> and{" "}
            <strong>PRSUP</strong>
            as the most influential factors affecting the Roof Fall Rate.
          </li>

          <li style={{ marginTop: 10 }}>
            CMRR and Depth of Cover also contribute meaningfully.
          </li>

          <li style={{ marginTop: 10 }}>
            Mining Height has the lowest global importance, consistent with
            field-level mining stability observations.
          </li>
        </p>
      </div>

      
    </div>
  );
}
