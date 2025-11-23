import React from "react";

export default function About() {
  const container = {
    position: 'relative',
    top: 0,
    left: 0,
    width: '100%',
    fontFamily: 'Inter, system-ui, sans-serif',
    padding: '48px 22px',
    background: '#0c111b',
    color: '#e2e8f0',
    minHeight: '100vh',
    display: 'flex',
    justifyContent: 'center'
  };

  const wrapper = {
    maxWidth: 980,
    width: '100%',
    background: 'linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02))',
    borderRadius: 18,
    padding: 32,
    border: '1px solid rgba(255,255,255,0.05)',
    backdropFilter: 'blur(8px)',
    boxShadow: '0 10px 35px rgba(0,0,0,0.45)'
  };

  const title = {
    fontSize: 36,
    fontWeight: 800,
    color: '#f8fafc',
    margin: 0,
    textAlign: 'center'
  };

  const subtitle = {
    textAlign: 'center',
    marginTop: 10,
    fontSize: 16,
    color: 'rgba(226,232,240,0.75)'
  };

  const section = {
    marginTop: 32
  };

  const h2 = {
    margin: '0 0 10px 0',
    fontSize: 22,
    color: '#f1f5f9'
  };

  const p = {
    margin: '6px 0',
    lineHeight: 1.6,
    fontSize: 15,
    color: 'rgba(226,232,240,0.88)'
  };

  const list = {
    paddingLeft: 20,
    marginTop: 10,
    color: 'rgba(226,232,240,0.85)'
  };

  return (
    <div style={container}>
      <div style={wrapper}>

        <h1 style={title}>About Mine‑Intel</h1>
        <p style={subtitle}>A modern intelligence platform for industrial prediction, analysis, and decision‑making.</p>

        <section style={section}>
          <h2 style={h2}>What is Mine‑Intel?</h2>
          <p style={p}>Mine‑Intel is a data‑driven platform designed to help industries understand operational behavior, predict risk, and optimize performance. Our tools transform raw inputs into meaningful intelligence that supports engineers, analysts, and field operators.</p>
          <p style={p}>We focus on clarity and speed — giving teams the insights they need without overwhelming dashboards or complex workflows.</p>
        </section>

        <section style={section}>
          <h2 style={h2}>What We Provide</h2>
          <ul style={list}>
            <li>Smart prediction tools for real‑world industrial scenarios.</li>
            <li>Parameter and feature analysis to reveal what truly impacts performance.</li>
            <li>Clean, interpretable graph exploration with trends and residual studies.</li>
            <li>Lightweight workflows that fit naturally into technical operations.</li>
          </ul>
        </section>

        <section style={section}>
          <h2 style={h2}>Our Mission</h2>
          <p style={p}>Our mission is simple: make advanced analytics accessible to every team — from field operators to leadership. We believe intelligence tools should empower people, not slow them down.</p>
        </section>

        <section style={section}>
          <h2 style={h2}>Why It Matters</h2>
          <p style={p}>Industrial environments generate massive amounts of technical data, yet most of it remains under‑used. Mine‑Intel bridges that gap by turning information into actionable understanding. Whether predicting failure, optimizing conditions, or explaining operational patterns — we make complex systems easier to manage.</p>
        </section>

        <footer style={{marginTop: 32, textAlign: 'center', fontSize: 13, color: 'rgba(226,232,240,0.55)'}}>
          © {new Date().getFullYear()} Mine‑Intel • Data‑driven, reliable, human‑focused.
        </footer>

      </div>
    </div>
  );
}
