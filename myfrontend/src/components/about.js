// About.js
import React, { useState } from "react";
import Navbar from "./navbar";

export default function About() {
  const [darkMode, setDarkMode] = useState(true);

  const toggleDarkMode = () => setDarkMode(!darkMode);

  const pageStyle = {
    minHeight: "100vh",
    backgroundColor: darkMode ? "#071019" : "#F8FAFC",
    color: darkMode ? "#E6EEF3" : "#1E293B",
    fontFamily: "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial",
    padding: "40px 20px",
    transition: "background-color 0.3s, color 0.3s",
  };

  const sectionStyle = {
    maxWidth: 900,
    margin: "0 auto",
    textAlign: "center",
  };

  return (
    <div style={pageStyle}>
      <Navbar darkMode={darkMode} toggleDarkMode={toggleDarkMode} />

      <div style={sectionStyle}>
        <h1 style={{ fontSize: 36, marginBottom: 20 }}>About Mine-Intel</h1>
        <p style={{ fontSize: 18, lineHeight: 1.6 }}>
          Mine-Intel is your smart mining insights platform, helping you predict and analyze coal mine roof conditions
          efficiently. Our goal is to provide actionable insights for mining engineers and operators, ensuring safety and
          productivity in every project.
        </p>

        <h2 style={{ fontSize: 28, marginTop: 40, marginBottom: 16 }}>Key Features</h2>
        <ul style={{ textAlign: "left", fontSize: 18, maxWidth: 600, margin: "0 auto", lineHeight: 1.6 }}>
          <li>Predict coal mine roof fate rates quickly and accurately.</li>
          <li>Upload and analyze your datasets easily.</li>
          <li>Interactive dashboard for monitoring trends and predictions.</li>
          <li>Supports both light and dark themes for better readability.</li>
        </ul>

        <h2 style={{ fontSize: 28, marginTop: 40, marginBottom: 16 }}>Our Mission</h2>
        <p style={{ fontSize: 18, lineHeight: 1.6 }}>
          To provide intelligent mining solutions that improve safety and operational efficiency, bridging the gap between
          data and actionable insights.
        </p>
      </div>
    </div>
  );
}
