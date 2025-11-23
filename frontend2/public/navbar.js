// Navbar.js
import React, { useState } from "react";
import { FiSun, FiMoon, FiMenu, FiX } from "react-icons/fi"; // For icons

export default function Navbar() {
  const [darkMode, setDarkMode] = useState(true);
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleDarkMode = () => setDarkMode(!darkMode);
  const toggleMenu = () => setMenuOpen(!menuOpen);

  const navStyle = {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    padding: "12px 24px",
    backgroundColor: darkMode ? "#1E293B" : "#F8FAFC",
    color: darkMode ? "#E6EEF3" : "#1E293B",
    fontFamily: "'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, Arial",
    position: "sticky",
    top: 0,
    zIndex: 1000,
    boxShadow: "0 2px 6px rgba(0,0,0,0.3)"
  };

  const linkStyle = {
    textDecoration: "none",
    color: darkMode ? "#E6EEF3" : "#1E293B",
    fontWeight: 500,
    cursor: "pointer"
  };

  const navLinks = (
    <div style={{ display: "flex", gap: 20, alignItems: "center" }}>
      <a href="#home" style={linkStyle}>
        Home
      </a>
      <a href="#about" style={linkStyle}>
        About
      </a>
      <a href="#dashboard" style={linkStyle}>
        Dashboard
      </a>
    
      <button
        onClick={toggleDarkMode}
        style={{
          background: "none",
          border: "none",
          cursor: "pointer",
          fontSize: 20,
          color: darkMode ? "#E6EEF3" : "#1E293B"
        }}
      >
        {darkMode ? <FiSun /> : <FiMoon />}
      </button>
    </div>
  );

  return (
    <nav style={navStyle}>
      {/* Logo / Site Name */}
      <div style={{ fontSize: 20, fontWeight: 700 }}>
        Mine-Intel
        <div style={{ fontSize: 12, fontWeight: 400 }}>Smart Mining Insights</div>
      </div>

      {/* Desktop Menu */}
      <div className="desktop-menu" style={{ display: "none" }}>
        {navLinks}
      </div>

      {/* Mobile Hamburger */}
      <div className="mobile-menu" style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          onClick={toggleMenu}
          style={{
            background: "none",
            border: "none",
            fontSize: 24,
            color: darkMode ? "#E6EEF3" : "#1E293B",
            cursor: "pointer"
          }}
        >
          {menuOpen ? <FiX /> : <FiMenu />}
        </button>
      </div>

      {/* Mobile Dropdown */}
      {menuOpen && (
        <div
          style={{
            position: "absolute",
            top: 64,
            right: 0,
            backgroundColor: darkMode ? "#1E293B" : "#F8FAFC",
            color: darkMode ? "#E6EEF3" : "#1E293B",
            width: "100%",
            padding: 20,
            display: "flex",
            flexDirection: "column",
            gap: 16,
            boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
          }}
        >
          {navLinks}
        </div>
      )}

      {/* Responsive CSS */}
      <style>{`
        @media(min-width: 768px) {
          .desktop-menu {
            display: flex !important;
          }
          .mobile-menu {
            display: none !important;
          }
          .mobile-dropdown {
            display: none !important;
          }
        }
      `}</style>
    </nav>
  );
}
