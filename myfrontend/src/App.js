// App.js
import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Navbar from "./components/navbar"; // <- Add your Navbar
import Home from "./components/main";
import Predict from "./components/predict";
import Params from "./components/params";
import Graphs from "./components/graph";

/*import "./App.css";*/
import "./components/main.css";

function App() {
  return (
    <Router>
      <div className="App">
        {/* Navbar always visible at top */}
        <Navbar />

        {/* Page content */}
        <div style={{ paddingTop: 80 }}>
          {/* Padding to prevent overlap with sticky navbar */}
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/predict" element={<Predict />} />
            <Route path="/params" element={<Params />} />
            <Route path="/graphs" element={<Graphs />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
