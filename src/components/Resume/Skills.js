// src/components/Resume/Skills.js
import React from "react";
import resumeData from "../../data/resumeData";

function Skills() {
  return (
    <div className="section">
      <h3>Skills</h3>
      <ul className="skills-list">
        {resumeData.skills.map((item, idx) => (
          <li key={idx}>
            <div className="skill-item">
              <div className="skill-label">
                <span>{item.skillName}</span>
                <span>{item.levelLabel}</span>
              </div>
              <div className="skill-bar">
                <div className="skill-level" style={{ width: item.levelWidth }}></div>
              </div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default Skills;
