// src/components/Resume/PersonalAchievements.js
import React from "react";
import resumeData from "../../data/resumeData";

function PersonalAchievements() {
  return (
    <div className="section-right">
      <h3>Personal Achievements</h3>
      <div className="card-container">
        {resumeData.personalAchievements.map((item, idx) => (
          <div className="card" key={idx}>
            <strong>{item.title}</strong>
            <p>{item.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default PersonalAchievements;
