// src/components/Resume/WorkExperience.js
import React from "react";
import resumeData from "../../data/resumeData";

function WorkExperience() {
  return (
    <div className="section-right">
      <h3>Work Experience</h3>
      <div className="card-container">
        {resumeData.workExperience.map((exp, idx) => (
          <div className="card" key={idx}>
            <h4>
              {exp.role}
              <span className="company">{exp.company}</span>
            </h4>
            <div className="date">{exp.date}</div>
            <ul>
              {exp.highlights.map((item, i) => (
                <li key={i}>{item}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>
    </div>
  );
}

export default WorkExperience;
