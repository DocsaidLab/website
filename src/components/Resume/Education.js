// src/components/Resume/Education.js
import React from "react";
import resumeData from "../../data/resumeData";

function Education() {
  return (
    <div className="section-right" style={{ padding: "15px 20px" }}>
      <h3>Education</h3>
      <div className="education-block">
        {resumeData.education.map((edu, idx) => (
          <div className="education-item" key={idx}>
            <h4>{edu.degree}</h4>
            <div className="date">{edu.date}</div>
            <p>{edu.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Education;
