// src/components/Resume/AboutMe.js
import React from "react";
import resumeData from "../../data/resumeData";

function AboutMe() {
  return (
    <div className="section-right">
      <h3>About Me</h3>
      <p className="about-me">{resumeData.aboutMe}</p>
    </div>
  );
}

export default AboutMe;
