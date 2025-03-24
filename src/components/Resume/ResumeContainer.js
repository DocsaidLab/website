// src/components/Resume/ResumeContainer.js
import React from "react";
import LeftColumn from "./LeftColumn";
import RightColumn from "./RightColumn";
import "./resume.css";

function ResumeContainer() {
  return (
    <div className="resume-container">
      <LeftColumn />
      <RightColumn />
    </div>
  );
}

export default ResumeContainer;
