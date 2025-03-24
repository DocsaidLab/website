// src/components/Resume/LeftColumn.js
import React from "react";
import resumeData from "../../data/resumeData";
import ContactInfo from "./ContactInfo";
import Skills from "./Skills";

function LeftColumn() {
  return (
    <div className="left-column">
      <div className="name-title">
        <h1>{resumeData.name}</h1>
        <h2>
          <i className="fa-solid fa-robot"></i> {resumeData.title}
        </h2>
      </div>
      <ContactInfo />
      <Skills />
    </div>
  );
}

export default LeftColumn;
