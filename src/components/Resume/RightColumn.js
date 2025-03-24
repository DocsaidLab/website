// src/components/Resume/RightColumn.js
import React from "react";
import AboutMe from "./AboutMe";
import Education from "./Education";
import PersonalAchievements from "./PersonalAchievements";
import WorkExperience from "./WorkExperience";

function RightColumn() {
  return (
    <div className="right-column">
      <AboutMe />
      <WorkExperience />
      <PersonalAchievements />
      <Education />
    </div>
  );
}

export default RightColumn;
