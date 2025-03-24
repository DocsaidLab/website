// src/components/Resume/ContactInfo.js
import React from "react";
import resumeData from "../../data/resumeData";

function ContactInfo() {
  const { phone, email, location, linkedin, github, website } = resumeData.contact;
  return (
    <div className="section">
      <h3>
        <i className="fa-solid fa-address-book"></i> Contact Information
      </h3>
      <ul className="contact-list">
        <li>
          <i className="fa-solid fa-phone"></i> {phone}
        </li>
        <li>
          <i className="fa-solid fa-envelope"></i> {email}
        </li>
        <li>
          <i className="fa-solid fa-location-dot"></i> {location}
        </li>
        <li>
          <a href={linkedin} target="_blank" rel="noreferrer">
            <i className="fa-brands fa-linkedin"></i> LinkedIn
          </a>
        </li>
        <li>
          <a href={github} target="_blank" rel="noreferrer">
            <i className="fa-brands fa-github"></i> GitHub
          </a>
        </li>
      </ul>
      <a className="website-button" href={website} target="_blank" rel="noreferrer">
        <i className="fa-solid fa-globe"></i> My Website
      </a>
    </div>
  );
}

export default ContactInfo;
