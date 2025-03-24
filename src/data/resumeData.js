const resumeData = {
  name: "Z. Yuan",
  title: "Senior CV/ML Engineer",
  contact: {
    phone: "09xx-xxx-xxx",
    email: "xxx@gmail.com",
    location: "XXX City, Taiwan",
    linkedin: "https://www.linkedin.com/in/ze-yuan-sh7/",
    github: "https://github.com/zephyr-sh",
    website: "https://docsaid.org"
  },
  skills: [
    { skillName: "Python", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "PyTorch", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "Deep Learning", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "Computer Vision", levelLabel: "Expert", levelWidth: "95%" },
    { skillName: "ONNX Runtime", levelLabel: "Proficient", levelWidth: "85%" }
  ],
  aboutMe: `
    Senior CV/ML Engineer with strong expertise in deep learning, MLOps, and document processing.
  `,
  workExperience: [
    {
      role: "Senior AI Engineer",
      company: "CompanyA, Taipei",
      date: "Aug 2020 - Present",
      highlights: [
        "Developed OCR and facial recognition solutions.",
        "Optimized deployment using Docker and ONNX Runtime."
      ]
    },
    {
      role: "ML Engineer",
      company: "CompanyB, Taipei",
      date: "Feb 2016 - Jun 2020",
      highlights: [
        "Built threat detection models.",
        "Improved data pipeline efficiency."
      ]
    }
  ],
  personalAchievements: [
    {
      title: "Web Design",
      description: "Created a multilingual technical blog."
    },
    {
      title: "Open Source",
      description: "Contributed to deep learning projects."
    }
  ],
  education: [
    {
      degree: "Master's Degree, XXXX",
      date: "XXXX - XXXX",
      desc: "Dept. of XXXX Engineering"
    }
  ]
};

export default resumeData;
