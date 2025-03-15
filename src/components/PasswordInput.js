// /src/components/PasswordInput.js
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import { Input, Progress, Typography } from "antd";
import React, { useState } from "react";
import zxcvbn from "zxcvbn";

const { Text } = Typography;

export default function PasswordInput({ onChange, hideStrength = false,...rest }) {
  const { i18n: { currentLocale } } = useDocusaurusContext();

  const localeText = {
    "zh-hant": {
      passwordStrengthTitle: "密碼強度：",
      strengthTexts: ["非常弱", "弱", "中等", "強", "非常強"],
    },
    en: {
      passwordStrengthTitle: "Password Strength: ",
      strengthTexts: ["Very Weak", "Weak", "Medium", "Strong", "Very Strong"],
    },
    ja: {
      passwordStrengthTitle: "パスワードの強度：",
      strengthTexts: ["非常に弱い", "弱い", "普通", "強い", "非常に強い"],
    },
  };

  const text = localeText[currentLocale] || localeText.en;

  const [score, setScore] = useState(0);
  const progressPercent = score * 25;
  const strokeColor = [
    "#ff4d4f", // 非常弱 / Very Weak
    "#ff7a45", // 弱 / Weak
    "#faad14", // 中等 / Medium
    "#52c41a", // 強 / Strong
    "#1677ff", // 非常強 / Very Strong
  ][score] || "#ff4d4f";

  const handleChange = (e) => {
    const value = e.target.value;
    const result = zxcvbn(value);
    setScore(result.score);
    if (onChange) onChange(e);
  };

  return (
    <div>
      <Input.Password {...rest} onChange={handleChange} />
      {!hideStrength && (
        <div style={{ marginTop: 8 }}>
          <Text>{`${text.passwordStrengthTitle}${text.strengthTexts[score]}`}</Text>
          <Progress percent={progressPercent} showInfo={false} strokeColor={strokeColor} />
        </div>
      )}
    </div>
  );
}
