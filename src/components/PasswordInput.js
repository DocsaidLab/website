// /src/components/PasswordInput.js
import { Input, Progress, Typography } from "antd";
import React, { useState } from "react";
import zxcvbn from "zxcvbn";

const { Text } = Typography;

export default function PasswordInput({ onChange, ...rest }) {
  const [score, setScore] = useState(0);
  const strengthTexts = ["非常弱", "弱", "中等", "強", "非常強"];
  const progressPercent = score * 25;
  const strokeColor = [
    "#ff4d4f", // 非常弱
    "#ff7a45", // 弱
    "#faad14", // 中等
    "#52c41a", // 強
    "#1677ff", // 非常強
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
      <div style={{ marginTop: 8 }}>
        <Text>{`密碼強度：${strengthTexts[score]}`}</Text>
        <Progress percent={progressPercent} showInfo={false} strokeColor={strokeColor} />
      </div>
    </div>
  );
}
