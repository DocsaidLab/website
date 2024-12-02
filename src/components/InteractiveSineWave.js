import { Card, Col, Row, Slider, Typography } from 'antd';
import React, { useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const { Title, Text } = Typography;

function InteractiveSineWave() {
  const [amplitude, setAmplitude] = useState(1);
  const [frequency, setFrequency] = useState(1);
  const [phase, setPhase] = useState(0);

  const data = Array.from({ length: 200 }, (_, i) => {
    const x = i * 0.05;
    const y = amplitude * Math.sin(2 * Math.PI * frequency * x + phase);
    return { x, y }; // 保留完整精度
  });

  // Formatter for X and Y axis ticks
  const formatTick = (value) => value.toFixed(2);

  return (
    <div style={{ maxWidth: '800px', margin: 'auto', padding: '20px' }}>
      <Title level={2} style={{ textAlign: 'center', marginBottom: '20px' }}>
        Interactive Sine Wave
      </Title>

      <Card>
        <ResponsiveContainer width="100%" height={400}>
          <LineChart data={data}>
            <XAxis
              dataKey="x"
              tickFormatter={formatTick} // Format X-axis ticks
              label={{ value: 'Time (t)', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              tickFormatter={formatTick} // Format Y-axis ticks
              domain={[
                -Math.max(Math.abs(amplitude), 1) * 1.2,
                Math.max(Math.abs(amplitude), 1) * 1.2,
              ]}
            />
            <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
            <Tooltip />
            <Line
              type="monotone"
              dataKey="y"
              stroke="#ff7300"
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Card>

      <Card style={{ marginTop: '20px' }} title="Adjust Parameters">
        <Row gutter={[16, 16]} align="middle">
          <Col span={6}>
            <Text strong>Amplitude:</Text>
          </Col>
          <Col span={14}>
            <Slider
              min={0}
              max={5}
              step={0.1}
              value={amplitude}
              onChange={setAmplitude}
              tooltip={{ formatter: (value) => value }}
            />
          </Col>
          <Col span={4}>
            <Text>{amplitude.toFixed(2)}</Text>
          </Col>
        </Row>

        <Row gutter={[16, 16]} align="middle" style={{ marginTop: '20px' }}>
          <Col span={6}>
            <Text strong>Frequency:</Text>
          </Col>
          <Col span={14}>
            <Slider
              min={0.1}
              max={5}
              step={0.1}
              value={frequency}
              onChange={setFrequency}
              tooltip={{ formatter: (value) => value }}
            />
          </Col>
          <Col span={4}>
            <Text>{frequency.toFixed(2)}</Text>
          </Col>
        </Row>

        <Row gutter={[16, 16]} align="middle" style={{ marginTop: '20px' }}>
          <Col span={6}>
            <Text strong>Phase:</Text>
          </Col>
          <Col span={14}>
            <Slider
              min={-Math.PI}
              max={Math.PI}
              step={0.1}
              value={phase}
              onChange={setPhase}
              tooltip={{ formatter: (value) => value.toFixed(2) }}
            />
          </Col>
          <Col span={4}>
            <Text>{phase.toFixed(2)}</Text>
          </Col>
        </Row>
      </Card>
    </div>
  );
}

export default InteractiveSineWave;
