import { Card, Col, Row, Slider, Typography } from 'antd';
import React, { useMemo, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const { Title, Text } = Typography;

function WaveSuperposition() {
  // Initialize state for mobile mode safely
  const [mobileMode, setMobileMode] = useState(false);

  React.useEffect(() => {
    if (typeof window !== 'undefined') {
      // Set initial state based on window width
      setMobileMode(window.innerWidth <= 768);

      // Update state on window resize
      const handleResize = () => {
        setMobileMode(window.innerWidth <= 768);
      };
      window.addEventListener('resize', handleResize);
      return () => window.removeEventListener('resize', handleResize);
    }
  }, []);

  const [amplitude1, setAmplitude1] = useState(1);
  const [frequency1, setFrequency1] = useState(1);
  const [phase1, setPhase1] = useState(0);

  const [amplitude2, setAmplitude2] = useState(1);
  const [frequency2, setFrequency2] = useState(2);
  const [phase2, setPhase2] = useState(0);

  const data = useMemo(() => {
    const pointCount = mobileMode ? 100 : 200; // Reduce points on mobile
    const timeRange = mobileMode ? 5 : 10; // Limit time range for mobile
    return Array.from({ length: pointCount }, (_, i) => {
      const x = i * (timeRange / pointCount); // Adjust x range based on mode
      const y1 = amplitude1 * Math.sin(2 * Math.PI * frequency1 * x + phase1);
      const y2 = amplitude2 * Math.sin(2 * Math.PI * frequency2 * x + phase2);
      const y = y1 + y2;

      return {
        x: parseFloat(x.toFixed(2)),
        y1: parseFloat(y1.toFixed(2)),
        y2: parseFloat(y2.toFixed(2)),
        y: parseFloat(y.toFixed(2)),
      };
    });
  }, [amplitude1, frequency1, phase1, amplitude2, frequency2, phase2, mobileMode]);

  const yValues = data.flatMap((d) => [d.y1, d.y2, d.y]);
  const yMin = parseFloat((Math.min(...yValues) * 1.2).toFixed(2));
  const yMax = parseFloat((Math.max(...yValues) * 1.2).toFixed(2));

  const renderWaveControls = (waveNumber, amplitude, setAmplitude, frequency, setFrequency, phase, setPhase) => (
    <Card
      title={`Wave ${waveNumber} Parameters`}
      bordered={false}
      style={{ marginBottom: '20px' }}
      bodyStyle={{ padding: '10px' }}
    >
      <Row gutter={[8, 8]} align="middle">
        <Col xs={8} sm={8}>
          <Text strong>Amp:</Text>
        </Col>
        <Col xs={14} sm={12}>
          <Slider
            min={0}
            max={5}
            step={0.1}
            value={amplitude}
            onChange={setAmplitude}
            tooltip={{ formatter: (value) => value }}
          />
        </Col>
        <Col xs={2} sm={4}>
          <Text>{amplitude}</Text>
        </Col>
      </Row>
      <Row gutter={[8, 8]} align="middle">
        <Col xs={8} sm={8}>
          <Text strong>Freq:</Text>
        </Col>
        <Col xs={14} sm={12}>
          <Slider
            min={0.1}
            max={5}
            step={0.1}
            value={frequency}
            onChange={setFrequency}
            tooltip={{ formatter: (value) => value }}
          />
        </Col>
        <Col xs={2} sm={4}>
          <Text>{frequency}</Text>
        </Col>
      </Row>
      <Row gutter={[8, 8]} align="middle">
        <Col xs={8} sm={8}>
          <Text strong>Phase:</Text>
        </Col>
        <Col xs={14} sm={12}>
          <Slider
            min={-Math.PI}
            max={Math.PI}
            step={0.1}
            value={phase}
            onChange={setPhase}
            tooltip={{ formatter: (value) => value.toFixed(2) }}
          />
        </Col>
        <Col xs={2} sm={4}>
          <Text>{phase.toFixed(2)}</Text>
        </Col>
      </Row>
    </Card>
  );

  return (
    <div style={{ maxWidth: '1200px', margin: 'auto', padding: '20px' }}>
      <Title level={2} style={{ textAlign: 'center', marginBottom: '20px' }}>
        Wave Superposition
      </Title>
      <Row gutter={[16, 16]}>
        {/* Control Panel on the Left */}
        <Col xs={24} md={8} lg={6}>
          {renderWaveControls(1, amplitude1, setAmplitude1, frequency1, setFrequency1, phase1, setPhase1)}
          {renderWaveControls(2, amplitude2, setAmplitude2, frequency2, setFrequency2, phase2, setPhase2)}
        </Col>

        {/* Chart Section on the Right */}
        <Col xs={24} md={16} lg={18}>
          <Card>
            <Title level={4} style={{ textAlign: 'center' }}>
              Superposition Chart
            </Title>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart
                data={data}
                margin={{ top: 20, right: 30, left: 0, bottom: 20 }}
              >
                <XAxis
                  dataKey="x"
                  type="number"
                  domain={[0, mobileMode ? 5 : 10]}
                  label={{ value: 'Time (t)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis domain={[yMin, yMax]} />
                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                <Tooltip />
                <Legend verticalAlign="top" height={36} />
                <Line
                  type="linear"
                  dataKey="y"
                  stroke="#ff7300"
                  dot={false}
                  isAnimationActive={false}
                  name="Superposed Wave"
                  strokeWidth={2}
                />
                <Line
                  type="linear"
                  dataKey="y1"
                  stroke="#8884d8"
                  dot={false}
                  isAnimationActive={false}
                  name="Wave 1"
                  strokeDasharray="5 5"
                />
                <Line
                  type="linear"
                  dataKey="y2"
                  stroke="#82ca9d"
                  dot={false}
                  isAnimationActive={false}
                  name="Wave 2"
                  strokeDasharray="3 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default WaveSuperposition;
