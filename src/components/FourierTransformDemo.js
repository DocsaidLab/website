// FourierTransformDemo.js
import { Card, Col, Row, Select, Slider, Typography } from 'antd';
import { fft, util } from 'fft-js';
import React, { useState } from 'react';
import {
  CartesianGrid,
  Label,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const { Option } = Select;
const { Title } = Typography;

function FourierTransformDemo() {
  const [signalType, setSignalType] = useState('sine');
  const [frequency, setFrequency] = useState(1);
  const [amplitude, setAmplitude] = useState(1);
  const [phase, setPhase] = useState(0);

  const sampleRate = 256;
  const sampleCount = 256;
  const time = Array.from({ length: sampleCount }, (_, i) => i / sampleRate);
  let signal = [];

  // Generate signal
  if (signalType === 'sine') {
    signal = time.map(
      (t) => amplitude * Math.sin(2 * Math.PI * frequency * t + phase)
    );
  } else if (signalType === 'square') {
    signal = time.map(
      (t) =>
        amplitude *
        Math.sign(Math.sin(2 * Math.PI * frequency * t + phase))
    );
  } else if (signalType === 'triangle') {
    signal = time.map((t) => {
      const value =
        (2 * amplitude) /
        Math.PI *
        Math.asin(Math.sin(2 * Math.PI * frequency * t + phase));
      return value;
    });
  } else if (signalType === 'sawtooth') {
    signal = time.map((t) => {
      const value =
        (2 * amplitude) *
        (t * frequency - Math.floor(0.5 + t * frequency));
      return value;
    });
  }

  // Perform Fourier Transform
  const phasors = fft(signal);
  const frequencies = util.fftFreq(phasors, sampleRate);
  const magnitudes = phasors.map((complex) =>
    Math.hypot(complex[0], complex[1])
  );

  // Prepare data for plotting
  const signalData = time.map((t, i) => ({
    t: t.toFixed(5),
    y: signal[i],
  }));

  const maxDisplayFrequency = 20;
  const freqData = frequencies
    .slice(0, sampleCount / 2)
    .map((f, i) => ({
      f: f,
      amp: magnitudes[i],
    }))
    .filter((point) => point.f >= 0 && point.f <= maxDisplayFrequency);

  const formatTick = (value) => Number.parseFloat(value).toFixed(2);

  return (
    <div style={{ padding: '20px' }}>
      <Row gutter={[16, 16]}>
        {/* Control Panel */}
        <Col xs={24} md={8}>
          <Card title="Control Panel" bordered={false}>
            <div style={{ marginBottom: '20px' }}>
              <label>Signal Type:</label>
              <Select
                value={signalType}
                onChange={(value) => setSignalType(value)}
                style={{ width: '100%' }}
              >
                <Option value="sine">Sine Wave</Option>
                <Option value="square">Square Wave</Option>
                <Option value="triangle">Triangle Wave</Option>
                <Option value="sawtooth">Sawtooth Wave</Option>
              </Select>
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label>Frequency (Hz): {frequency}</label>
              <Slider
                min={0.5}
                max={10}
                step={0.1}
                value={frequency}
                onChange={(value) => setFrequency(value)}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label>Amplitude: {amplitude}</label>
              <Slider
                min={0.5}
                max={5}
                step={0.1}
                value={amplitude}
                onChange={(value) => setAmplitude(value)}
              />
            </div>

            <div style={{ marginBottom: '20px' }}>
              <label>Phase (rad): {phase.toFixed(2)}</label>
              <Slider
                min={-Math.PI}
                max={Math.PI}
                step={0.1}
                value={phase}
                onChange={(value) => setPhase(value)}
              />
            </div>
          </Card>
        </Col>

        {/* Chart Section */}
        <Col xs={24} md={16}>
          <Card>
            <Title level={4} style={{ textAlign: 'center' }}>
              Time Domain Signal
            </Title>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={signalData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <XAxis
                  dataKey="t"
                  label={{
                    value: 'Time (s)',
                    position: 'insideBottom',
                    offset: -10,
                  }}
                  tickFormatter={formatTick}
                />
                <YAxis tickFormatter={formatTick}>
                  <Label
                    value="Amplitude"
                    angle={-90}
                    position="insideLeft"
                    offset={-5}
                  />
                </YAxis>
                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                <Tooltip
                  formatter={(value) => value.toFixed(2)}
                  labelFormatter={(label) => `Time: ${label}s`}
                />
                <Line
                  type="monotone"
                  dataKey="y"
                  stroke="#8884d8"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>

          <Card style={{ marginTop: '20px' }}>
            <Title level={4} style={{ textAlign: 'center' }}>
              Frequency Domain Signal
            </Title>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart
                data={freqData}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <XAxis
                  dataKey="f"
                  label={{
                    value: 'Frequency (Hz)',
                    position: 'insideBottom',
                    offset: -10,
                  }}
                  tickFormatter={formatTick}
                />
                <YAxis tickFormatter={formatTick}>
                  <Label
                    value="Magnitude"
                    angle={-90}
                    position="insideLeft"
                    offset={-5}
                  />
                </YAxis>
                <CartesianGrid stroke="#ccc" strokeDasharray="5 5" />
                <Tooltip
                  formatter={(value) => value.toFixed(2)}
                  labelFormatter={(label) => `Frequency: ${label}Hz`}
                />
                <Line
                  type="monotone"
                  dataKey="amp"
                  stroke="#82ca9d"
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
      </Row>
    </div>
  );
}

export default FourierTransformDemo;
