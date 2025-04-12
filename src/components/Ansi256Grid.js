// src/components/Ansi256Grid.jsx
import React from 'react';

function ansi256ToRGB(n) {
  if (n < 16) {
    const standard = [
      [0,0,0], [128,0,0], [0,128,0], [128,128,0],
      [0,0,128], [128,0,128], [0,128,128], [192,192,192],
      [128,128,128], [255,0,0], [0,255,0], [255,255,0],
      [0,0,255], [255,0,255], [0,255,255], [255,255,255]
    ];
    return standard[n];
  } else if (n < 232) {
    const c = [0, 95, 135, 175, 215, 255];
    const r = Math.floor((n - 16) / 36);
    const g = Math.floor((n - 16) / 6) % 6;
    const b = (n - 16) % 6;
    return [c[r], c[g], c[b]];
  } else {
    const gray = 8 + (n - 232) * 10;
    return [gray, gray, gray];
  }
}

export default function Ansi256Grid() {
  const items = Array.from({ length: 256 }, (_, i) => {
    const [r, g, b] = ansi256ToRGB(i);
    const bg = `rgb(${r}, ${g}, ${b})`;
    const fg = (r + g + b) / 3 > 128 ? '#000' : '#fff';
    return (
      <div
        key={i}
        style={{
          width: '42px',
          height: '20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: bg,
          color: fg,
          fontFamily: 'monospace',
          fontSize: '12px',
          borderRadius: '2px',
          margin: '2px',
        }}
      >
        {i.toString().padStart(3, ' ')}
      </div>
    );
  });

  return (
    <div
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '2px',
        maxWidth: '100%',
        background: '#111',
        padding: '10px',
        borderRadius: '8px',
      }}
    >
      {items}
    </div>
  );
}
