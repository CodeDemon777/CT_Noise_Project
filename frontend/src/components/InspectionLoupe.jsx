import React, { useState, useRef } from 'react';
import { Search, ZoomIn } from 'lucide-react';

export default function InspectionLoupe({ imgSrc, zoomLevel = 2.5, loupeSize = 130 }) {
  const [loupeVisible, setLoupeVisible] = useState(false);
  const [pos, setPos] = useState({ x: 0, y: 0, bgX: 0, bgY: 0 });
  const containerRef = useRef(null);

  const handleMouseMove = (e) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();

    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      setLoupeVisible(false);
      return;
    }

    setLoupeVisible(true);

    const bgX = (x / rect.width) * 100;
    const bgY = (y / rect.height) * 100;

    setPos({ x, y, bgX, bgY });
  };

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setLoupeVisible(false)}
      className="relative w-full h-full flex items-center justify-center cursor-crosshair group select-none overflow-hidden"
    >
      <img src={imgSrc} alt="Diagnostic Scan" className="w-full h-auto max-h-[420px] object-contain" />

      {/* Floating Loupe */}
      {loupeVisible && (
        <div
          className="absolute pointer-events-none rounded-full border-2 border-sky-400 shadow-2xl shadow-sky-500/50 overflow-hidden z-30"
          style={{
            width: `${loupeSize}px`,
            height: `${loupeSize}px`,
            left: `${pos.x - loupeSize / 2}px`,
            top: `${pos.y - loupeSize / 2}px`,
            backgroundImage: `url(${imgSrc})`,
            backgroundRepeat: 'no-repeat',
            backgroundPosition: `${pos.bgX}% ${pos.bgY}%`,
            backgroundSize: `${zoomLevel * 100}%`,
            backgroundColor: '#020617',
          }}
        >
          {/* Reticle Crosshair */}
          <div className="absolute inset-0 flex items-center justify-center opacity-40">
            <div className="w-full h-[1px] bg-sky-400" />
            <div className="h-full w-[1px] bg-sky-400 absolute" />
          </div>
          <div className="absolute bottom-1 right-2 text-[9px] font-mono text-sky-400 font-bold bg-slate-950/80 px-1 rounded">
            {zoomLevel}x ZOOM
          </div>
        </div>
      )}
    </div>
  );
}
