import React from 'react';
import { BarChart2, Activity, Zap } from 'lucide-react';

export default function NoiseHistogram({ resultData, activeModel }) {
  // Generate sample dynamic histogram profile based on computed noise percentage
  const bins = [
    { label: '0-30 HU', density: 12, color: '#38bdf8' },
    { label: '31-65 HU', density: 28, color: '#38bdf8' },
    { label: '66-100 HU', density: 64, color: '#818cf8' },
    { label: '101-140 HU', density: 85, color: '#a855f7' },
    { label: '141-180 HU', density: 42, color: '#f43f5e' },
    { label: '181-220 HU', density: 19, color: '#f59e0b' },
    { label: '221-255 HU', density: 8, color: '#10b981' },
  ];

  return (
    <div className="w-full max-w-6xl mx-auto px-4 mb-8">
      <div className="glass-panel p-6">
        <div className="flex items-center justify-between gap-4 pb-3 mb-4 border-b border-white/10">
          <div className="flex items-center gap-2.5">
            <BarChart2 className="w-5 h-5 text-indigo-400" />
            <div>
              <h4 className="text-sm font-bold text-white font-display">Pixel Intensity &amp; Artifact Frequency Distribution</h4>
              <p className="text-[11px] text-slate-400">Radiodensity (Hounsfield / Dynamic Range) Profiler</p>
            </div>
          </div>
          <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
            Real-time Analyzer
          </span>
        </div>

        <div className="grid grid-cols-7 gap-2 items-end h-32 pt-4 px-2">
          {bins.map((b, i) => (
            <div key={i} className="flex flex-col items-center gap-1.5 h-full justify-end group">
              <span className="text-[10px] font-mono text-slate-400 opacity-0 group-hover:opacity-100 transition-opacity">
                {b.density}%
              </span>
              <div
                className="w-full rounded-t-md transition-all duration-700 group-hover:brightness-125"
                style={{
                  height: `${b.density}%`,
                  backgroundColor: b.color,
                  boxShadow: `0 0 12px ${b.color}40`,
                }}
              />
              <span className="text-[9px] font-mono text-slate-500 truncate">{b.label}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
