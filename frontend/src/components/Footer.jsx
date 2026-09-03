import React from 'react';
import { Activity, Heart } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="w-full border-t border-white/10 bg-slate-950/80 backdrop-blur-xl py-12 px-4 lg:px-8 mt-16">
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-6 text-center md:text-left">
        <div>
          <div className="flex items-center justify-center md:justify-start gap-2 mb-2">
            <Activity className="w-5 h-5 text-sky-400" />
            <span className="text-lg font-bold font-display text-white">LungCT AI</span>
            <span className="text-xs font-mono font-semibold px-2 py-0.5 rounded bg-sky-500/20 text-sky-400">v2.0</span>
          </div>
          <p className="text-xs text-slate-400 max-w-md">
            Multi-Model CT Noise Classification &amp; Severity Analysis System. Capstone Software Engineering Project.
          </p>
        </div>

        <div className="text-xs text-slate-400 max-w-sm">
          <p className="text-slate-300 font-medium mb-1">Academic &amp; Research Use Only</p>
          <p className="text-[11px] leading-relaxed">
            Model predictions are designed to assist medical imaging quality assurance and should be verified with clinical CT hardware.
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto mt-8 pt-6 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between gap-4 text-[11px] text-slate-500">
        <p>&copy; 2026 LungCT AI. All rights reserved.</p>
        <p className="flex items-center gap-1">
          Engineered with PyTorch, Flask, and React
        </p>
      </div>
    </footer>
  );
}
