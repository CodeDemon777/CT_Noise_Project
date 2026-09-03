import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Cpu, Layers, GitBranch, Download } from 'lucide-react';
import { resolveApiUrl } from '../utils/constants';

export default function Navbar() {
  const [serverOnline, setServerOnline] = useState(false);

  useEffect(() => {
    fetch(resolveApiUrl('/health'))
      .then((res) => res.json())
      .then((data) => {
        if (data.status === 'healthy') setServerOnline(true);
      })
      .catch(() => setServerOnline(false));
  }, []);

  return (
    <header className="sticky top-0 z-50 w-full backdrop-blur-xl bg-slate-950/80 border-b border-white/10 px-4 lg:px-8 py-3 transition-all">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Logo */}
        <a href="#" className="flex items-center gap-3 group">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 p-0.5 shadow-lg shadow-sky-500/20 group-hover:scale-105 transition-transform">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Activity className="w-5 h-5 text-sky-400" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xl font-bold font-display tracking-tight text-white">LungCT</span>
              <span className="text-xs font-mono font-bold px-1.5 py-0.5 rounded bg-sky-500/20 text-sky-400 border border-sky-500/30">AI v2.0</span>
            </div>
            <p className="text-[11px] text-slate-400 hidden sm:block">Multi-Model Noise Classification & Severity</p>
          </div>
        </a>

        {/* Navigation Links */}
        <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-300">
          <a href="#models" className="hover:text-sky-400 transition-colors flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-sky-400" /> 4-Model Suite
          </a>
          <a href="#specs" className="hover:text-sky-400 transition-colors flex items-center gap-1.5">
            <Cpu className="w-4 h-4 text-emerald-400" /> Architecture Specs
          </a>
          <a href="#tech" className="hover:text-sky-400 transition-colors flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-amber-400" /> Tech Stack
          </a>
          <a href="#agile" className="hover:text-sky-400 transition-colors flex items-center gap-1.5">
            <GitBranch className="w-4 h-4 text-purple-400" /> Agile Lifecycle
          </a>
        </nav>

        {/* System Status Pill */}
        <div className="flex items-center gap-3">
          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono font-medium ${
            serverOnline 
              ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400' 
              : 'bg-amber-500/10 border-amber-500/30 text-amber-400'
          }`}>
            <span className={`w-2 h-2 rounded-full ${serverOnline ? 'bg-emerald-400 animate-ping' : 'bg-amber-400'}`} />
            <span>{serverOnline ? 'SERVER ACTIVE' : 'CONNECTING…'}</span>
          </div>
        </div>
      </div>
    </header>
  );
}
