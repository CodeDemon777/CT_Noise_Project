import React, { useState, useEffect } from 'react';
import { Activity, ShieldCheck, Cpu, Layers, GitBranch, Settings, Check, X, Server, RefreshCw } from 'lucide-react';
import { resolveApiUrl, getApiBaseUrl, setApiBaseUrl } from '../utils/constants';

export default function Navbar() {
  const [serverState, setServerState] = useState('checking'); // 'active' | 'waking' | 'offline' | 'checking'
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [backendInput, setBackendInput] = useState('');

  const checkHealth = async () => {
    try {
      const res = await fetch(resolveApiUrl('/health'), {
        method: 'GET',
        headers: { 'Accept': 'application/json' },
      });

      if (res.status === 200) {
        setServerState('active');
      } else if (res.status === 502 || res.status === 503 || res.status === 504) {
        setServerState('waking');
      } else {
        setServerState('offline');
      }
    } catch (err) {
      setServerState('waking');
    }
  };

  useEffect(() => {
    setBackendInput(getApiBaseUrl());
    checkHealth();
    const interval = setInterval(checkHealth, 6000);
    return () => clearInterval(interval);
  }, []);

  const handleSaveBackendUrl = (e) => {
    e.preventDefault();
    setApiBaseUrl(backendInput);
    setIsModalOpen(false);
    setServerState('checking');
    setTimeout(checkHealth, 300);
  };

  return (
    <>
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
              <p className="text-[11px] text-slate-400 hidden sm:block">Multi-Model Noise Classification &amp; Severity</p>
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

          {/* System Status Pill & Config Trigger */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => { setBackendInput(getApiBaseUrl()); setIsModalOpen(true); }}
              title="Click to configure Render backend URL"
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-mono font-medium transition-all cursor-pointer ${
                serverState === 'active'
                  ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-400 hover:bg-emerald-500/20' 
                  : serverState === 'waking'
                  ? 'bg-amber-500/10 border-amber-500/30 text-amber-400 hover:bg-amber-500/20'
                  : 'bg-rose-500/10 border-rose-500/30 text-rose-400 hover:bg-rose-500/20'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${
                serverState === 'active' 
                  ? 'bg-emerald-400 animate-ping' 
                  : serverState === 'waking' 
                  ? 'bg-amber-400 animate-pulse' 
                  : 'bg-rose-400'
              }`} />
              <span>
                {serverState === 'active' 
                  ? 'SERVER ACTIVE' 
                  : serverState === 'waking' 
                  ? 'WAKING UP SERVER (Cold Start)...' 
                  : 'SET BACKEND URL'}
              </span>
              <Settings className="w-3.5 h-3.5 ml-0.5 text-slate-400" />
            </button>
          </div>
        </div>
      </header>

      {/* Backend API Configuration Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 bg-slate-950/80 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-panel max-w-md w-full p-6 relative border-sky-500/30 shadow-2xl">
            <button
              onClick={() => setIsModalOpen(false)}
              className="absolute top-4 right-4 text-slate-400 hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400">
                <Server className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white font-display">Render Backend Connection</h3>
                <p className="text-xs text-slate-400">Connect Vercel frontend to your live Render backend</p>
              </div>
            </div>

            <form onSubmit={handleSaveBackendUrl} className="space-y-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5">
                  Backend API URL (Render)
                </label>
                <input
                  type="url"
                  placeholder="https://ct-noise-project-1.onrender.com"
                  value={backendInput}
                  onChange={(e) => setBackendInput(e.target.value)}
                  className="w-full px-3.5 py-2.5 rounded-xl bg-slate-900 border border-white/10 text-white text-xs font-mono placeholder:text-slate-600 focus:outline-none focus:border-sky-400"
                />
                <p className="text-[11px] text-slate-400 mt-1.5 leading-relaxed">
                  💡 <strong>Tip</strong>: Render Free Tier automatically sleeps after 15 minutes of inactivity. When accessed, it takes ~30-40 seconds to spin up, which temporarily returns 502/503 during wake-up.
                </p>
              </div>

              <div className="flex items-center justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setIsModalOpen(false)}
                  className="px-4 py-2 rounded-xl text-xs font-medium text-slate-400 hover:text-white bg-slate-900 border border-white/10"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl text-xs font-semibold text-white bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 shadow-lg shadow-sky-500/20 flex items-center gap-1.5 cursor-pointer"
                >
                  <Check className="w-3.5 h-3.5" /> Save &amp; Connect
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
