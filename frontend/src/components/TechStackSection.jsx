import React from 'react';
import { Code, Server, Brain, Database, FileText, CheckCircle } from 'lucide-react';

export default function TechStackSection() {
  const stackLayers = [
    {
      title: 'Frontend & UI Layer',
      icon: Code,
      color: '#38bdf8',
      items: ['React 18 & Vite', 'Framer Motion (Micro-animations)', 'Chart.js & React-Chartjs-2', 'Glassmorphic Dark UI & Modern CSS']
    },
    {
      title: 'Backend & REST Engine',
      icon: Server,
      color: '#10b981',
      items: ['Python 3.10+ & Flask 3.0', 'Gunicorn WSGI Production Server', 'Multi-Model Dynamic Routing', 'Flask-CORS Security Layer']
    },
    {
      title: 'Deep Learning & CV Core',
      icon: Brain,
      color: '#a855f7',
      items: ['PyTorch 2.2+ Tensor Engine', 'OpenCV 4.8 Image Processing', 'SciPy 2D Fourier Transform (FFT)', 'Segmentation Models PyTorch']
    },
    {
      title: 'Dataset & Volumetric IO',
      icon: Database,
      color: '#f59e0b',
      items: ['HDF5 (h5py) 3D Ingestion', 'LoDoPaB-CT Dataset Pipeline', 'Synthetic CT Phantom Generator', 'NumPy Vectorized Normalization']
    },
    {
      title: 'Clinical PDF Engine',
      icon: FileText,
      color: '#f43f5e',
      items: ['ReportLab 4.0 PDF Builder', 'Diagnostic Severity Badges', 'Side-by-Side CT Exhibits', 'Radiologist Recommendation Engine']
    },
    {
      title: 'QA, Testing & DevOps',
      icon: CheckCircle,
      color: '#38bdf8',
      items: ['PyTest & Python Unittest', 'Git LFS Model Weight Tracking', 'Docker & Procfile Cloud Hosting', 'Ground-Truth Verification Suite']
    }
  ];

  return (
    <section id="tech" className="w-full max-w-6xl mx-auto px-4 mb-16 pt-8">
      <div className="text-center mb-10">
        <span className="text-xs font-mono font-semibold px-3 py-1 rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 uppercase tracking-wider">
          Full-Stack Engineering Stack
        </span>
        <h2 className="text-3xl font-extrabold font-display text-white mt-3">Technology Stack Architecture</h2>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto mt-2 font-light">
          Built on production-grade Python backend microservices, PyTorch tensor acceleration, and modern reactive client engineering.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {stackLayers.map((layer, idx) => {
          const Icon = layer.icon;
          return (
            <div key={idx} className="glass-panel p-6 relative overflow-hidden flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 rounded-xl bg-slate-900 border border-white/10 flex items-center justify-center" style={{ color: layer.color }}>
                    <Icon className="w-5 h-5" />
                  </div>
                  <h3 className="text-base font-bold text-white font-display">{layer.title}</h3>
                </div>

                <ul className="space-y-2 mb-4">
                  {layer.items.map((item, i) => (
                    <li key={i} className="text-xs text-slate-300 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: layer.color }} />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>

              <div className="pt-3 border-t border-white/5 text-[10px] font-mono text-slate-500">
                Layer {idx + 1} of 6 — Production Ready
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
