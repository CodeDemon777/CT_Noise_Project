import React from 'react';
import { GitBranch, CheckCircle2, RefreshCw, Layers, ShieldCheck } from 'lucide-react';

export default function AgileRoadmapSection() {
  const pillars = [
    { title: 'Iterative Scrum Cadence', desc: 'Bi-weekly sprint milestones with automated CI validation and clinical deliverable reviews.' },
    { title: 'Test-Driven Validation (TDD)', desc: 'Ground-truth multi-distribution test suites ensuring robust zero-error segmentation.' },
    { title: 'Modular SOLID Architecture', desc: 'Decoupled model wrappers, standalone visualizers, and unified REST API routing.' },
    { title: 'Domain-Driven Healthcare Design', desc: 'Medical SRS alignment, strict pixel-level severity thresholds, and clinical PDF exports.' },
  ];

  const sprints = [
    { num: 'Sprint 1', title: 'Requirements & HDF5 Pipeline', tasks: ['Medical SRS & Noise Physics Modeling', '3D CT Volume Slice Extractor', 'Synthetic CT Phantom Generator'] },
    { num: 'Sprint 2', title: 'Deep Learning Model Architectures', tasks: ['U-Net++, Attention U-Net, DeepLabV3+, NoiseCNN', 'Checkpoints & Multi-Class Loss Functions', '2D FFT Fourier Diagnostic Engine'] },
    { num: 'Sprint 3', title: 'Backend REST API & Routing', tasks: ['Flask Microservices & Dynamic Routing', 'Multi-Model Inference Handlers', 'Severity Calculator Engine'] },
    { num: 'Sprint 4', title: 'Diagnostic UI & Viewport Canvas', tasks: ['Interactive Before/After Split-Slider', 'Bounding Boxes & Opacity Controls', 'Model Switcher with Scan Sync'] },
    { num: 'Sprint 5', title: 'QA & Dataset Verification Harness', tasks: ['Unit & Integration Test Suites', 'Automated HDF5 Slice Verification', 'Dual-Noise Sensitivity Testing'] },
    { num: 'Sprint 6', title: 'PDF Engine, React UI & Hosting', tasks: ['ReportLab Clinical PDF Report Builder', 'React 18 + Vite Upgraded UI/UX', 'Cloud Deployment & Docker Hosting'] },
  ];

  return (
    <section id="agile" className="w-full max-w-6xl mx-auto px-4 mb-16 pt-8">
      <div className="text-center mb-10">
        <span className="text-xs font-mono font-semibold px-3 py-1 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/30 uppercase tracking-wider">
          Software Engineering Methodology
        </span>
        <h2 className="text-3xl font-extrabold font-display text-white mt-3">Agile Scrum &amp; Engineering Lifecycle</h2>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto mt-2 font-light">
          Disciplined software engineering practices, continuous integration, and healthcare domain-driven architecture.
        </p>
      </div>

      {/* 4 Pillars */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-12">
        {pillars.map((p, i) => (
          <div key={i} className="glass-panel p-5">
            <div className="w-8 h-8 rounded-lg bg-purple-500/10 border border-purple-500/20 flex items-center justify-center text-purple-400 mb-3 font-mono font-bold text-xs">
              0{i + 1}
            </div>
            <h4 className="text-sm font-bold text-white mb-1.5 font-display">{p.title}</h4>
            <p className="text-xs text-slate-400 font-light leading-relaxed">{p.desc}</p>
          </div>
        ))}
      </div>

      {/* 6 Sprints Grid */}
      <div className="glass-panel p-6">
        <h3 className="text-lg font-bold font-display text-white mb-6 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-purple-400" />
          <span>6-Sprint Milestone Roadmap</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {sprints.map((s, idx) => (
            <div key={idx} className="p-4 rounded-xl bg-slate-900/70 border border-white/5 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between gap-2 pb-2 mb-3 border-b border-white/5">
                  <span className="text-[11px] font-mono font-bold text-purple-400">{s.num}</span>
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
                    Completed
                  </span>
                </div>
                <h4 className="text-xs font-bold text-white mb-2 font-display">{s.title}</h4>
                <ul className="space-y-1.5 text-[11px] text-slate-300">
                  {s.tasks.map((task, tIdx) => (
                    <li key={tIdx} className="flex items-start gap-1.5">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0 mt-0.5" />
                      <span>{task}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
