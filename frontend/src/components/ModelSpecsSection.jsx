import React from 'react';
import { Cpu, CheckCircle2, Shield, Layers } from 'lucide-react';
import { MODELS_CONFIG } from '../utils/constants';

export default function ModelSpecsSection() {
  const models = Object.values(MODELS_CONFIG);

  return (
    <section id="specs" className="w-full max-w-6xl mx-auto px-4 mb-16 pt-8">
      <div className="text-center mb-10">
        <span className="text-xs font-mono font-semibold px-3 py-1 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/30 uppercase tracking-wider">
          Deep Architecture Reference
        </span>
        <h2 className="text-3xl font-extrabold font-display text-white mt-3">4-Model Architectural Specifications</h2>
        <p className="text-sm text-slate-400 max-w-2xl mx-auto mt-2 font-light">
          Deep learning models engineered and fine-tuned for specialized CT artifact localization and clinical noise decomposition.
        </p>
      </div>

      {/* 4 Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
        {models.map((m) => (
          <div key={m.id} className="glass-panel p-6 relative overflow-hidden flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between gap-3 pb-3 mb-4 border-b border-white/10">
                <div>
                  <span className="text-xs font-mono font-bold text-sky-400">{m.name}</span>
                  <h3 className="text-lg font-bold text-white font-display">{m.architecture}</h3>
                </div>
                <span className="text-xs font-mono px-2.5 py-1 rounded bg-slate-900 border border-white/10 text-slate-300">
                  {m.resolution}
                </span>
              </div>

              <p className="text-xs text-slate-300 mb-5 font-light leading-relaxed">{m.description}</p>

              <div className="space-y-2 mb-4">
                <div className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Targeted Noise Artifacts:</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {m.noises.map((n) => (
                    <div key={n.key} className="p-2.5 rounded-xl bg-slate-950/60 border border-white/5 flex items-start gap-2">
                      <span className="w-2.5 h-2.5 rounded-full mt-1 shrink-0" style={{ backgroundColor: n.color }} />
                      <div>
                        <p className="text-xs font-semibold text-slate-200">{n.label}</p>
                        <p className="text-[10px] text-slate-400">{n.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-white/10 flex items-center justify-between text-xs text-slate-400 font-mono">
              <span>Weights: {m.weights}</span>
              <span className="text-emerald-400 flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Trained &amp; Verified
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Comparison Matrix Table */}
      <div className="glass-panel p-6 overflow-hidden">
        <h3 className="text-lg font-bold font-display text-white mb-4 flex items-center gap-2">
          <Layers className="w-5 h-5 text-sky-400" />
          <span>Model Comparison &amp; Specification Matrix</span>
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-white/10 bg-slate-900/80 text-slate-300">
                <th className="p-3 font-semibold">Parameter</th>
                <th className="p-3 font-semibold text-sky-400">Model 1 (U-Net++)</th>
                <th className="p-3 font-semibold text-emerald-400">Model 2 (Attention UNet)</th>
                <th className="p-3 font-semibold text-amber-400">Model 3 (DeepLabV3+)</th>
                <th className="p-3 font-semibold text-purple-400">Model 4 (NoiseCNN)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5 text-slate-300">
              <tr>
                <td className="p-3 font-semibold text-slate-400">Task Objective</td>
                <td className="p-3">Thermal &amp; Shot Noise</td>
                <td className="p-3">Acoustic Speckle Gating</td>
                <td className="p-3">Impulse Context Isolation</td>
                <td className="p-3">Harmonic &amp; ADC Bit Steps</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold text-slate-400">Input Size</td>
                <td className="p-3 font-mono">512 × 512 × 1</td>
                <td className="p-3 font-mono">512 × 512 × 1</td>
                <td className="p-3 font-mono">512 × 512 × 1</td>
                <td className="p-3 font-mono">128 × 128 (FFT 512)</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold text-slate-400">Core Mechanism</td>
                <td className="p-3">Nested Dense Skips</td>
                <td className="p-3">Additive Attention Gates</td>
                <td className="p-3">Dilated ASPP Convolutions</td>
                <td className="p-3">2D FFT Magnitude Spectrum</td>
              </tr>
              <tr>
                <td className="p-3 font-semibold text-slate-400">Loss / Metric</td>
                <td className="p-3">Dice 0.9886</td>
                <td className="p-3">Attention Co-occurrence</td>
                <td className="p-3">Multi-class Pixel CE</td>
                <td className="p-3">Softmax Confidence &gt; 99%</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}
