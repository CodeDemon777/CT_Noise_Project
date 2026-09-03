import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, Activity, CheckCircle, ShieldAlert } from 'lucide-react';
import { MODELS_CONFIG } from '../utils/constants';

export default function SeverityDashboard({ activeModel, resultData }) {
  const modelConfig = MODELS_CONFIG[activeModel] || MODELS_CONFIG.model1;

  // Extract stats per model
  let noise1 = { label: modelConfig.noises[0]?.label, val: 0.0, lvl: 'None', color: modelConfig.noises[0]?.color };
  let noise2 = { label: modelConfig.noises[1]?.label, val: 0.0, lvl: 'None', color: modelConfig.noises[1]?.color };
  let totalNoise = { label: 'Total Noise Area', val: 0.0, lvl: 'None' };

  if (resultData) {
    if (activeModel === 'model1') {
      noise1.val = resultData.gaussian || 0.0;
      noise1.lvl = resultData.gaussian_level || 'None';
      noise2.val = resultData.poisson || 0.0;
      noise2.lvl = resultData.poisson_level || 'None';
      totalNoise.val = resultData.total_noise || 0.0;
      totalNoise.lvl = resultData.total_level || 'None';
    } else if (activeModel === 'model2') {
      const p = resultData.noise?.poisson;
      const s = resultData.noise?.speckle;
      const sum = resultData.summary;
      noise1.val = p?.severity_percentage || 0.0;
      noise1.lvl = p?.severity_level || 'None';
      noise2.val = s?.severity_percentage || 0.0;
      noise2.lvl = s?.severity_level || 'None';
      totalNoise.val = sum?.total_noise_percentage || 0.0;
      totalNoise.lvl = sum?.total_noise_level || 'None';
    } else if (activeModel === 'model3') {
      const sp = resultData.noise?.salt_pepper;
      const rv = resultData.noise?.rvin;
      const sum = resultData.summary;
      noise1.val = sp?.severity_percentage || 0.0;
      noise1.lvl = sp?.severity_level || 'None';
      noise2.val = rv?.severity_percentage || 0.0;
      noise2.lvl = rv?.severity_level || 'None';
      totalNoise.val = sum?.total_noise_percentage || 0.0;
      totalNoise.lvl = sum?.total_noise_level || 'None';
    } else if (activeModel === 'model4') {
      const q = resultData.noise?.quantization;
      const pe = resultData.noise?.periodic;
      const sum = resultData.summary;
      noise1.label = 'Quantization Noise';
      noise1.val = q?.severity_percentage || 0.0;
      noise1.lvl = q?.severity_level || 'None';
      noise2.label = 'Periodic Noise';
      noise2.val = pe?.severity_percentage || 0.0;
      noise2.lvl = pe?.severity_level || 'None';
      totalNoise.label = 'Scan Status';
      totalNoise.val = resultData.confidence || 100.0;
      totalNoise.lvl = resultData.predicted_class?.toUpperCase() || 'CLEAN';
    }
  }

  const getBadgeClass = (lvl) => {
    const l = (lvl || '').toLowerCase();
    if (l === 'none' || l === 'clean') return 'badge-none';
    if (l === 'mild') return 'badge-mild';
    if (l === 'moderate') return 'badge-moderate';
    if (l === 'severe') return 'badge-severe';
    if (l === 'critical') return 'badge-critical';
    return 'badge-none';
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 mb-8">
      <div className="text-center mb-6">
        <h2 className="text-2xl font-bold font-display text-white">Quantitative Noise Severity Assessment</h2>
        <p className="text-xs text-slate-400 mt-1">Multi-distribution pixel area coverage &amp; clinical severity mapping</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Card 1 */}
        <motion.div
          whileHover={{ y: -4 }}
          className="glass-panel p-5 relative overflow-hidden"
        >
          <div className="flex items-center justify-between gap-2 pb-3 mb-4 border-b border-white/10">
            <span className="text-xs font-semibold text-slate-300 truncate">{noise1.label}</span>
            <span className={`badge-pill ${getBadgeClass(noise1.lvl)}`}>{noise1.lvl}</span>
          </div>
          <div className="flex items-baseline gap-2 mb-3">
            <span className="text-3xl font-extrabold font-mono tracking-tight text-white">{noise1.val.toFixed(2)}%</span>
            <span className="text-xs text-slate-400">coverage</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-white/5">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(100, noise1.val)}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className="h-full rounded-full"
              style={{ backgroundColor: noise1.color }}
            />
          </div>
        </motion.div>

        {/* Card 2 */}
        <motion.div
          whileHover={{ y: -4 }}
          className="glass-panel p-5 relative overflow-hidden"
        >
          <div className="flex items-center justify-between gap-2 pb-3 mb-4 border-b border-white/10">
            <span className="text-xs font-semibold text-slate-300 truncate">{noise2.label}</span>
            <span className={`badge-pill ${getBadgeClass(noise2.lvl)}`}>{noise2.lvl}</span>
          </div>
          <div className="flex items-baseline gap-2 mb-3">
            <span className="text-3xl font-extrabold font-mono tracking-tight text-white">{noise2.val.toFixed(2)}%</span>
            <span className="text-xs text-slate-400">coverage</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-white/5">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(100, noise2.val)}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className="h-full rounded-full"
              style={{ backgroundColor: noise2.color }}
            />
          </div>
        </motion.div>

        {/* Card 3: Total Noise / Scan Status */}
        <motion.div
          whileHover={{ y: -4 }}
          className="glass-panel glass-panel-glow p-5 relative overflow-hidden bg-gradient-to-b from-slate-900/90 to-slate-950/90"
        >
          <div className="flex items-center justify-between gap-2 pb-3 mb-4 border-b border-white/10">
            <span className="text-xs font-semibold text-sky-400 truncate">{totalNoise.label}</span>
            <span className={`badge-pill ${getBadgeClass(totalNoise.lvl)}`}>{totalNoise.lvl}</span>
          </div>
          <div className="flex items-baseline gap-2 mb-3">
            <span className="text-3xl font-extrabold font-mono tracking-tight text-white">{totalNoise.val.toFixed(2)}%</span>
            <span className="text-xs text-slate-400">{activeModel === 'model4' ? 'confidence' : 'total impacted'}</span>
          </div>
          <div className="w-full bg-slate-950 h-2.5 rounded-full overflow-hidden border border-white/5">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${Math.min(100, totalNoise.val)}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className="h-full rounded-full bg-gradient-to-r from-sky-400 to-indigo-500 shadow-md shadow-sky-500/20"
            />
          </div>
        </motion.div>
      </div>
    </div>
  );
}
