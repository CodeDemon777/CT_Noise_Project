import React from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Scan, FileText, CheckCircle2 } from 'lucide-react';

export default function HeroSection() {
  return (
    <section className="relative pt-12 pb-8 overflow-hidden">
      {/* Background glowing orbs */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[300px] bg-gradient-to-tr from-sky-500/10 via-indigo-500/10 to-purple-500/10 blur-3xl pointer-events-none rounded-full" />

      <div className="max-w-7xl mx-auto px-4 lg:px-8 text-center relative z-10">
        {/* Top badge */}
        <motion.div 
          initial={{ opacity: 0, y: -15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-slate-900/90 border border-sky-500/30 text-sky-400 text-xs font-medium shadow-lg shadow-sky-500/10 mb-6"
        >
          <Sparkles className="w-3.5 h-3.5 text-sky-400 animate-spin" style={{ animationDuration: '6s' }} />
          <span>Next-Generation Deep Learning Diagnostic Pipeline</span>
        </motion.div>

        {/* Title */}
        <motion.h1 
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-4xl sm:text-5xl lg:text-6xl font-extrabold tracking-tight text-white max-w-4xl mx-auto leading-tight"
        >
          Multi-Model <span className="bg-gradient-to-r from-sky-400 via-teal-300 to-indigo-400 bg-clip-text text-transparent">CT Noise Classification</span> &amp; Severity Estimation
        </motion.h1>

        {/* Description */}
        <motion.p 
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="mt-4 text-base sm:text-lg text-slate-300 max-w-2xl mx-auto font-light leading-relaxed"
        >
          Isolate, classify, and quantify 8 clinical noise distributions across 4 synchronized neural networks with sub-pixel bounding boxes and automated clinical PDF reports.
        </motion.p>

        {/* Live Metrics Pills */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="mt-8 flex flex-wrap justify-center items-center gap-3 sm:gap-6 text-xs sm:text-sm"
        >
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-white/10 text-slate-200">
            <Scan className="w-4 h-4 text-sky-400" />
            <span><strong>4</strong> Neural Architectures</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-white/10 text-slate-200">
            <CheckCircle2 className="w-4 h-4 text-emerald-400" />
            <span><strong>98.86%</strong> Dice Accuracy</span>
          </div>
          <div className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-900/60 border border-white/10 text-slate-200">
            <FileText className="w-4 h-4 text-purple-400" />
            <span><strong>Auto-Generated</strong> Clinical PDFs</span>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
