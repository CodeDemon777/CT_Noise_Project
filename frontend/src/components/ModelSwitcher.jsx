import React from 'react';
import { motion } from 'framer-motion';
import { MODELS_CONFIG } from '../utils/constants';

export default function ModelSwitcher({ activeModel, onSelectModel }) {
  const models = Object.values(MODELS_CONFIG);

  return (
    <div className="w-full max-w-4xl mx-auto px-4 my-6">
      <div className="p-1.5 rounded-2xl bg-slate-900/90 border border-white/10 backdrop-blur-xl grid grid-cols-2 md:grid-cols-4 gap-1.5 shadow-2xl">
        {models.map((m) => {
          const isSelected = activeModel === m.id;
          return (
            <button
              key={m.id}
              onClick={() => onSelectModel(m.id)}
              className={`relative flex flex-col items-center text-center p-3 rounded-xl transition-all select-none ${
                isSelected ? 'text-white' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/40'
              }`}
            >
              {isSelected && (
                <motion.div
                  layoutId="activeTabPill"
                  className="absolute inset-0 rounded-xl bg-gradient-to-r from-sky-500/20 via-indigo-500/20 to-purple-500/20 border border-sky-500/40 shadow-lg shadow-sky-500/10"
                  transition={{ type: 'spring', bounce: 0.2, duration: 0.5 }}
                />
              )}
              <span className="relative z-10 text-[11px] font-mono uppercase tracking-wider font-bold text-sky-400">
                {m.name}
              </span>
              <span className="relative z-10 text-xs font-semibold font-display truncate max-w-full mt-0.5">
                {m.architecture}
              </span>
              <span className="relative z-10 text-[10px] text-slate-400 mt-0.5 truncate max-w-full">
                {m.noises.map((n) => n.label.split(' ')[0]).join(' + ')}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
