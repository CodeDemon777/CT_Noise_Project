import React, { useRef, useState } from 'react';
import { UploadCloud, Play, Zap, FileImage, AlertCircle, RefreshCw } from 'lucide-react';
import { MODELS_CONFIG } from '../utils/constants';

export default function UploadZone({
  activeModel,
  currentFile,
  previewUrl,
  isLoading,
  onFileUpload,
  onRunAnalysis,
  onRunDemo
}) {
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const modelConfig = MODELS_CONFIG[activeModel] || MODELS_CONFIG.model1;

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      onFileUpload(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      onFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 mb-8">
      <div className="glass-panel p-6 relative overflow-hidden">
        {/* Model info banner */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-white/10">
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono font-bold px-2.5 py-1 rounded-md bg-sky-500/20 text-sky-400 border border-sky-500/30">
              {modelConfig.name}
            </span>
            <div>
              <h3 className="text-base font-bold text-white font-display">{modelConfig.architecture}</h3>
              <p className="text-xs text-slate-400">{modelConfig.subTitle}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {modelConfig.noises.map((n) => (
              <span
                key={n.key}
                className="text-[11px] font-medium px-2.5 py-1 rounded-full bg-slate-800/80 border border-white/10 text-slate-300 flex items-center gap-1.5"
              >
                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: n.color }} />
                {n.label}
              </span>
            ))}
          </div>
        </div>

        {/* Drop area */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all ${
            isDragOver 
              ? 'border-sky-400 bg-sky-500/10 shadow-lg shadow-sky-500/10' 
              : 'border-slate-700/80 hover:border-slate-500 bg-slate-900/40 hover:bg-slate-900/60'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleChange}
          />

          <div className="flex flex-col items-center justify-center gap-3">
            <div className="w-12 h-12 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center text-sky-400">
              <UploadCloud className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-semibold text-white">
                Drag and drop your CT scan here or <span className="text-sky-400 underline">browse</span>
              </p>
              <p className="text-xs text-slate-400 mt-1">Supports PNG, JPEG, BMP — Normalized to 512×512 Grayscale</p>
            </div>
          </div>
        </div>

        {/* Preview Thumbnail if selected */}
        {previewUrl && (
          <div className="mt-4 p-3 rounded-xl bg-slate-900/70 border border-white/10 flex items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <img src={previewUrl} alt="Preview" className="w-12 h-12 rounded-lg object-cover border border-white/10" />
              <div>
                <p className="text-xs font-semibold text-white truncate max-w-xs">{currentFile?.name || 'Selected CT Scan'}</p>
                <p className="text-[11px] text-slate-400">Ready for {modelConfig.architecture} inference</p>
              </div>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
              className="text-xs text-slate-400 hover:text-white flex items-center gap-1 px-2.5 py-1 rounded bg-slate-800 border border-white/10"
            >
              <RefreshCw className="w-3 h-3" /> Change
            </button>
          </div>
        )}

        {/* Action buttons */}
        <div className="mt-5 flex flex-wrap items-center justify-between gap-4">
          <button
            onClick={onRunDemo}
            disabled={isLoading}
            className="text-xs font-medium text-sky-400 hover:text-sky-300 flex items-center gap-1.5 px-3 py-2 rounded-lg bg-sky-500/10 border border-sky-500/20 hover:bg-sky-500/20 transition-all disabled:opacity-50"
          >
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <span>Quick Demo (Synthesize CT &amp; Run)</span>
          </button>

          <button
            onClick={onRunAnalysis}
            disabled={!previewUrl || isLoading}
            className="flex items-center gap-2 px-6 py-2.5 rounded-xl font-semibold text-sm text-white bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 shadow-lg shadow-sky-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin" />
                <span>Running Inference…</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>Execute {modelConfig.name} Analysis</span>
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
