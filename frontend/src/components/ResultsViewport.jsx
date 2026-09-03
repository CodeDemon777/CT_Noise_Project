import React, { useState } from 'react';
import { Eye, Layers, Sliders, Maximize2, SplitSquareVertical, Search } from 'lucide-react';
import { MODELS_CONFIG } from '../utils/constants';
import InspectionLoupe from './InspectionLoupe';

export default function ResultsViewport({ activeModel, resultData, isLoading, rawImageUrl }) {
  const [viewMode, setViewMode] = useState('annotated'); // 'annotated' | 'overlay' | 'split' | 'spectrum' | 'loupe'
  const [sliderPos, setSliderPos] = useState(50);
  const [overlayOpacity, setOverlayOpacity] = useState(0.85);

  const modelConfig = MODELS_CONFIG[activeModel] || MODELS_CONFIG.model1;

  // Resolve Image URLs
  let originalImg = rawImageUrl || resultData?.original_url;
  let annotatedImg = resultData?.annotated_url || resultData?.images?.annotated;
  let overlayImg = resultData?.overlay_url || resultData?.images?.overlay;
  let maskImg = resultData?.images?.mask;
  let spectrumImg = resultData?.images?.spectrum;

  const handleSliderMove = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = Math.max(0, Math.min(e.clientX - rect.left, rect.width));
    setSliderPos((x / rect.width) * 100);
  };

  return (
    <div className="w-full max-w-6xl mx-auto px-4 mb-8">
      <div className="glass-panel p-6">
        {/* Viewport Header */}
        <div className="flex flex-wrap items-center justify-between gap-4 pb-4 mb-5 border-b border-white/10">
          <div>
            <h3 className="text-lg font-bold text-white font-display flex items-center gap-2">
              <Eye className="w-5 h-5 text-sky-400" />
              <span>Diagnostic CT Visual Viewport</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">High-resolution segmentation mask &amp; artifact localization</p>
          </div>

          {/* Mode switch buttons */}
          <div className="flex items-center gap-2 flex-wrap">
            <button
              onClick={() => setViewMode('annotated')}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all ${
                viewMode === 'annotated'
                  ? 'bg-sky-500/20 text-sky-400 border-sky-500/40 shadow-md shadow-sky-500/10'
                  : 'bg-slate-900/60 text-slate-400 border-white/10 hover:text-white'
              }`}
            >
              Annotated HUD
            </button>
            <button
              onClick={() => setViewMode('overlay')}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all ${
                viewMode === 'overlay'
                  ? 'bg-sky-500/20 text-sky-400 border-sky-500/40 shadow-md shadow-sky-500/10'
                  : 'bg-slate-900/60 text-slate-400 border-white/10 hover:text-white'
              }`}
            >
              Color Overlay
            </button>
            <button
              onClick={() => setViewMode('split')}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 ${
                viewMode === 'split'
                  ? 'bg-sky-500/20 text-sky-400 border-sky-500/40 shadow-md shadow-sky-500/10'
                  : 'bg-slate-900/60 text-slate-400 border-white/10 hover:text-white'
              }`}
            >
              <SplitSquareVertical className="w-3.5 h-3.5" />
              <span>Split Slider</span>
            </button>
            <button
              onClick={() => setViewMode('loupe')}
              className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all flex items-center gap-1.5 ${
                viewMode === 'loupe'
                  ? 'bg-sky-500/20 text-sky-400 border-sky-500/40 shadow-md shadow-sky-500/10'
                  : 'bg-slate-900/60 text-slate-400 border-white/10 hover:text-white'
              }`}
            >
              <Search className="w-3.5 h-3.5" />
              <span>2.5x Loupe</span>
            </button>

            {spectrumImg && (
              <button
                onClick={() => setViewMode('spectrum')}
                className={`text-xs font-semibold px-3 py-1.5 rounded-lg border transition-all ${
                  viewMode === 'spectrum'
                    ? 'bg-purple-500/20 text-purple-400 border-purple-500/40'
                    : 'bg-slate-900/60 text-slate-400 border-white/10 hover:text-white'
                }`}
              >
                2D FFT Spectrum
              </button>
            )}
          </div>
        </div>

        {/* Viewport Canvas Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 items-center">
          {/* 1. Raw / Original CT */}
          <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-white/10 shadow-inner flex flex-col items-center justify-center min-h-[340px]">
            <div className="absolute top-3 left-3 z-20 px-2.5 py-1 rounded bg-slate-900/80 backdrop-blur-md border border-white/10 text-[11px] font-mono font-medium text-slate-300">
              INPUT RAW SCAN
            </div>
            {originalImg ? (
              <img src={originalImg} alt="Original CT" className="w-full h-auto max-h-[420px] object-contain" />
            ) : (
              <div className="text-center p-8 text-slate-500">
                <Layers className="w-10 h-10 mx-auto mb-2 opacity-40" />
                <p className="text-xs">No scan uploaded yet</p>
              </div>
            )}
          </div>

          {/* 2. Processed Result / Split Slider / Loupe View */}
          <div className="relative rounded-2xl overflow-hidden bg-slate-950 border border-white/10 shadow-inner flex flex-col items-center justify-center min-h-[340px]">
            <div className="absolute top-3 left-3 z-20 px-2.5 py-1 rounded bg-slate-900/80 backdrop-blur-md border border-white/10 text-[11px] font-mono font-medium text-sky-400">
              {viewMode === 'split' ? 'INTERACTIVE SPLIT COMPARISON' : viewMode === 'loupe' ? '2.5X RETICLE INSPECTION' : viewMode === 'spectrum' ? '2D FOURIER SPECTRUM' : 'AI DIAGNOSTIC VIEW'}
            </div>

            {/* Laser scanning ray animation if loading */}
            {isLoading && (
              <>
                <div className="laser-scanner-line" />
                <div className="laser-scanner-grid" />
              </>
            )}

            {/* View rendering */}
            {viewMode === 'split' && originalImg && (annotatedImg || overlayImg) ? (
              <div
                className="comparison-container w-full h-[380px] cursor-ew-resize relative flex items-center justify-center"
                onMouseMove={handleSliderMove}
                onTouchMove={(e) => {
                  if (e.touches && e.touches[0]) {
                    const rect = e.currentTarget.getBoundingClientRect();
                    const x = Math.max(0, Math.min(e.touches[0].clientX - rect.left, rect.width));
                    setSliderPos((x / rect.width) * 100);
                  }
                }}
              >
                {/* Background: Annotated */}
                <img
                  src={annotatedImg || overlayImg}
                  alt="Annotated"
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none"
                />
                {/* Foreground: Original (Clipped) */}
                <div
                  className="absolute inset-0 overflow-hidden pointer-events-none"
                  style={{ width: `${sliderPos}%` }}
                >
                  <img
                    src={originalImg}
                    alt="Original"
                    className="absolute inset-0 w-full h-full object-contain max-w-none"
                    style={{ width: '100%', height: '100%' }}
                  />
                </div>
                {/* Divider Handle */}
                <div className="comparison-slider-handle" style={{ left: `${sliderPos}%` }}>
                  <div className="comparison-handle-knob">
                    <SplitSquareVertical className="w-4 h-4" />
                  </div>
                </div>
              </div>
            ) : viewMode === 'loupe' && (annotatedImg || overlayImg || originalImg) ? (
              <InspectionLoupe imgSrc={annotatedImg || overlayImg || originalImg} zoomLevel={2.5} />
            ) : viewMode === 'spectrum' && spectrumImg ? (
              <img src={spectrumImg} alt="FFT Spectrum" className="w-full h-auto max-h-[420px] object-contain" />
            ) : viewMode === 'overlay' && overlayImg ? (
              <img
                src={overlayImg}
                alt="Overlay CT"
                style={{ opacity: overlayOpacity }}
                className="w-full h-auto max-h-[420px] object-contain transition-opacity"
              />
            ) : annotatedImg ? (
              <img src={annotatedImg} alt="Annotated CT" className="w-full h-auto max-h-[420px] object-contain" />
            ) : (
              <div className="text-center p-8 text-slate-500">
                <Sliders className="w-10 h-10 mx-auto mb-2 opacity-40" />
                <p className="text-xs">Run analysis to generate diagnostic output</p>
              </div>
            )}
          </div>
        </div>

        {/* Legend / Info Bar */}
        <div className="mt-5 pt-4 border-t border-white/10 flex flex-wrap items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-4">
            <span className="text-slate-400 font-medium">Color Class Legend:</span>
            {modelConfig.noises.map((n) => (
              <div key={n.key} className="flex items-center gap-1.5 text-slate-300">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: n.color }} />
                <span>{n.label}</span>
              </div>
            ))}
          </div>

          {/* Opacity slider for overlay mode */}
          {viewMode === 'overlay' && (
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Mask Opacity:</span>
              <input
                type="range"
                min="0.2"
                max="1.0"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-24 accent-sky-400 cursor-pointer"
              />
              <span className="font-mono text-sky-400">{Math.round(overlayOpacity * 100)}%</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
