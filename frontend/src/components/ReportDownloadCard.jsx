import React, { useState } from 'react';
import { FileText, Download, CheckCircle2, RefreshCw } from 'lucide-react';
import { MODELS_CONFIG } from '../utils/constants';

export default function ReportDownloadCard({ activeModel, resultData, currentFile }) {
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadSuccess, setDownloadSuccess] = useState(false);

  const modelConfig = MODELS_CONFIG[activeModel] || MODELS_CONFIG.model1;

  const handleDownload = () => {
    const filename = resultData?.filename || currentFile?.name;
    if (!filename) {
      alert('Please analyze a CT scan first before generating the clinical PDF report.');
      return;
    }

    setIsDownloading(true);
    setDownloadSuccess(false);

    const downloadUrl = `/report?filename=${encodeURIComponent(filename)}&model=${encodeURIComponent(activeModel)}`;
    const tempLink = document.createElement('a');
    tempLink.href = downloadUrl;
    tempLink.setAttribute('download', `${filename.replace(/\.[^/.]+$/, '')}_${activeModel}_report.pdf`);
    tempLink.target = '_blank';
    document.body.appendChild(tempLink);
    tempLink.click();

    setTimeout(() => {
      if (document.body.contains(tempLink)) {
        document.body.removeChild(tempLink);
      }
      setIsDownloading(false);
      setDownloadSuccess(true);
      setTimeout(() => setDownloadSuccess(false), 3000);
    }, 2000);
  };

  return (
    <div className="w-full max-w-4xl mx-auto px-4 mb-12">
      <div className="glass-panel p-8 text-center relative overflow-hidden bg-gradient-to-b from-slate-900/90 to-slate-950/90 border-sky-500/20">
        <div className="w-14 h-14 mx-auto mb-4 rounded-2xl bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400 shadow-lg shadow-sky-500/10">
          <FileText className="w-7 h-7" />
        </div>

        <h3 className="text-xl font-bold font-display text-white mb-2">
          Denoising Report Generation <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-sky-500/20 text-sky-400">({modelConfig.name})</span>
        </h3>
        <p className="text-xs sm:text-sm text-slate-300 max-w-xl mx-auto mb-6 font-light">
          Prepare and download a PDF report containing original scans, segmented regions, metric tables and radiologist diagnostic recommendations.
        </p>

        <button
          onClick={handleDownload}
          disabled={isDownloading}
          className="inline-flex items-center gap-2 px-8 py-3 rounded-xl font-bold text-sm text-white bg-gradient-to-r from-sky-500 via-indigo-500 to-purple-600 hover:from-sky-400 hover:to-purple-500 shadow-xl shadow-sky-500/25 transition-all transform hover:-translate-y-0.5 active:translate-y-0 disabled:opacity-50 cursor-pointer"
        >
          {isDownloading ? (
            <>
              <RefreshCw className="w-4 h-4 animate-spin" />
              <span>Generating Clinical PDF Report…</span>
            </>
          ) : downloadSuccess ? (
            <>
              <CheckCircle2 className="w-4 h-4 text-emerald-300" />
              <span>Report Downloaded Successfully!</span>
            </>
          ) : (
            <>
              <Download className="w-4 h-4" />
              <span>Download PDF Clinical Report</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
