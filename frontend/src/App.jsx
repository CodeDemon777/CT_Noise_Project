import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import ModelSwitcher from './components/ModelSwitcher';
import UploadZone from './components/UploadZone';
import ResultsViewport from './components/ResultsViewport';
import SeverityDashboard from './components/SeverityDashboard';
import NoiseHistogram from './components/NoiseHistogram';
import ReportDownloadCard from './components/ReportDownloadCard';
import PerformanceCharts from './components/PerformanceCharts';
import ModelSpecsSection from './components/ModelSpecsSection';
import TechStackSection from './components/TechStackSection';
import AgileRoadmapSection from './components/AgileRoadmapSection';
import Footer from './components/Footer';
import ParticleCanvas from './components/ParticleCanvas';
import { MODELS_CONFIG } from './utils/constants';

export default function App() {
  const [activeModel, setActiveModel] = useState('model1');
  const [currentFile, setCurrentFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  // Cache results per model
  const [resultsByModel, setResultsByModel] = useState({
    model1: null,
    model2: null,
    model3: null,
    model4: null,
  });

  // Keyboard shortcut support
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === '1') setActiveModel('model1');
      if (e.key === '2') setActiveModel('model2');
      if (e.key === '3') setActiveModel('model3');
      if (e.key === '4') setActiveModel('model4');
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  const handleFileUpload = (file) => {
    setCurrentFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    setErrorMessage(null);
  };

  const handleRunAnalysis = async () => {
    if (!currentFile) return;

    setIsLoading(true);
    setErrorMessage(null);

    const modelConfig = MODELS_CONFIG[activeModel] || MODELS_CONFIG.model1;
    const formData = new FormData();
    formData.append('file', currentFile);

    try {
      const response = await fetch(modelConfig.predictEndpoint, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.error || `Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResultsByModel((prev) => ({
        ...prev,
        [activeModel]: data,
      }));
    } catch (err) {
      console.error('Inference error:', err);
      setErrorMessage(err.message || 'Analysis failed. Please ensure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRunDemo = async () => {
    setIsLoading(true);
    setErrorMessage(null);

    const modelConfig = MODELS_CONFIG[activeModel] || MODELS_CONFIG.model1;

    try {
      const response = await fetch(modelConfig.demoEndpoint);
      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.error || `Demo failed with status ${response.status}`);
      }

      const data = await response.json();

      if (data.original_url) {
        setPreviewUrl(data.original_url);
        setCurrentFile({ name: data.filename || 'demo_phantom.png' });
      }

      setResultsByModel((prev) => ({
        ...prev,
        [activeModel]: data,
      }));
    } catch (err) {
      console.error('Demo error:', err);
      setErrorMessage(err.message || 'Demo execution failed.');
    } finally {
      setIsLoading(false);
    }
  };

  const currentResult = resultsByModel[activeModel];

  return (
    <div className="min-h-screen flex flex-col bg-slate-950 text-slate-100 relative">
      <ParticleCanvas />

      <div className="relative z-10 flex-1 flex flex-col">
        <Navbar />

        <main className="flex-1">
          <HeroSection />

          <div id="models">
            <ModelSwitcher activeModel={activeModel} onSelectModel={setActiveModel} />

            <UploadZone
              activeModel={activeModel}
              currentFile={currentFile}
              previewUrl={previewUrl}
              isLoading={isLoading}
              onFileUpload={handleFileUpload}
              onRunAnalysis={handleRunAnalysis}
              onRunDemo={handleRunDemo}
            />

            {errorMessage && (
              <div className="w-full max-w-4xl mx-auto px-4 mb-6">
                <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-400 text-xs font-medium flex items-center gap-2">
                  <span>⚠️ {errorMessage}</span>
                </div>
              </div>
            )}

            {currentResult && (
              <>
                <ResultsViewport
                  activeModel={activeModel}
                  resultData={currentResult}
                  isLoading={isLoading}
                  rawImageUrl={previewUrl}
                />

                <SeverityDashboard activeModel={activeModel} resultData={currentResult} />

                <NoiseHistogram resultData={currentResult} activeModel={activeModel} />

                <ReportDownloadCard
                  activeModel={activeModel}
                  resultData={currentResult}
                  currentFile={currentFile}
                />
              </>
            )}
          </div>

          <PerformanceCharts />

          <ModelSpecsSection />

          <TechStackSection />

          <AgileRoadmapSection />
        </main>

        <Footer />
      </div>
    </div>
  );
}
