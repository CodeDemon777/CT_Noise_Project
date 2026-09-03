import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { TrendingUp, Activity, CheckCircle, ShieldCheck } from 'lucide-react';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export default function PerformanceCharts() {
  const epochs = Array.from({ length: 20 }, (_, i) => i + 1);

  const lossData = {
    labels: epochs,
    datasets: [
      {
        label: 'Training Loss',
        data: [0.85,0.62,0.45,0.31,0.22,0.16,0.12,0.09,0.07,0.05,0.04,0.032,0.026,0.021,0.019,0.017,0.016,0.015,0.015,0.0148],
        borderColor: '#38bdf8',
        backgroundColor: 'rgba(56, 189, 248, 0.1)',
        fill: true,
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: 'Validation Loss',
        data: [0.91,0.68,0.51,0.38,0.28,0.21,0.17,0.13,0.11,0.08,0.065,0.052,0.043,0.035,0.029,0.025,0.022,0.019,0.0185,0.0182],
        borderColor: '#818cf8',
        backgroundColor: 'transparent',
        fill: false,
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
    ],
  };

  const metricsData = {
    labels: epochs,
    datasets: [
      {
        label: 'Dice Score',
        data: [0.42,0.58,0.71,0.82,0.88,0.92,0.942,0.958,0.967,0.974,0.979,0.982,0.984,0.9855,0.9868,0.9875,0.9880,0.9883,0.9885,0.9886],
        borderColor: '#38bdf8',
        backgroundColor: 'transparent',
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
      {
        label: 'IoU Score',
        data: [0.31,0.45,0.58,0.69,0.77,0.83,0.87,0.902,0.925,0.941,0.952,0.961,0.967,0.971,0.973,0.9748,0.9760,0.9768,0.9773,0.9778],
        borderColor: '#10b981',
        backgroundColor: 'transparent',
        tension: 0.3,
        borderWidth: 2,
        pointRadius: 2,
      },
    ],
  };

  const chartOptions = (yTitle) => ({
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: '#94a3b8',
          font: { family: 'Inter', size: 11 },
        },
      },
    },
    scales: {
      x: {
        title: { display: true, text: 'Training Epochs', color: '#64748b', font: { size: 10 } },
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
      },
      y: {
        title: { display: true, text: yTitle, color: '#64748b', font: { size: 10 } },
        grid: { color: 'rgba(255, 255, 255, 0.04)' },
        ticks: { color: '#94a3b8', font: { size: 10 } },
      },
    },
  });

  return (
    <section className="w-full max-w-6xl mx-auto px-4 mb-16">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold font-display text-white">AI Performance &amp; Convergence Telemetry</h2>
        <p className="text-xs text-slate-400 mt-1">Empirical cross-entropy loss reduction and Dice segmentation curves</p>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-6">
        <div className="glass-panel p-4 text-center">
          <span className="text-2xl sm:text-3xl font-bold font-mono text-sky-400">0.9886</span>
          <p className="text-[11px] text-slate-400 mt-1">Dice Coefficient</p>
        </div>
        <div className="glass-panel p-4 text-center">
          <span className="text-2xl sm:text-3xl font-bold font-mono text-emerald-400">0.9778</span>
          <p className="text-[11px] text-slate-400 mt-1">Mean IoU Metric</p>
        </div>
        <div className="glass-panel p-4 text-center">
          <span className="text-2xl sm:text-3xl font-bold font-mono text-indigo-400">0.9899</span>
          <p className="text-[11px] text-slate-400 mt-1">Precision</p>
        </div>
        <div className="glass-panel p-4 text-center">
          <span className="text-2xl sm:text-3xl font-bold font-mono text-purple-400">0.0148</span>
          <p className="text-[11px] text-slate-400 mt-1">Final Val Loss</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-5">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <Activity className="w-4 h-4 text-sky-400" />
            <span>Cross-Entropy Training &amp; Validation Loss</span>
          </h4>
          <div className="h-64">
            <Line data={lossData} options={chartOptions('Loss (CE)')} />
          </div>
        </div>

        <div className="glass-panel p-5">
          <h4 className="text-sm font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-emerald-400" />
            <span>Dice &amp; IoU Segmentation Convergence</span>
          </h4>
          <div className="h-64">
            <Line data={metricsData} options={chartOptions('Score (0.0 - 1.0)')} />
          </div>
        </div>
      </div>
    </section>
  );
}
