export const MODELS_CONFIG = {
  model1: {
    id: 'model1',
    name: 'Model 1',
    architecture: 'U-Net++ (Nested UNet)',
    badgeColor: 'cyan',
    accentColor: '#38bdf8',
    subTitle: 'Gaussian & Poisson Noise Segmentation',
    description: 'Nested dense skip pathways connecting intermediate sub-networks for high-frequency thermal electronic and photon starvation noise isolation.',
    resolution: '512 × 512',
    weights: 'best_model.pth',
    noises: [
      { key: 'gaussian', label: 'Gaussian Noise', color: '#f43f5e', desc: 'Electronic sensor thermal noise' },
      { key: 'poisson', label: 'Poisson Noise', color: '#38bdf8', desc: 'Photon shot starvation noise' }
    ],
    predictEndpoint: '/predict',
    demoEndpoint: '/demo/model1',
    metrics: { dice: '0.9886', iou: '0.9778', precision: '0.9899', recall: '0.9875' }
  },
  model2: {
    id: 'model2',
    name: 'Model 2',
    architecture: 'Attention U-Net',
    badgeColor: 'emerald',
    accentColor: '#10b981',
    subTitle: 'Poisson & Multiplicative Speckle Noise',
    description: 'Additive Attention Gates (AG) automatically suppress irrelevant background lung regions while highlighting granular acoustic scattering and photon shot artifacts.',
    resolution: '512 × 512',
    weights: 'Joshna.pth',
    noises: [
      { key: 'poisson', label: 'Poisson Noise', color: '#f43f5e', desc: 'Photon shot starvation noise' },
      { key: 'speckle', label: 'Speckle Noise', color: '#10b981', desc: 'Multiplicative acoustic scattering' }
    ],
    predictEndpoint: '/predict/model2',
    demoEndpoint: '/demo/model2',
    metrics: { gating: 'Additive AG', params: '34.8M', sensitivity: 'Co-occurrence' }
  },
  model3: {
    id: 'model3',
    name: 'Model 3',
    architecture: 'DeepLabV3+ (ASPP)',
    badgeColor: 'amber',
    accentColor: '#f59e0b',
    subTitle: 'Salt & Pepper & RVIN Impulse Noise',
    description: 'Atrous Spatial Pyramid Pooling (ASPP) with dilated convolutions captures multi-scale context to distinguish isolated saturated impulse pixels from delicate lung parenchymal textures.',
    resolution: '512 × 512',
    weights: 'Jahnavi (1).pth',
    noises: [
      { key: 'salt_pepper', label: 'Salt & Pepper Noise', color: '#f59e0b', desc: 'Min/max saturated pixel impulses' },
      { key: 'rvin', label: 'RVIN Noise', color: '#a855f7', desc: 'Random-Valued Impulse Noise' }
    ],
    predictEndpoint: '/predict/model3',
    demoEndpoint: '/demo/model3',
    metrics: { asppRates: '[1, 6, 12, 18]', decoder: 'Low-Level Fusion', precision: 'Pixel-level' }
  },
  model4: {
    id: 'model4',
    name: 'Model 4',
    architecture: 'NoiseCNN & 2D FFT',
    badgeColor: 'purple',
    accentColor: '#c084fc',
    subTitle: 'Quantization & Periodic Striping Noise',
    description: 'Deep convolutional feature extractor combined with Fast Fourier Transform (2D FFT) spectrum analysis to expose harmonic frequency spikes and ADC bit-depth truncation.',
    resolution: '128 × 128 (FFT 512×512)',
    weights: 'Vasanth (2).pth',
    noises: [
      { key: 'quantization', label: 'Quantization Noise', color: '#eab308', desc: 'ADC bit-depth rounding steps' },
      { key: 'periodic', label: 'Periodic Noise', color: '#a855f7', desc: 'Harmonic detector stripe interference' }
    ],
    predictEndpoint: '/predict/model4',
    demoEndpoint: '/demo/model4',
    metrics: { fftAnalysis: '2D Spectrum', speed: '< 25ms', output: 'Softmax Confidence' }
  }
};
