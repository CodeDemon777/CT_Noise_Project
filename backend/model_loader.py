"""
Model Loader Module
Loads the trained CT noise detection model
"""

import torch
import torch.nn as nn
from pathlib import Path


class UNet3Class(nn.Module):
    """
    UNet architecture for 3-class CT noise segmentation
    Classes: 0=Clean, 1=Gaussian, 2=Poisson
    """
    
    def __init__(self, in_channels=1, out_channels=3, init_features=32):
        super(UNet3Class, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.init_features = init_features
        features = init_features
        
        # Encoder
        self.encoder1 = self._conv_block(in_channels, features)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.encoder2 = self._conv_block(features, features * 2)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.encoder3 = self._conv_block(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.encoder4 = self._conv_block(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._conv_block(features * 8, features * 16)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, 2, 2)
        self.decoder4 = self._conv_block(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, 2, 2)
        self.decoder3 = self._conv_block(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, 2, 2)
        self.decoder2 = self._conv_block(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, 2, 2)
        self.decoder1 = self._conv_block(features * 2, features)
        
        # Final output
        self.final_conv = nn.Conv2d(features, out_channels, 1)
    
    @staticmethod
    def _conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder
        d4 = self.upconv4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.decoder4(d4)
        
        d3 = self.upconv3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.decoder3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.decoder2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.decoder1(d1)
        
        # Final output
        out = self.final_conv(d1)
        return out


def _looks_like_unetplusplus_checkpoint(state_dict: dict) -> bool:
    return all(
        key in state_dict
        for key in (
            "encoder.conv1.weight",
            "decoder.blocks.x_0_0.conv1.0.weight",
            "segmentation_head.0.weight",
        )
    )


def _build_model_from_state_dict(state_dict: dict) -> nn.Module:
    if _looks_like_unetplusplus_checkpoint(state_dict):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "This checkpoint requires segmentation-models-pytorch. "
                "Install the project requirements before running inference."
            ) from exc

        in_channels = state_dict["encoder.conv1.weight"].shape[1]
        out_channels = state_dict["segmentation_head.0.bias"].shape[0]
        return smp.UnetPlusPlus(
            encoder_name="resnet18",
            encoder_weights=None,
            in_channels=in_channels,
            classes=out_channels,
        )

    return UNet3Class(in_channels=1, out_channels=3)


def load_model(model_path: str, device: str = None) -> nn.Module:
    """
    Load the trained model from disk
    
    Args:
        model_path: Path to best_model.pth
        device: Device to load model on ('cpu', 'cuda', None for auto)
    
    Returns:
        Loaded model in evaluation mode
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = _build_model_from_state_dict(state_dict)
    model.load_state_dict(state_dict)
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"📊 Device: {device}")
    
    return model


if __name__ == "__main__":
    # Test model loading
    model_path = Path(__file__).parent.parent / "model" / "best_model.pth"
    model = load_model(str(model_path))
    print(f"Model architecture:\n{model}")
