import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchaudio as ta
from torch.cuda.amp import GradScaler, autocast
import timm

from .gem import GeM
from .mixup import Mixup


class BirdCLEFModel(nn.Module):
    def __init__(
        self,
        n_classes,
        sample_rate=32000,
        window_size=2048,
        f_min=50,
        f_max=16000,
        power=2.0,
        mel_bins=128,
        hop_size=512,
        top_db=80.0,
        backbone="tf_efficientnet_b0_ns",
        pretrained=True,
        pretrained_weights=None,
        in_channels=1,
        mix_beta=0.2,
        wav_crop_len=5,
        mel_norm=False,
        mixup1=False,
        mixup2=False,
    ):
        super(BirdCLEFModel, self).__init__()
        self.n_classes = n_classes

        self.mel_spec = ta.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=window_size,
            win_length=window_size,
            f_min=f_min,
            f_max=f_max,
            pad=0,
            n_mels=mel_bins,
            hop_length=hop_size,
            power=power,
            normalized=False,
        )

        self.amplitude_to_db = ta.transforms.AmplitudeToDB(top_db=top_db)
        self.wav2img = torch.nn.Sequential(self.mel_spec, self.amplitude_to_db)
        self.masking = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=in_channels,
        )

        if "efficientnet" in backbone:
            backbone_out = self.backbone.num_features
        else:
            backbone_out = self.backbone.feature_info[-1]["num_chs"]

        self.global_pool = GeM()

        self.head_23 = nn.Linear(backbone_out, self.n_classes)

        if pretrained_weights is not None:
            sd = torch.load(pretrained_weights, map_location="cpu")["model"]
            sd = {k.replace("module.", ""): v for k, v in sd.items()}
            self.load_state_dict(sd, strict=True)
            print("weights loaded from", pretrained_weights)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        self.mixup1 = mixup1
        self.mixup2 = mixup2
        self.mixup = Mixup(mix_beta=mix_beta)

        self.factor = int(wav_crop_len / 5.0)

        self.mel_norm = mel_norm

    def forward(self, batch):

        if not self.training:
            x = batch["input"]
            if len(x.shape) == 3:
                bs, parts, time = x.shape
            elif len(x.shape) == 2:
                bs, time = x.shape
                parts = bs
            else:
                raise ValueError("NUIMDAAAAA")
            x = x.reshape(parts, time)
            y = batch["target"]
            y = y[0]
        else:

            x = batch["input"]
            y = batch["target"]
            bs, time = x.shape
            x = x.reshape(bs * self.factor, time // self.factor)

        with autocast(enabled=False):
            x = self.masking(self.wav2img(x))  # (bs, mel, time)
            if self.mel_norm:
                x = (x + 80) / 80

        x = x.permute(0, 2, 1)
        x = x[:, None, :, :]

        weight = batch["weight"]

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)

            if self.mixup1:
                x, y, weight = self.mixup(x, y, weight)
            if self.mixup2:
                x, y, weight = self.mixup(x, y, weight)

            x = x.reshape(b, t, c, f)
            x = x.permute(0, 2, 1, 3)
        
        #print(x.shape)
        x = self.backbone(x)

        if self.training:
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3)
            x = x.reshape(b // self.factor, self.factor * t, c, f)
            x = x.permute(0, 2, 1, 3)
        x = self.global_pool(x)
        x = x[:, :, 0, 0]
        logits = self.head_23(x)

        return {"logits": logits, "target": y}