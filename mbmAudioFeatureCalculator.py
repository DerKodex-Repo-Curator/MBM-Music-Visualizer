# MBM's Music Visualizer: Audio Feature Calculator
# Calculates relevant audio features from loaded audio.

import librosa
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from scipy.signal import resample

from .mbmMVShared import normalizeArray, chartData, renderChart


class AudioFeatureCalculator:
    """
    Calculates relevant audio features from loaded audio.
    """

    DEF_FEAT_MOD_MAX = 10000.0
    DEF_FEAT_MOD_MIN = -10000.0

    RETURN_TYPES = ("TENSOR_1D", "FLOAT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("FEAT_MODS", "FEAT_SECONDS", "FPS", "CHARTS")
    FUNCTION = "process"
    CATEGORY = "MBMnodes/Audio"

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "intensity": ("FLOAT", {"default": 1.0}),
                "hop_length": ("INT", {"default": 512}),
                "fps_target": ("FLOAT", {"default": 6, "min": -1, "max": 10000}),
                "feat_mod_max": ("FLOAT", {"default": s.DEF_FEAT_MOD_MAX}),
                "feat_mod_min": ("FLOAT", {"default": s.DEF_FEAT_MOD_MIN}),
                "feat_mod_normalize": ([False, True],),
            }
        }

    def process(
        self,
        audio,
        intensity: float,
        hop_length: int,
        fps_target: float,
        feat_mod_max: float,
        feat_mod_min: float,
        feat_mod_normalize: bool,
    ):
        # -----------------------
        # Validation
        # -----------------------
        if feat_mod_max < feat_mod_min:
            raise ValueError("feat_mod_max must be >= feat_mod_min")

        if feat_mod_max == self.DEF_FEAT_MOD_MAX:
            feat_mod_max = None
        if feat_mod_min == self.DEF_FEAT_MOD_MIN:
            feat_mod_min = None

        # -----------------------
        # Load audio (robusto)
        # -----------------------
        if isinstance(audio, str):
            y, sr = librosa.load(audio, sr=None, mono=True)

        elif isinstance(audio, dict):
            sr = int(audio["sample_rate"])
            y = audio["waveform"]
            if hasattr(y, "detach"):
                y = y.detach().cpu().numpy()

        else:
            y, sr = audio
            if hasattr(y, "detach"):
                y = y.detach().cpu().numpy()

        # -----------------------
        # FORCE MONO 1D
        # -----------------------
        y = np.asarray(y, dtype=np.float32)

        if y.ndim == 2:
            # (channels, samples) → mono
            y = librosa.to_mono(y)

        y = y.reshape(-1)

        # -----------------------
        # Duration
        # -----------------------
        duration = librosa.get_duration(y=y, sr=sr, hop_length=hop_length)

        # -----------------------
        # TEMPO (1D)
        # -----------------------
        onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
        tempo = librosa.beat.tempo(
            onset_envelope=onset,
            sr=sr,
            hop_length=hop_length,
            aggregate=None,
        )
        tempo = normalizeArray(np.asarray(tempo).reshape(-1))

        # -----------------------
        # MEL SPECTROGRAM → MEAN (1D)
        # -----------------------
        spectro = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=128,
            fmax=8000,
            hop_length=hop_length,
        )

        spectroMean = np.mean(spectro, axis=0).reshape(-1)
        spectroMean = normalizeArray(spectroMean)

        # DELTA (FORZADO 1D)
        spectroMeanDelta = librosa.feature.delta(spectroMean[None, :])[0]
        spectroMeanDelta = spectroMeanDelta.reshape(-1)

        # -----------------------
        # CHROMA → MEAN (1D)
        # -----------------------
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
        chromaMean = np.mean(chroma, axis=0).reshape(-1)

        # -----------------------
        # FPS / FRAMES
        # -----------------------
        if fps_target <= 0:
            fps = len(tempo) / duration
        else:
            fps = fps_target

        desiredFrames = max(1, round(fps * duration))

        # -----------------------
        # RESAMPLE (TODO 1D)
        # -----------------------
        tempo = resample(tempo, desiredFrames)
        spectroMean = resample(spectroMean, desiredFrames)
        spectroMeanDelta = resample(spectroMeanDelta, desiredFrames)
        chromaMean = resample(chromaMean, desiredFrames)

        # -----------------------
        # FEATURE MODIFIER (1D)
        # -----------------------
        featModifiers = torch.tensor(
            [
                self._calcFeatModifier(
                    intensity,
                    tempo[i],
                    spectroMean[i],
                    spectroMeanDelta[i],
                    chromaMean[i],
                    modMax=feat_mod_max,
                    modMin=feat_mod_min,
                )
                for i in range(desiredFrames)
            ],
            dtype=torch.float32,
        )

        if feat_mod_normalize:
            featModifiers = normalizeArray(
                featModifiers,
                minVal=0.0,
                maxVal=float(featModifiers.max()),
            )

        # -----------------------
        # CHARTS
        # -----------------------
        chartImages = torch.vstack(
            [
                chartData(tempo, "Tempo"),
                chartData(spectroMean, "Spectrogram Mean"),
                chartData(spectroMeanDelta, "Spectrogram Mean Delta"),
                chartData(chromaMean, "Chroma Mean"),
                self._chartFeatMod(
                    featModifiers,
                    intensity,
                    feat_mod_normalize,
                    modMax=feat_mod_max,
                    modMin=feat_mod_min,
                ),
            ]
        )

        return (
            featModifiers,                # 1D tensor ✔
            duration / desiredFrames,     # seconds per frame
            fps,
            chartImages,
        )

    # ------------------------------------------------------------------

    def _calcFeatModifier(
        self,
        intensity: float,
        tempo: float,
        spectroMean: float,
        spectroMeanDelta: float,
        chromaMean: float,
        modMax: Optional[float] = None,
        modMin: Optional[float] = None,
    ) -> float:
        modVal = (
            (tempo + 1.0)
            * (spectroMean + 1.0)
            * (chromaMean + 1.0)
            * (intensity + spectroMeanDelta)
        )

        if modMax is not None and modVal > modMax:
            return modMax
        if modMin is not None and modVal < modMin:
            return modMin
        return modVal

    # ------------------------------------------------------------------

    def _chartFeatMod(
        self,
        featModifiers: torch.Tensor,
        intensity: float,
        isNormalized: bool,
        modMax: Optional[float] = None,
        modMin: Optional[float] = None,
    ) -> torch.Tensor:

        fig, ax = plt.subplots(figsize=(20, 4))
        ax.plot(featModifiers, label="Modifiers")

        if modMax is not None:
            ax.axhline(y=featModifiers.max() if isNormalized else modMax, linestyle="dotted")
        if modMin is not None:
            ax.axhline(y=0 if isNormalized else modMin, linestyle="dotted")

        ax.set_title("Feature Modifiers")
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)

        return renderChart(fig)
