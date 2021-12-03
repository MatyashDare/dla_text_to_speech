import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from tqdm import tqdm


from typing import Tuple, Dict, Optional, List, Union
import torch
from torch import nn
import torchaudio
from dataclasses import dataclass
from matplotlib import pyplot as plt
from torch.nn.utils.rnn import pad_sequence
from src.data.melspectrogram import MelSpectrogramConfig






@dataclass
class Segment:


    def __repr__(self):
        return "{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


class GraphemeAligner(nn.Module):

    def __init__(self):
        super().__init__()

        self._wav2vec2 = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model()
        self._labels = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_labels()
        self._char2index = {c: i for i, c in enumerate(self._labels)}
        self._unk_index = self._char2index['<unk>']
        self._resampler = torchaudio.transforms.Resample(
            orig_freq=MelSpectrogramConfig.sr, new_freq=16_000
        )

    def _decode_text(self, text):
        text = text.replace(' ', '|').upper()
        return torch.tensor([
            self._char2index.get(char, self._unk_index)
            for char in text
        ]).long()

    @torch.no_grad()
    def forward(
            self,
            wavs: torch.Tensor,
            wav_lengths: torch.Tensor,
            texts: Union[str, List[str]]
    ):
        if isinstance(texts, str):
            texts = [texts]
        batch_size = wavs.shape[0]

        durations = []
        for index in range(batch_size):
            current_wav = wavs[index, :wav_lengths[index]].unsqueeze(dim=0)
            current_wav = self._resampler(current_wav)
            emission, _ = self._wav2vec2(current_wav)
            emission = emission.log_softmax(dim=-1).squeeze(dim=0).cpu()

            tokens = self._decode_text(texts[index])

            trellis = self._get_trellis(emission, tokens)
            path = self._backtrack(trellis, emission, tokens)
            segments = self._merge_repeats(texts[index], path)

            num_frames = emission.shape[0]
            relative_durations = torch.tensor([
                segment.length / num_frames for segment in segments
            ])

            durations.append(relative_durations)

        durations = pad_sequence(durations).transpose(0, 1)
        return durations

    def _get_trellis(self, emission, tokens, blank_id=0):
        num_frame = emission.size(0)
        num_tokens = len(tokens)

        # Trellis has extra dimension for both time axis and tokens.
        # The extra dim for tokens represents <SoS> (start-of-sentence)
        # The extra dim for time axis is for simplification of the code.
        trellis = torch.full((num_frame + 1, num_tokens + 1), -float('inf'))
        trellis[:, 0] = 0
        for t in range(num_frame):
            trellis[t + 1, 1:] = torch.maximum(
                # Score for staying at the same token
                trellis[t, 1:] + emission[t, blank_id],

                # Score for changing to the next token
                trellis[t, :-1] + emission[t, tokens],
            )
        return trellis

    def _backtrack(self, trellis, emission, tokens, blank_id=0):
        # Note:
        # j and t are indices for trellis, which has extra dimensions
        # for time and tokens at the beginning.
        # When refering to time frame index `T` in trellis,
        # the corresponding index in emission is `T-1`.
        # Similarly, when refering to token index `J` in trellis,
        # the corresponding index in transcript is `J-1`.
        j = trellis.size(1) - 1
        t_start = torch.argmax(trellis[:, j]).item()

        path = []
        for t in range(t_start, 0, -1):
            # 1. Figure out if the current position was stay or change
            # Note (again):
            # `emission[J-1]` is the emission at time frame `J` of trellis dimension.
            # Score for token staying the same from time frame J-1 to T.
            stayed = trellis[t - 1, j] + emission[t - 1, blank_id]
            # Score for token changing from C-1 at T-1 to J at T.
            changed = trellis[t - 1, j - 1] + emission[t - 1, tokens[j - 1]]

            # 2. Store the path with frame-wise probability.
            prob = emission[t - 1, tokens[j - 1]
            if changed > stayed else 0].exp().item()
            # Return token index and time index in non-trellis coordinate.
            path.append(Point(j - 1, t - 1, prob))

            # 3. Update the token
            if changed > stayed:
                j -= 1
                if j == 0:
                    break

        else:
            raise ValueError('Failed to align')

        return path[::-1]

    def _merge_repeats(self, text, path):
        i1, i2 = 0, 0
        segments = []
        while i1 < len(path):
            while i2 < len(path) and path[i1].token_index == path[i2].token_index:
                i2 += 1
            score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
            segments.append(
                Segment(
                    text[path[i1].token_index],
                    path[i1].time_index,
                    path[i2 - 1].time_index + 1,
                    score
                )
            )
            i1 = i2

        return segments

    @staticmethod
    def plot_trellis_with_path(trellis, path):
        # to plot trellis with path, we take advantage of 'nan' value
        trellis_with_path = trellis.clone()
        for i, p in enumerate(path):
            trellis_with_path[p.time_index, p.token_index] = float('nan')
        plt.imshow(trellis_with_path[1:, 1:].T, origin='lower')



from typing import Tuple, Dict, Optional, List, Union
from itertools import islice
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from dataclasses import dataclass
from src.data.aligner import GraphemeAligner
from src.data.melspectrogram import MelSpectrogram, MelSpectrogramConfig


@dataclass
class Batch:
    waveform: torch.Tensor
    waveforn_length: torch.Tensor
    melspec: torch.Tensor
    melspec_length: torch.Tensor
    transcript: List[str]
    tokens: torch.Tensor
    token_lengths: torch.Tensor
    duration_multipliers: Optional[torch.Tensor] = None

    def to(self, device: torch.device) -> 'Batch':
        raise NotImplementedError


class LJSpeechCollator:
    def __init__(self, device='cpu'):
        self.aligner = GraphemeAligner().to(device)
        self.featurizer = MelSpectrogram(MelSpectrogramConfig()).to(device)
        self.hop_length = MelSpectrogramConfig().hop_length

    def __call__(self, instances: List[Tuple]) -> Dict:
        waveform, waveforn_length, transcript, tokens, token_lengths = list(
            zip(*instances)
        )
        waveform = pad_sequence([
            waveform_[0] for waveform_ in waveform
        ]).transpose(0, 1)
        waveforn_length = torch.cat(waveforn_length)

        tokens = pad_sequence([
            tokens_[0] for tokens_ in tokens
        ]).transpose(0, 1)
        token_lengths = torch.cat(token_lengths)
        melspec_length = waveforn_length // self.hop_length
        durations = self.aligner(
            waveform, waveforn_length, transcript
        )
        melspec = self.featurizer(waveform)
        token_padded_length = tokens.shape[1]
        melspec_padded_length = melspec.shape[2]
        duration_multipliers = durations * melspec_length[:, None]

        n_mels_for_padds = (melspec_padded_length - melspec_length) / (token_padded_length - token_lengths)
        n_mels_for_padds[n_mels_for_padds.isinf()] = 0
        padding_durations = (torch.arange(token_padded_length)[None, :] > token_lengths[:, None]) * n_mels_for_padds[:, None]
        duration_multipliers += padding_durations
        duration_multipliers = torch.round(duration_multipliers)

        error = melspec_padded_length - duration_multipliers.sum(1)
        error_shift = (torch.arange(token_padded_length)[None, :] < torch.abs(error)[:, None]).int() * torch.sign(error)[:, None]
        duration_multipliers += error_shift
        return {"waveform" : waveform,
                "waveforn_length" : waveforn_length,
                "melspec" : melspec,
                "melspec_length" : melspec_length,
                "transcript" : transcript,
                "tokens" : tokens,
                "token_lengths" : token_lengths,
                "duration_multipliers" : duration_multipliers}



import torch
from torch import nn
import torchaudio


class LJSpeechDataset(torchaudio.datasets.LJSPEECH):

    def __init__(self, root):
        super().__init__(root=root)
        self._tokenizer = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()

    def __getitem__(self, index: int):
        waveform, _, _, transcript = super().__getitem__(index)
        waveforn_length = torch.tensor([waveform.shape[-1]]).int()

        tokens, token_lengths = self._tokenizer(transcript)

        return waveform, waveforn_length, transcript, tokens, token_lengths

    def decode(self, tokens, lengths):
        result = []
        for tokens_, length in zip(tokens, lengths):
            text = "".join([
                self._tokenizer.tokens[token]
                for token in tokens_[:length]
            ])
            result.append(text)
        return result
mport torch
from torch import nn
import torchaudio
import librosa
from dataclasses import dataclass


@dataclass
class MelSpectrogramConfig:
    sr: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    n_fft: int = 1024
    f_min: int = 0
    f_max: int = 8000
    n_mels: int = 80
    power: float = 1.0

    # value of melspectrograms if we fed a silence into `MelSpectrogram`
    pad_value: float = -11.5129251


class MelSpectrogram(nn.Module):

    def __init__(self, config: MelSpectrogramConfig):
        super(MelSpectrogram, self).__init__()

        self.config = config

        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sr,
            win_length=config.win_length,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            f_min=config.f_min,
            f_max=config.f_max,
            n_mels=config.n_mels
        )

        # The is no way to set power in constructor in 0.5.0 version.
        self.mel_spectrogram.spectrogram.power = config.power

        # Default `torchaudio` mel basis uses HTK formula. In order to be compatible with WaveGlow
        # we decided to use Slaney one instead (as well as `librosa` does by default).
        mel_basis = librosa.filters.mel(
            sr=config.sr,
            n_fft=config.n_fft,
            n_mels=config.n_mels,
            fmin=config.f_min,
            fmax=config.f_max
        ).T
        self.mel_spectrogram.mel_scale.fb.copy_(torch.tensor(mel_basis))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        :param audio: Expected shape is [B, T]
        :return: Shape is [B, n_mels, T']
        """

        mel = self.mel_spectrogram(audio) \
            .clamp_(min=1e-5) \
            .log_()

        return mel




dataloader = DataLoader(LJSpeechDataset('.'), batch_size=10, collate_fn=LJSpeechCollator())
model = FastSpeechModel(51, 10000, 1, 80, 30, 60, 9, nn.ReLU, 1, 128, 0)
vocoder = Vocoder().eval()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

log_audio_every = 250
log_loss_every = 10

with wandb.init(project="text_to_speech", name="batch_overfit") as run:
    batch = next(iter(dataloader))
    print(batch)
    for i in tqdm(range(1, 5000)):
        result = model(batch)
        print(result.shape)
        loss = criterion(result, batch['melspec'])
        if i % log_loss_every == 0:
            run.log({"loss" : loss})
        if i % log_audio_every == 0:
            print("Logging audio")
            mel_to_log = result[0]
            melspec_to_log  = result[0][:, :batch['melspec_length'][0]].unsqueeze(0)
            reconstructed_wav = vocoder.inference(melspec_to_log).squeeze().detach().cpu().numpy()
            run.log({"Audio" : wandb.Audio(reconstructed_wav, 22050)})
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
