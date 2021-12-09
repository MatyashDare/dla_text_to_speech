import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchaudio
import wandb
from tqdm import tqdm
from model import FSModel
import sys
from data_preprocessing import LJSpeechCollator, LJSpeechDataset


class Vocoder(nn.Module):

    def __init__(self):
        super(Vocoder, self).__init__()

        model = torch.load('waveglow_256channels_universal_v5.pt', map_location='cpu')[
            'model']
        self.net = model.remove_weightnorm(model)

    @torch.no_grad()
    def inference(self, spect: torch.Tensor):
        spect = self.net.upsample(spect)

        # trim the conv artifacts
        time_cutoff = self.net.upsample.kernel_size[0] - \
            self.net.upsample.stride[0]
        spect = spect[:, :, :-time_cutoff]

        spect = spect.unfold(2, self.net.n_group, self.net.n_group) \
            .permute(0, 2, 1, 3) \
            .contiguous() \
            .flatten(start_dim=2) \
            .transpose(-1, -2)

        # generate prior
        audio = torch.randn(spect.size(0), self.net.n_remaining_channels, spect.size(-1)) \
            .to(spect.device)

        for k in reversed(range(self.net.n_flows)):
            n_half = int(audio.size(1) / 2)
            audio_0 = audio[:, :n_half, :]
            audio_1 = audio[:, n_half:, :]

            output = self.net.WN[k]((audio_0, spect))

            s = output[:, n_half:, :]
            b = output[:, :n_half, :]
            audio_1 = (audio_1 - b) / torch.exp(s)
            audio = torch.cat([audio_0, audio_1], 1)

            audio = self.net.convinv[k](audio, reverse=True)

            if k % self.net.n_early_every == 0 and k > 0:
                z = torch.randn(
                    spect.size(0), self.net.n_early_size, spect.size(2),
                    device=spect.device
                )
                audio = torch.cat((z, audio), 1)

        audio = audio.permute(0, 2, 1) \
            .contiguous() \
            .view(audio.size(0), -1)

        return audio

    
tokenizer  = torchaudio.pipelines.TACOTRON2_GRIFFINLIM_CHAR_LJSPEECH.get_text_processor()
val_texts = ['A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space']

val_tokens = []
val_lens = []

for text in val_texts:
    tokens, length = tokenizer(text.lower())
    val_tokens.append(tokens)
    val_lens.append(length)

dataloader = DataLoader(LJSpeechDataset('.'), batch_size=10, collate_fn=LJSpeechCollator('cpu'))
model = FSModel(6, 384, (3, 1), 2, 96)
vocoder = Vocoder().eval()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
sr = 22050

with wandb.init(project="tts_check", name="check") as run:
    for n in range(100):   
        model.train()
        i = 0
        for batch in tqdm(dataloader):
            tokens, token_lengths, durations, melspecs_true = batch
            mels_pred, durations_pred = model(tokens, token_lengths, durations)
            mels_pred = mels_pred.permute(0, 2, 1)
            mel_loss = criterion(mels_pred, melspecs_true)
            dur_loss = criterion(torch.log1p(durations_pred), torch.log1p(durations))
            loss = mel_loss + dur_loss
            run.log({"Total loss" : loss}, step=n * len(dataloader) + i)
            run.log({"Melspec Loss" : mel_loss}, step=n * len(dataloader) + i)
            run.log({"Duration Loss" : dur_loss}, step=n * len(dataloader) + i)
            if i + 1 == len(dataloader):
                mel = melspecs_true[0][:, :melspecs_true[0]].unsqueeze(0)
                reconstructed_wav = vocoder.inference(mel)
                run.log({"Audio" : wandb.Audio(reconstructed_wav.squeeze().detach().cpu().numpy(), sr)})
                dur_to_log = (torch.exp(durations_pred[0]) - 1).round().int()
                cumsum = dur_to_log.cumsum(0)
                a = (torch.arange(dur_to_log.sum().item())[None, :].to(device) < (cumsum[:, None])).float()
                b = (torch.arange(dur_to_log.sum().item())[None, :].to(device) >= (cumsum - dur_to_log)[:, None]).float()
                mask = a * b
                run.log({"`Durations predicted" : wandb.Image(mask.detach().cpu().numpy())}, step=n * len(dataloader) + i)
                dur_to_log = durations[0]
                cumsum = dur_to_log.cumsum(0)
                a = (torch.arange(dur_to_log.sum().item())[None, :].to(device) < (cumsum[:, None])).float()
                b = (torch.arange(dur_to_log.sum().item())[None, :].to(device) >= (cumsum - dur_to_log)[:, None]).float()
                mask = a * b
                run.log({"`Durations true" : wandb.Image(mask.detach().cpu().numpy())}, step=n * len(dataloader) + i)
            break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        model.eval()
        audios = []
        for text, tokens, len in zip(val_texts, val_tokens, val_lens):
            tokens = torch.tensor(tokens)
            token_lengths = torch.tensor(token_lengths)
            mels_pred, durations_pred = model(tokens, token_lengths, None)
            mels_pred = mels_pred.permute(0, 2, 1)
            reconstructed_wav = vocoder.inference(mels_pred).squeeze().detach().cpu().numpy()
            audios.append(wandb.Audio(reconstructed_wav, sr, caption=text))
        run.log({"Audio validation" : audios}, step=iteration)
