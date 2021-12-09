# DLA
## Text To Speech
### hw3

| model  | Duration loss| Melspec loss| total loss |
| ------ | ------- | ---- | --- |
|FastSpeech + FS alignments| 0.01 | 0.28 | 0.3|


Download dataset, waveglow, alignments
```
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xjf LJSpeech-1.1.tar.bz2
pip install torch==1.10.0+cu111 torchaudio==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

git clone https://github.com/NVIDIA/waveglow.git
pip install -r requirements.txt


wget https://github.com/xcmyz/FastSpeech/blob/master/alignments.zip
unzip alignments.zip
```