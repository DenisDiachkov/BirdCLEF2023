from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import os
import librosa
import random
import soundfile as sf
import ast
from tqdm import tqdm

DEFAULT_BIRD_NAMES = ['abethr1', 'abhori1', 'abythr1', 'afbfly1', 'afdfly1', 'afecuc1',
       'affeag1', 'afgfly1', 'afghor1', 'afmdov1', 'afpfly1', 'afpkin1',
       'afpwag1', 'afrgos1', 'afrgrp1', 'afrjac1', 'afrthr1', 'amesun2',
       'augbuz1', 'bagwea1', 'barswa', 'bawhor2', 'bawman1', 'bcbeat1',
       'beasun2', 'bkctch1', 'bkfruw1', 'blacra1', 'blacuc1', 'blakit1',
       'blaplo1', 'blbpuf2', 'blcapa2', 'blfbus1', 'blhgon1', 'blhher1',
       'blksaw1', 'blnmou1', 'blnwea1', 'bltapa1', 'bltbar1', 'bltori1',
       'blwlap1', 'brcale1', 'brcsta1', 'brctch1', 'brcwea1', 'brican1',
       'brobab1', 'broman1', 'brosun1', 'brrwhe3', 'brtcha1', 'brubru1',
       'brwwar1', 'bswdov1', 'btweye2', 'bubwar2', 'butapa1', 'cabgre1',
       'carcha1', 'carwoo1', 'categr', 'ccbeat1', 'chespa1', 'chewea1',
       'chibat1', 'chtapa3', 'chucis1', 'cibwar1', 'cohmar1', 'colsun2',
       'combul2', 'combuz1', 'comsan', 'crefra2', 'crheag1', 'crohor1',
       'darbar1', 'darter3', 'didcuc1', 'dotbar1', 'dutdov1', 'easmog1',
       'eaywag1', 'edcsun3', 'egygoo', 'equaka1', 'eswdov1', 'eubeat1',
       'fatrav1', 'fatwid1', 'fislov1', 'fotdro5', 'gabgos2', 'gargan',
       'gbesta1', 'gnbcam2', 'gnhsun1', 'gobbun1', 'gobsta5', 'gobwea1',
       'golher1', 'grbcam1', 'grccra1', 'grecor', 'greegr', 'grewoo2',
       'grwpyt1', 'gryapa1', 'grywrw1', 'gybfis1', 'gycwar3', 'gyhbus1',
       'gyhkin1', 'gyhneg1', 'gyhspa1', 'gytbar1', 'hadibi1', 'hamerk1',
       'hartur1', 'helgui', 'hipbab1', 'hoopoe', 'huncis1', 'hunsun2',
       'joygre1', 'kerspa2', 'klacuc1', 'kvbsun1', 'laudov1', 'lawgol',
       'lesmaw1', 'lessts1', 'libeat1', 'litegr', 'litswi1', 'litwea1',
       'loceag1', 'lotcor1', 'lotlap1', 'luebus1', 'mabeat1', 'macshr1',
       'malkin1', 'marsto1', 'marsun2', 'mcptit1', 'meypar1', 'moccha1',
       'mouwag1', 'ndcsun2', 'nobfly1', 'norbro1', 'norcro1', 'norfis1',
       'norpuf1', 'nubwoo1', 'pabspa1', 'palfly2', 'palpri1', 'piecro1',
       'piekin1', 'pitwhy', 'purgre2', 'pygbat1', 'quailf1', 'ratcis1',
       'raybar1', 'rbsrob1', 'rebfir2', 'rebhor1', 'reboxp1', 'reccor',
       'reccuc1', 'reedov1', 'refbar2', 'refcro1', 'reftin1', 'refwar2',
       'rehblu1', 'rehwea1', 'reisee2', 'rerswa1', 'rewsta1', 'rindov',
       'rocmar2', 'rostur1', 'ruegls1', 'rufcha2', 'sacibi2', 'sccsun2',
       'scrcha1', 'scthon1', 'shesta1', 'sichor1', 'sincis1', 'slbgre1',
       'slcbou1', 'sltnig1', 'sobfly1', 'somgre1', 'somtit4', 'soucit1',
       'soufis1', 'spemou2', 'spepig1', 'spewea1', 'spfbar1', 'spfwea1',
       'spmthr1', 'spwlap1', 'squher1', 'strher', 'strsee1', 'stusta1',
       'subbus1', 'supsta1', 'tacsun1', 'tafpri1', 'tamdov1', 'thrnig1',
       'trobou1', 'varsun2', 'vibsta2', 'vilwea1', 'vimwea1', 'walsta1',
       'wbgbir1', 'wbrcha2', 'wbswea1', 'wfbeat1', 'whbcan1', 'whbcou1',
       'whbcro2', 'whbtit5', 'whbwea1', 'whbwhe3', 'whcpri2', 'whctur2',
       'wheslf1', 'whhsaw1', 'whihel1', 'whrshr1', 'witswa1', 'wlwwar',
       'wookin1', 'woosan', 'wtbeat1', 'yebapa1', 'yebbar1', 'yebduc1',
       'yebere1', 'yebgre1', 'yebsto1', 'yeccan1', 'yefcan', 'yelbis1',
       'yenspu1', 'yertin1', 'yesbar1', 'yespet1', 'yetgre1', 'yewgre1'
]

class BirdCLEFDataset(Dataset):
    def __init__(
        self, 
        mode, 
        csv_path,
        audio_folder,
        wav_crop_len,
        sample_rate,
        albumentations,
        bird_names=DEFAULT_BIRD_NAMES,
        **kwargs
    ):
        self.mode = mode
        self.audio_folder = audio_folder
        if csv_path is not None:
            self.data_frame = pd.read_csv(csv_path)
        if self.mode == "train":
            self.data_frame = self.data_frame[self.data_frame["rating"] >= kwargs["min_rating"]]
        self.filenames = self.data_frame["filename"].unique()

        self.bird2id = {bird: idx for idx, bird in enumerate(bird_names)}
        self.n_classes = len(bird_names)
        self.wav_crop_len = wav_crop_len
        self.sample_rate = sample_rate
        self.data_frame = self.setup_data_frame()
        self.albumentations_audio = albumentations
    
    def get_duration(self, idx):
        if "duration" in self.data_frame.iloc[idx] and not np.isnan(self.data_frame.iloc[idx]["duration"]):
            return self.data_frame.iloc[idx]["duration"]
        duration = BirdCLEFDataset.get_file_duration(
            os.path.join(self.audio_folder, self.data_frame.iloc[idx]["filename"])
        )
        return duration 
    
    def setup_data_frame(self):
        data_frame = self.data_frame.copy()

        if self.mode == "train":
            data_frame["weight"] = np.clip(data_frame["rating"] / data_frame["rating"].max(), 0.1, 1.0)
            data_frame['target'] = data_frame['primary_label'].apply(self.bird2id.get)
            labels = np.eye(self.n_classes)[data_frame["target"].astype(int).values]
            label2 = data_frame["secondary_labels"].apply(lambda x: self.secondary2target(x)).values
            for i, t in enumerate(label2):
                labels[i, t] = 1
        else:
            targets = data_frame["primary_label"].apply(lambda x: self.birds2target(x)).values
            labels = np.zeros((data_frame.shape[0], self.n_classes))
            for i, t in enumerate(targets):
                labels[i, t] = 1

        data_frame[[f"t{i}" for i in range(self.n_classes)]] = labels

        if self.mode != "train":
            data_frame = data_frame.groupby("filename")

        return data_frame

    @staticmethod
    def get_file_duration(file_path):
        sound, sr = librosa.load(file_path)
        
        return librosa.get_duration(y=sound, sr=sr)

    def __getitem__(self, idx):
        if self.mode == "train":
            row = self.data_frame.iloc[idx]
            fn = row["filename"]
            label = row[[f"t{i}" for i in range(self.n_classes)]].values
            weight = row["weight"]
            #fold = row["fold"]
            fold = -1

            #wav_len = row["length"]
            parts = 1
        else:
            fn = self.filenames[idx]
            row = self.data_frame.get_group(fn)
            label = row[[f"t{i}" for i in range(self.n_classes)]].values
            wav_len = None
            parts = label.shape[0]
            if parts == 1:
                label = label.squeeze(0)
            fold = -1
            weight = 1
        if self.mode == "train":
            #wav_len_sec = wav_len / self.sample_rate
            wav_len_sec = self.get_duration(idx)
            duration = self.wav_crop_len
            max_offset = wav_len_sec - duration
            max_offset = max(max_offset, 1)
            offset = np.random.randint(max_offset)
        else:
            offset = 0.0
            duration = self.wav_crop_len
        wav = self.load_one(fn, offset, duration)
        if wav.shape[0] < (self.wav_crop_len * self.sample_rate):
            pad = int(self.wav_crop_len * self.sample_rate - wav.shape[0])
            wav = np.pad(wav, (0, pad))

        # wav = self.albumentations_audio(samples=wav, sample_rate=self.sample_rate)["data"]
        # if self.mode == "train":
        #     if self.aug_audio:
        #         wav = self.aug_audio(samples=wav, sample_rate=self.sample_rate)
        # else:
        #     if self.val_aug:
        #         wav = self.val_aug(samples=wav, sample_rate=self.sample_rate)

        wav_tensor = torch.tensor(wav)  # (n_samples)
        if parts > 1:
            n_samples = wav_tensor.shape[0]
            wav_tensor = wav_tensor[: n_samples // parts * parts].reshape(
                parts, n_samples // parts
            )
            raise Exception("NOT IMPLEMENTED")
        feature_dict = {
            "input": wav_tensor,
            "target": torch.tensor(label.astype(np.float32)),
            "weight": torch.tensor(weight),
            "fold": torch.tensor(fold),
        }
        return feature_dict

    def __len__(self):
        return len(self.filenames)

    def load_one(self, id_, offset, duration):
        fp = os.path.join(self.audio_folder, id_)
        try:
            wav, sr = librosa.load(fp, sr=None, offset=offset, duration=duration)
        except:
            # print("FAIL READING rec", fp)
            raise Exception("FAIL READING rec", fp)
        return wav

    def birds2target(self, birds):
        birds = birds.split()
        target = [self.bird2id.get(item) for item in birds if not item == "nocall"]
        return target

    def secondary2target(self, secondary_label):
        birds = ast.literal_eval(secondary_label)
        target = [self.bird2id.get(item) for item in birds if not item == "nocall"]
        return target