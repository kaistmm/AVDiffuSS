import numpy as np
import glob
import tqdm
import torch
import os
from argparse import ArgumentParser
import time
from pypapi import events, papi_high as high

from sgmse.model import StochasticRegenerationModel

from sgmse.util.other import *

from sgmse.util.other import si_sdr, pad_spec
from pesq import pesq
from pystoi import stoi

import cv2
import pickle
import librosa

EPS_LOG = 1e-10


def videocap(path, start_frame, for_sync=False): # for VoxCeleb2
    # start_frame is set according to the audio frames
    vid_start = int(start_frame//16000*25)
    cap = cv2.VideoCapture(path)
    if cap.isOpened():
        frames=[]
        for i in range(vid_start+51):
            ret, img = cap.read()
            if i<vid_start:
                continue

            if ret:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (112,112))
                frames.append(img)
            else:
                frames = np.array(frames)
                frames = np.pad(frames, ((0, 51-i), (0,0), (0,0)), 'wrap')
                assert frames.shape == (51, 112, 112), "padding is set wrong"
                return frames
        frames = np.array(frames)
        return frames # (51, H, W)
    else:
        #print(path, " is not opened. Getting other video…")
        return None



def load_audio_vox(file_path, max_len, sample_rate=16000):
    audio, sample_rate = librosa.load(file_path, sr=sample_rate) # mono as default
    audiosize = audio.shape[0]
    if audiosize < max_len:
        start_frame=0
        shortage = max_len - audiosize
        min_len = sample_rate*30//25 # 최소 30프레임
        if audiosize < min_len:
            #print(file_path + " is too short. Trying another video…")
            return None, 0
        audio = np.pad(audio, (0, shortage), 'wrap')
        if np.all((audio==0)):
            #print(file_path+" loaded as zero array. Trying another video…")
            return None, 0
        
    else:
        start_frame = 0
        audio = audio[0:max_len]
        while np.all((audio==0)):
            
            #print(file_path+" loaded as zero array. Trying to get next section…")
            start_frame += max_len
            if audiosize < start_frame+max_len:
                #print(f"End of {file_path}. Trying another video…")
                return None, 0
            audio = audio[start_frame:start_frame+max_len]
        
    
    return audio, start_frame

def activelev(data):
    max_amp = np.std(data)
    return data/max_amp

def main():
    # Tags
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()
    for parser_ in (base_parser, parser):
        parser_.add_argument("--ckpt", type=str, default='./AVDiffuSS.ckpt')
        parser_.add_argument("--mode", type=str, default="storm")
        parser_.add_argument('--log_path', type=str, default='./test_result.txt')
        parser_.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
        parser_.add_argument("--corrector-steps", type=int, default=1, help="Number of corrector steps")
        parser_.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics.")
        parser_.add_argument("--N", type=int, default=30, help="Number of reverse steps")
        parser_.add_argument("--testset", default='vox', type=str, choices=['lrs3', 'vox'])
        parser_.add_argument("--data_dir", default='/mnt/datasets/voxcelebs/voxceleb2/', type=str, help='path of data directory corresponding to the testset choice')  # LRS3: /mnt/datasets/lip_reading/lrs3/ 

    args = parser.parse_args()

    checkpoint_file = args.ckpt
    model_sr = 16000
    model_cls = StochasticRegenerationModel
    model = model_cls.load_from_checkpoint(
        checkpoint_file, base_dir="",
        batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()

    pckl_path = f'./{args.testset}_test.pckl'
    with open(pckl_path, 'rb') as f:
        test_data = pickle.load(f)

    n_total = len(test_data)

    scores = {'pesq':[], 'stoi':[], 'estoi':[], 'si_sdr':[]}
    with open(args.log_path, 'a') as f:
        f.write(f"Evaluate separation for outputs using {args.testset}\n")
        f.write("  pesq,     stoi,    estoi,    si_sdr\n")
    
    if args.testset=='vox':
        audio_dir = os.path.join(args.data_dir,'test/wav')
        video_dir = os.path.join(args.data_dir, 'test/mp4')
            
    elif args.testset=='lrs3':
        audio_dir = os.path.join(args.data_dir,'test')
        video_dir = os.path.join(args.data_dir,'test')


    for iden_dict in tqdm.tqdm(test_data, dynamic_ncols=True):
        if args.testset=='vox':
            iden1, iden2 = iden_dict.values()
            audio1_path = os.path.join(audio_dir, iden1+'.wav')
            audio2_path = os.path.join(audio_dir, iden2+'.wav')
            video1_path = os.path.join(video_dir, iden1+'.mp4')
            video2_path = os.path.join(video_dir, iden2+'.mp4')

        elif args.testset=='lrs3':
            iden1, iden2 = iden_dict.values()
            audio1_path = os.path.join(audio_dir, iden1)
            audio2_path = os.path.join(audio_dir, iden2)
            video1_path = os.path.join(video_dir, iden1[:-4]+'.mp4')
            video2_path = os.path.join(video_dir, iden2[:-4]+'.mp4')

        clean1, start_frame1 = load_audio_vox(audio1_path, max_len=int(16000 * 2.04), sample_rate=model_sr)
        clean2, start_frame2 = load_audio_vox(audio2_path, max_len=int(16000 * 2.04), sample_rate=model_sr)

        if clean1 is None or clean2 is None:
            continue
        

        visualFeature1 = videocap(video1_path, start_frame1)
        visualFeature2 = videocap(video2_path, start_frame2)

        clean1_n = activelev(clean1)
        clean2_n = activelev(clean2)
        noisy = clean1_n + clean2_n
        
        t = np.random.normal() * 0.5 + 0.9
        lower=0.3
        upper=0.99
        if t < lower or t > upper:
            t = np.random.uniform(lower, upper) 
        scale = t

        max_amp = np.max(np.abs([clean1_n, clean2_n, noisy]))
        mix_scale = 1/max_amp*scale
        clean1 = clean1_n * mix_scale
        clean2 = clean2_n * mix_scale
        mix = noisy * mix_scale

        x1 = np.expand_dims(clean1, 0)
        x2 = np.expand_dims(clean2, 0)
        y = torch.Tensor(np.expand_dims(mix, 0)).cuda()
        visualFeature1 = torch.Tensor(visualFeature1).cuda()
        visualFeature2 = torch.Tensor(visualFeature2).cuda()

        visualFeatures = [visualFeature1, visualFeature2]
        gt_list = [x1, x2]

        _pesq, _si_sdr, _estoi, _stoi = 0., 0., 0., 0.
        for idx, visfeat in enumerate(visualFeatures):
            x = gt_list[idx]
            y = torch.Tensor(np.expand_dims(mix, 0)).cuda()
            x_hat, _ = model.enhance(y, context = visfeat, corrector=args.corrector, corrector_steps=args.corrector_steps, snr=args.snr, N=args.N)
            if x_hat.ndim == 1:
                x_hat = x_hat.unsqueeze(0)
                
            if x.ndim == 1:
                x = x
                x_hat = x_hat.cpu().numpy()
                y = y.cpu().numpy()
            else: #eval only first channel
                x = x[0]
                x_hat = x_hat[0].cpu().numpy()
                y = y[0].cpu().numpy()

            _si_sdr += si_sdr(x, x_hat)
            _pesq += pesq(16000, x, x_hat, 'wb') 
            _estoi += stoi(x, x_hat, 16000, extended=True)
            _stoi += stoi(x, x_hat, 16000, extended=False)
            
        pesq_score = _pesq/2
        stoi_score = _stoi/2
        estoi_score = _estoi/2
        si_sdr_score = _si_sdr/2
        scores['pesq'].append(pesq_score)
        scores['stoi'].append(stoi_score)
        scores['estoi'].append(estoi_score)
        scores['si_sdr'].append(si_sdr_score)
        output_file = open(args.log_path,'a+')
        output_file.write("%3f, %3f, %3f, %3f\n" % (pesq_score, stoi_score, estoi_score, si_sdr_score))
        output_file.close()

    avg_metrics = {}
    for metric, values in scores.items():
        avg_metric = sum(values)/len(values)
        print(f"{metric}: {avg_metric}")
        avg_metrics[metric] = avg_metric

    output_file = open(args.log_path, 'a+')
    for metric, avg_metric in avg_metrics.items():
        output_file.write("%s: %3f\n" % (metric, avg_metric))
    output_file.close()
    print(f"Finished evaluating for {args.ckpt}.")


if __name__=='__main__':
    main()
