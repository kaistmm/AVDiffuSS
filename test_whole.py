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
import soundfile as sf

EPS_LOG = 1e-10
sr = 16000
	

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def prep_audio(audio1_path, audio2_path, sample_rate=sr):
	clean1, sample_rate = librosa.load(audio1_path, sr=sr)
	clean2, sample_rate = librosa.load(audio2_path, sr=sr)

	min_len = min(len(clean1), len(clean2))

	clean1_n = activelev(clean1[:min_len])
	clean2_n = activelev(clean2[:min_len])
	clean1_n = activelev(clean1[:min_len])
	clean2_n = activelev(clean2[:min_len])
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

	return [clean1, clean2], mix 


def videocap(path, start_frame):
	vid_start = int(start_frame/16000*25)
	cap = cv2.VideoCapture(path)
	vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
				if len(frames)==0:
					print(path, " is not opened…")
					import pdb; pdb.set_trace()
					return None
				frames = np.array(frames)
				
				try:
					frames = np.pad(frames, ((0, vid_start+51-i), (0,0), (0,0)), 'wrap')
				except:
					import pdb; pdb.set_trace()
				assert frames.shape == (51, 112, 112), "padding is set wrong"
				return frames
		frames = np.array(frames)
		return frames # (51, H, W)
	else:
		print(path, " is not opened…")
		return None

def prep_video(video1_path, video2_path, start_frame):
	visualFeature1 = videocap(video1_path, start_frame)
	visualFeature2 = videocap(video2_path, start_frame)
	if visualFeature1 is None:
		print(f"{video1_path} is invalid!!! ")
		return None
	elif visualFeature2 is None:
		print(f"{video2_path} is invalid!!! ")
		return None
	visualFeature1 = torch.Tensor(visualFeature1).cuda()
	visualFeature2 = torch.Tensor(visualFeature2).cuda()
	return [visualFeature1, visualFeature2]


def activelev(data):
	max_amp = np.std(data)
	return data/max_amp

def save_audio(pred_list, den_list, save_root, sr=16000):
	i=0
	pred1_path = os.path.join(save_root, '%02d_pred1.wav' % i)
	while os.path.exists(pred1_path):
		i+=1
		pred1_path = os.path.join(save_root, '%02d_pred1.wav' % i)
	pred2_path = os.path.join(save_root, '%02d_pred2.wav' % i)
	den1_path = os.path.join(save_root, '%02d_den1.wav' % i)
	den2_path = os.path.join(save_root, '%02d_den2.wav' % i)
	sf.write(pred1_path, pred_list[0], sr)
	sf.write(pred2_path, pred_list[1], sr)
	sf.write(den1_path, den_list[0], sr)
	sf.write(den2_path, den_list[1], sr)
	return



def main():
	# Tags
	base_parser = ArgumentParser(add_help=False)
	parser = ArgumentParser()
	for parser_ in (base_parser, parser):
		parser_.add_argument("--ckpt", type=str, default='./AVDiffuSS.ckpt')
		parser_.add_argument("--mode", type=str, default="storm")
		parser_.add_argument('--log_path', type=str, default='./test_results.txt')
		parser_.add_argument("--corrector", type=str, choices=("ald", "langevin", "none"), default="ald", help="Corrector class for the PC sampler.")
		parser_.add_argument("--corrector-steps", type=int, default=1, help="Number of corrector steps")
		parser_.add_argument("--snr", type=float, default=0.5, help="SNR value for (annealed) Langevin dynamics.")
		parser_.add_argument("--N", type=int, default=30, help="Number of reverse steps")
		parser_.add_argument("--testset", default='vox', type=str, choices=['lrs3', 'vox'])
		parser_.add_argument("--data_dir", default='/mnt/datasets/voxcelebs/voxceleb2/', type=str, help='path of data directory corresponding to the testset choice')  # LRS3: /mnt/datasets/lip_reading/lrs3/ 
		parser_.add_argument("--audio_save_root", default='', type=str, help='Specify this to save enhanced audios')
		parser_.add_argument("--hop_length", type=float, default=0.04)
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

	if not os.path.isdir(args.audio_save_root):
		if args.audio_save_root!='':
			os.makedirs(args.audio_save_root)

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
		args.data_dir = '/mnt/datasets/lip_reading/lrs3/'
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

		gt_list_long, mix_long = prep_audio(audio1_path, audio2_path)
		audio_length = len(mix_long)


		#perform separation over the whole audio using a sliding window approach
		sliding_window_start = 0
		overlap_count = np.zeros((audio_length))
		sep_audio1 = np.zeros((audio_length))
		sep_audio2 = np.zeros((audio_length))
		sep_audio_list = [sep_audio1, sep_audio2]
		den_audio1 = np.zeros((audio_length))
		den_audio2 = np.zeros((audio_length))
		den_audio_list = [den_audio1, den_audio2]
		avged_sep_audio1 = np.zeros((audio_length))
		avged_sep_audio2 = np.zeros((audio_length))
		avged_den_audio1 = np.zeros((audio_length))
		avged_den_audio2 = np.zeros((audio_length))

		samples_per_window = int(2.04 * sr)
		while sliding_window_start + samples_per_window < audio_length:
			sliding_window_end = sliding_window_start + samples_per_window
			gt_list = [aud[sliding_window_start:sliding_window_end] for aud in gt_list_long]
			mix = mix_long[sliding_window_start:sliding_window_end]
			visualFeatures = prep_video(video1_path, video2_path, sliding_window_start)
			if visualFeatures is None:
				continue

			for i, visfeat in enumerate(visualFeatures):
				x = gt_list[i].squeeze()
				y = torch.Tensor(np.expand_dims(mix, 0)).cuda()

				x_spec = model._stft(torch.from_numpy(x))
				x_hat_spec, y_spec, y_den_spec, T_orig, norm_factor = model.enhance(y, context = visfeat, return_stft=True, corrector=args.corrector, corrector_steps=args.corrector_steps, snr=args.snr, N=args.N)
				
				x_hat = model.to_audio(x_hat_spec, T_orig)
				x_hat = x_hat * norm_factor
				x_hat = x_hat.squeeze()
				y_den = model.to_audio(y_den_spec, T_orig)
				y_den = y_den * norm_factor
				y_den = y_den.squeeze()
				
				if x.ndim == 1:
					x_hat = x_hat.cpu().numpy()
					y_den = y_den.cpu().numpy()
				else:
					x_hat = x_hat[0].cpu().numpy()
					y_den = y_den[0].cpu().numpy()
				sep_audio_list[i][sliding_window_start:sliding_window_end] += x_hat
				den_audio_list[i][sliding_window_start:sliding_window_end] += y_den
			overlap_count[sliding_window_start:sliding_window_end] = overlap_count[sliding_window_start:sliding_window_end] + 1
			sliding_window_start = sliding_window_start + int(args.hop_length * sr)
		
		# deal with the last segment
		pad_amount = sliding_window_start + samples_per_window - audio_length
		last_seg_len = audio_length - sliding_window_start
		if int(last_seg_len/16000*25) >= 1:
			gt_list = []
			for aud in gt_list_long:
				gt_cut = aud[sliding_window_start:]
				gt_padded = np.pad(gt_cut, (0,pad_amount), 'wrap')
				gt_list.append(gt_padded)
			mix_cut = mix_long[sliding_window_start:]
			mix = np.pad(mix_cut, (0,pad_amount), 'wrap')
			visualFeatures = prep_video(video1_path, video2_path, sliding_window_start)
			if visualFeatures is None:
				continue

			for i, visfeat in enumerate(visualFeatures):
				x = gt_list[i].squeeze()
				y = torch.Tensor(np.expand_dims(mix, 0)).cuda()

				x_spec = model._stft(torch.from_numpy(x))
				x_hat_spec, y_spec, y_den_spec, T_orig, norm_factor = model.enhance(y, context = visfeat, return_stft=True, corrector=args.corrector, corrector_steps=args.corrector_steps, snr=args.snr, N=args.N)
				
				x_hat = model.to_audio(x_hat_spec, T_orig)
				x_hat = x_hat * norm_factor
				x_hat = x_hat.squeeze()
				y_den = model.to_audio(y_den_spec, T_orig)
				y_den = y_den * norm_factor
				y_den = y_den.squeeze()
				
				if x.ndim == 1:
					x_hat = x_hat.cpu().numpy()
					y_den = y_den.cpu().numpy()
				else:
					x_hat = x_hat[0].cpu().numpy()
					y_den = y_den[0].cpu().numpy()
				sep_audio_list[i][sliding_window_start:] += x_hat[:last_seg_len]
				den_audio_list[i][sliding_window_start:] += y_den[:last_seg_len]
			overlap_count[sliding_window_start:] = overlap_count[sliding_window_start:] + 1
		else:
			overlap_count[sliding_window_start:] = 1

		avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio_list[0], overlap_count))
		avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio_list[1], overlap_count))
		avged_den_audio1 = avged_den_audio1 + clip_audio(np.divide(den_audio_list[0], overlap_count))
		avged_den_audio2 = avged_den_audio2 + clip_audio(np.divide(den_audio_list[1], overlap_count))
		pred_list =[avged_sep_audio1, avged_sep_audio2]
		den_audio_list = [avged_den_audio1, avged_den_audio2]

		if args.audio_save_root != '':
			save_audio(pred_list, den_audio_list, args.audio_save_root)

		# calculate metric for full audio
		_pesq, _si_sdr, _estoi, _stoi = 0., 0., 0., 0.
		for x, x_hat, y_den in zip(gt_list_long, pred_list, den_audio_list):
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