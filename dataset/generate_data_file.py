import numpy as np
import cv2
import argparse
import yaml
import pandas as pd
from tqdm import tqdm
from utils.flow_utils import writeFlowFile, readFlowFile, calc_flow

from .datasets import load_emotions_file


def generate_datafile(data_dir_F, data_dir_NF, dataset_name, emotions_file_path, save_dir_F, save_dir_NF, flow_algo, step):

	emotions_df = load_emotions_file(emotions_file_path)

	data_file = []
	for sub_id, seq_id, label in tqdm(emotions_df.values, desc=f'generating data file ... '):
		
		## frontal & non-frontal frames
		F_frames = sorted(os.listdir(os.path.join(data_dir_F, sub_id, seq_id)))
		NF_frames = sorted(os.listdir(os.path.join(data_dir_NF, sub_id, seq_id)))

		for i in range(len(F_frames)-step):
			current_F_frame, current_NF_frame = F_frames[i], NF_frames[i]
			next_F_frame, next_NF_frame = F_frames[i+step], NF_frames[i+step]

			fname_current_frame_F = os.path.join(data_dir_F, sub_id, seq_id, current_F_frame)
			fname_next_frame_F = os.path.join(data_dir_F, sub_id, seq_id, next_F_frame)
			fname_current_frame_NF = os.path.join(data_dir_NF, sub_id, seq_id, current_NF_frame)
			fname_next_frame_NF = os.path.join(data_dir_NF, sub_id, seq_id, next_NF_frame)

			info = [dataset_name, sub_id, seq_id, str(label), flow_algo,
					fname_current_frame_F, fname_next_frame_F,
					fname_current_frame_NF, fname_next_frame_NF,
					str(step)]

			fname_flow_F = '_'.join([dataset_name, 
				              		 sub_id, 
				              		 seq_id, 
				              		 str(label), 
									 flow_algo, 
									 current_F_frame.replace('.png', ''), 
									 next_F_frame.replace('.png', ''), 
									 str(step)])

			fname_flow_NF = '_'.join([dataset_name, 
				  					  sub_id, 
				  					  seq_id, 
				  					  str(label), 
				  					  flow_algo, 
				  					  current_NF_frame.replace('.png', ''), 
				  					  next_NF_frame.replace('.png', ''), 
				  					  str(step)])

			data_file.append(','.join(info + [os.path.join(save_dir_NF, fname_flow_NF + '.flo')] +\
											 [os.path.join(save_dir_NF, fname_flow_INF + '.flo')]+\
											 [os.path.join(save_dir_F, fname_flow_F + '.flo')]))
			
			## calculate flows
			frame_F_1 = cv2.imread(os.path.join(data_dir_F, sub_id, seq_id, current_F_frame), 0)
			frame_F_2 = cv2.imread(os.path.join(data_dir_F, sub_id, seq_id, next_F_frame), 0) 
			
			frame_NF_1 = cv2.imread(os.path.join(data_dir_NF, sub_id, seq_id, current_NF_frame), 0)
			frame_NF_2 = cv2.imread(os.path.join(data_dir_NF, sub_id, seq_id, next_NF_frame), 0)
			
			flow_F = calc_flow(frame_F_1, frame_F_2, flow_algo=flow_algo)
			flow_NF = calc_flow(frame_NF_1, frame_NF_2, flow_algo=flow_algo)		

			writeFlowFile(flow_F, os.path.join(save_dir_F, fname_flow_F + '.flo'))
			writeFlowFile(flow_NF, os.path.join(save_dir_NF, fname_flow_NF + '.flo'))

	text = '\n'.join(data_file) + '\n'
	parent_dir = os.path.abspath(os.path.join(save_dir_F, os.pardir))
	with open(os.path.join(parent_dir, 'data_file.txt'), 'a') as f:
			  f.write(text)
	print(f'[DONE] {len(data_file)} files saved !')


if __name__ == '__main__':

	## settings & configuration
	parser = argparse.ArgumentParser(description='Create Optical Flow Datasets')
	parser.add_argument('--dataset_name', type=str, help='name of dataset')
	parser.add_argument('--data_dir_F', type=str, help='directory to frontal dataset')
	parser.add_argument('--data_dir_NF', type=str, help='directory to non frontal dataset')
	parser.add_argument('--save_dir_F', type=str, help='directory to save frontal flow dataset to')
	parser.add_argument('--save_dir_NF', type=str, help='directory to save non frontal flow dataset to')
	parser.add_argument('--emotions_file_path', type=str, help='emotions csv file path')
	parser.add_argument('--flow_algo', type=str, help='the algorithm to be used to calculate motion.')
	parser.add_argument('--step', type=str, help='time step')

	## make save directory
	if not os.path.exists(args.save_dir_F):
		os.mkdir(args.save_dir_F)
	if not os.path.exists(args.save_dir_NF):
		os.mkdir(args.save_dir_NF)

	## generate data file
	for step in range(1, 11):
		generate_datafile(args.data_dir_F, args.data_dir_NF, args.dataset_name, args.emotions_file, args.save_dir_F, args.save_dir_NF, args.flow_algo, step)
