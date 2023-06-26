import torch as th
import math
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import pandas as pd
from video_loader import VideoLoader
from torch.utils.data import DataLoader
from preprocessing import Preprocessing
from random_sequence_shuffler import RandomSequenceSampler
import os
# from args import MODEL_DIR
import clip


scenic_zeroshot_eval_folder_path = "" #add
video_name = "" #add
feature_name = "" #add
extract_feature_csv_path = "" #add
video_path = os.path.join("video", video_name)
feature_path = os.path.join("feature", feature_name)
df = pd.DataFrame([[os.path.join(scenic_zeroshot_eval_folder_path, video_path), os.path.join(scenic_zeroshot_eval_folder_path, feature_path)]], columns = ["video_path", "feature_path"])
df.to_csv(extract_feature_csv_path, index=None)

parser = argparse.ArgumentParser(description="Easy video feature extractor")

parser.add_argument(
    "--csv",
    type=str,
    help="input csv with columns video_path (input video) and feature_path (output path to feature)",
)
parser.add_argument(
    "--batch_size", type=int, default=128, help="batch size for extraction"
)
parser.add_argument(
    "--half_precision",
    type=int,
    default=1,
    help="whether to output half precision float or not",
)
parser.add_argument(
    "--num_decoding_thread",
    type=int,
    default=3,
    help="number of parallel threads for video decoding",
)
parser.add_argument(
    "--l2_normalize",
    type=int,
    default=0,
    help="whether to l2 normalize the output feature",
)
parser.add_argument(
    "--feature_dim", type=int, default=768, help="output video feature dimension"
)
args = parser.parse_args()

dataset = VideoLoader(
    args.csv,
    framerate=1,  # one feature per second max
    size=224,
    centercrop=True,
)
n_dataset = len(dataset)
print(n_dataset)
sampler = RandomSequenceSampler(n_dataset, 10)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=args.num_decoding_thread,
    sampler=sampler if n_dataset > 10 else None,
)
print(len(loader))
preprocess = Preprocessing()
model, _ = clip.load("ViT-L/14") #, download_root=MODEL_DIR

#load fine-tuned clip
# checkpoint = th.load("/home/hlpark/shared/REDUCE/clip/finetune/model_checkpoint/best_model.pt")

# Use these 3 lines if you use default model setting(not training setting) of the clip. For example, if you set context_length to 100 since your string is very long during training, then assign 100 to checkpoint['model_state_dict']["context_length"] 
# checkpoint['model_state_dict']["input_resolution"] = model.input_resolution #default is 224
# checkpoint['model_state_dict']["context_length"] = model.context_length # default is 77
# checkpoint['model_state_dict']["vocab_size"] = model.vocab_size 

# model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
model = model.cuda()

with th.no_grad():
    
    for k, data in enumerate(loader):
        input_file = data["input"][0]
        output_file = data["output"][0]
        if len(data["video"].shape) > 3:
            print(
                "Computing features of video {}/{}: {}".format(
                    k + 1, n_dataset, input_file
                )
            )
            video = data["video"].squeeze()
            if len(video.shape) == 4:
                video = preprocess(video)
                n_chunk = len(video)
                features = th.cuda.FloatTensor(n_chunk, args.feature_dim).fill_(0)
                n_iter = int(math.ceil(n_chunk / float(args.batch_size)))
                print(n_chunk, n_iter)
                for i in tqdm(range(n_iter)):
                    min_ind = i * args.batch_size
                    max_ind = (i + 1) * args.batch_size
                    video_batch = video[min_ind:max_ind].cuda()
                    #print(video_batch)
                    batch_features = model.encode_image(video_batch)
                    if args.l2_normalize:
                        batch_features = F.normalize(batch_features, dim=1)
                    features[min_ind:max_ind] = batch_features
                features = features.cpu().numpy()
                if args.half_precision:
                    features = features.astype("float16")
                np.save(output_file, features)
        else:
            print("Video {} already processed.".format(input_file))
