import numpy as np
import pandas as pd
import json
import random


def load_json(file):
    with open(file) as json_file:
        json_data = json.load(json_file)
        return json_data


anno_df = pd.read_csv("./data/activitynet_annotations/video_info_new.csv")
anno_database = load_json("./data/activitynet_annotations/anet_anno_action.json")
subset = 'training'
training_video = []
action_dict = {}
action_dict_num = {}
# get all training video names
for i in range(len(anno_df)):
    video_name = anno_df.video.values[i]
    video_info = anno_database[video_name]
    video_subset = anno_df.subset.values[i]
    if subset in video_subset:
        training_video.append(video_name)
        label = video_info["annotations"][0]['label']
        if label not in action_dict:
            action_dict[label] = [video_name]
        else:
            action_dict[label].append(video_name)

for label_name in action_dict:
    action_dict_num[label_name] = len(action_dict[label_name])

opt_file = open("./data/activitynet_annotations/per_label_num.json", "w")
json.dump(action_dict_num, opt_file)
opt_file.close()

# unlabel percents
label_percent = np.linspace(0.1, 0.9, 9)
# unlabeled_video = []
for percent in label_percent:
    unlabeled_video = []
    new_props = []
    for label_name in action_dict:
        unlabeled_video.extend(random.sample(action_dict[label_name], round(percent*len(action_dict[label_name]))))
    for i in range(len(anno_df)):
        video_name = anno_df.video.values[i]
        numFrame = anno_df.numFrame.values[i]
        seconds = anno_df.seconds.values[i]
        fps = anno_df.fps.values[i]
        rfps = anno_df.rfps.values[i]
        featureFrame = anno_df.featureFrame.values[i]
        video_info = anno_database[video_name]
        video_subset = anno_df.subset.values[i]
        if video_name in unlabeled_video:
            new_props.append([video_name, numFrame, seconds, fps, rfps, 'training_unlabel', featureFrame])
        else:
            new_props.append([video_name, numFrame, seconds, fps, rfps, video_subset, featureFrame])
    new_props = np.stack(new_props)

    col_name = ["video", "numFrame", "seconds", "fps", "rfps", "subset", "featureFrame"]
    new_df = pd.DataFrame(new_props, columns=col_name)
    new_df.to_csv("./data/activitynet_annotations/video_info_new_{}.csv".format(round(percent, 1)), index=False)