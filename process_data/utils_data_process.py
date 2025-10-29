# ---------------------------------------------------------------------------
# FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning
# https://arxiv.org/abs/2502.17432
# Copyright (c) 2025 Jason Jingzhou Liu and Yulong Li

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------------


import cv2
import numpy as np
from robobuf.buffers import ObsWrapper, Transition, ReplayBuffer

def gaussian_norm(list_of_array):
    data_array = np.concatenate(list_of_array, axis=0)
    
    print('Using in-place gaussian norm')
    mean = np.mean(data_array, axis=0)
    std = np.std(data_array, axis=0)
    if not std.all():  # handle situation with all 0 actions
        std[std == 0] = 1e-17

    for array in list_of_array:
        array -= mean
        array /= std
    normalization_stats = dict(
        mean=[float(x) for x in mean],
        std=[float(x) for x in std]
    )
    return normalization_stats

def generate_robobuf(trajectories):
    DOWNSAMPLE_RATE = 2
    buffer = ReplayBuffer()
    for traj in trajectories:
        num_steps = traj['num_steps']
        actions = traj['actions']
        states = traj['states']
        for i in range(0, num_steps, DOWNSAMPLE_RATE):
            obs = {
                'state': states[i],
            }
            for k, v in traj.items():
                if k.startswith('enc_cam_'):
                    obs[k] = v[i]
            transition = Transition(
                obs = ObsWrapper(obs), 
                action = actions[i], 
                reward = (i==num_steps-1), 
            )
            buffer.add(transition, is_first = (i < DOWNSAMPLE_RATE))

    return buffer

def get_diff_timestamps(timestamps):
    timestamps = np.array(timestamps)
    diff_timestamps = np.diff(timestamps)
    return diff_timestamps * 1e-9

def sync_data_slowest(traj_data, all_topics):
    timestamps = traj_data["timestamps"]
    for topic in all_topics:
        assert topic in timestamps, f"Topic {topic} not found in recorded data"
        timestamps[topic] = np.array(timestamps[topic])
    data = traj_data["data"]
    synced_data = {topic: [] for topic in all_topics}
    message_counts = [len(timestamps[topic]) for topic in all_topics]
    min_freq_topic = all_topics[int(np.argmin(message_counts))]
    
    timestamp_diffs = get_diff_timestamps(timestamps[min_freq_topic])
    avg_freq = 1/np.mean(timestamp_diffs)
    
    for i, target_ts in enumerate(timestamps[min_freq_topic]):
        for topic in all_topics:
            if topic == min_freq_topic:
                synced_data[topic].append(data[topic][i])
            else:
                closest_idx = np.argmin(np.abs(timestamps[topic] - target_ts))
                synced_data[topic].append(data[topic][closest_idx])
                
    return synced_data, avg_freq
    
def process_decoded_image(img):
    img = cv2.resize(img, (256, 256))
    return img
        
def process_image(img_dict):
    # img_dict は {"width": 640, "height": 480, "encoding": "rgb8", "data": ...} 
    # のような辞書であると仮定します
    
    # 1. 辞書からメタデータと生のピクセルデータを抽出
    try:
        h = img_dict['height'] # 480
        w = img_dict['width']  # 640
        # 実際のピクセルデータが 'data' キーに格納されていると仮定
        raw_data = img_dict['data'] 
    except KeyError as e:
        print(f"エラー: 画像辞書に必要なキー ({e}) が見つかりません。辞書のキー: {img_dict.keys()}")
        raise e
    except TypeError:
         print(f"エラー: 画像データが辞書ではありません。受け取ったデータ型: {type(img_dict)}")
         raise

    # 2. データを numpy 配列に変換
    if isinstance(raw_data, bytes):
        decoded_image = np.frombuffer(raw_data, dtype=np.uint8)
    else:
        # すでに numpy 配列などの場合は、それをuint8型に変換
        decoded_image = np.array(raw_data, dtype=np.uint8) 

    # 3. データを (Height, Width, Channels) の形状に変更
    try:
        # チャンネル数 (3 for rgb8)
        channels = 3 
        decoded_image = decoded_image.reshape((h, w, channels))
    except Exception as e:
        print(f"画像のReshapeに失敗しました。期待するShape: {(h, w, 3)}, 実際のデータサイズ: {decoded_image.size}, エラー: {e}")
        raise e

    # 4. OpenCVはBGR形式を標準とするため、RGB (rgb8) からBGRに変換
    decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_RGB2BGR)

    # 5. process_decoded_image でリサイズ (256x256)
    decoded_image = process_decoded_image(decoded_image)
    
    # 6. JPEG形式（品質50）に圧縮（エンコード）して保存
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    _, compressed_image = cv2.imencode('.jpg', decoded_image, encode_param)
    
    return compressed_image

# def process_image(img_enc):
#     # decode, process, and encode
#     decoded_image = cv2.imdecode(img_enc, cv2.IMREAD_COLOR)    
#     decoded_image = process_decoded_image(decoded_image)
#     encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
#     _, compressed_image = cv2.imencode('.jpg', decoded_image, encode_param)
#     return compressed_image