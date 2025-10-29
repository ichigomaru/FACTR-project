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


import yaml
import hydra
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from omegaconf import DictConfig

from utils_data_process import sync_data_slowest, process_image, gaussian_norm, generate_robobuf

@hydra.main(version_base=None, config_path="cfg", config_name="default")
def main(cfg: DictConfig):

    input_path = cfg.input_path
    output_path = cfg.output_path
    
    rgb_obs_topics = list(cfg.cameras_topics)
    state_obs_topics = list(cfg.obs_topics)
    action_config = dict(cfg.action_config)
    action_topics = list(action_config.keys())
    
    assert len(state_obs_topics) > 0, "Require low-dim observation topics"
    assert len(rgb_obs_topics) > 0, "Require visual observation topics"
    assert len(action_topics) > 0, "Require visual observation topics"
    
    
    data_folder = Path(input_path)
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # initialize topics
    all_topics = state_obs_topics + rgb_obs_topics + action_topics
    
    all_episodes = sorted([f for f in data_folder.iterdir() if f.name.startswith('data_log_') and f.name.endswith('.pkl')]) #ep_ -> data_log_に変更
    
    trajectories = []
    all_states = []
    all_actions = []
    pbar = tqdm(all_episodes)
    for episode_pkl in pbar:
        # === 1. 破損ファイル(EOFError)のチェック ===
        try:
            with open(episode_pkl, 'rb') as f:
                traj_data = pickle.load(f)
        except EOFError:
            print(f"\n[エラー] ファイル '{episode_pkl.name}' は空か破損しています (EOFError)。スキップします。")
            continue  # 次のファイルへ
        except Exception as e:
            print(f"\n[エラー] ファイル '{episode_pkl.name}' の読み込みに失敗: {e}。スキップします。")
            continue  # 次のファイルへ

        # === 2. 中身が空(データ数<=1)でないかチェック ===
        try:
            timestamps = traj_data.get("timestamps")
            if timestamps is None:
                print(f"\n[警告] ファイル '{episode_pkl.name}' に 'timestamps' キーがありません。スキップします。")
                continue

            message_counts = [len(timestamps[topic]) for topic in all_topics if topic in timestamps]
            if not message_counts:
                print(f"\n[警告] ファイル '{episode_pkl.name}' の 'timestamps' にデータがありません。スキップします。")
                continue

            min_freq_topic_len = min(message_counts)
            if min_freq_topic_len <= 1:
                print(f"\n[警告] ファイル '{episode_pkl.name}' のデータが {min_freq_topic_len} 件しかありません。スキップします。")
                continue
        except Exception as e:
            print(f"\n[エラー] ファイル '{episode_pkl.name}' のデータ検証中に失敗: {e}。スキップします。")
            continue

        # === 3. (ここに来たファイルは正常) データ処理を実行 ===
        traj_data, avg_freq = sync_data_slowest(traj_data, all_topics)
        pbar.set_postfix({'avg_freq': f'{avg_freq:.1f} Hz'})

        traj = {}
        num_steps = len(traj_data[action_topics[0]])
        traj['num_steps'] = num_steps
    # === 1. 状態 (State) の抽出 ===
        state_list = []
        for topic in state_obs_topics:
            try:
                topic_data_list = traj_data[topic] # 同期済みの辞書のリスト
                
                # (修正点) トピック名でチェックせず、'effort' の抽出を試みる
                extracted_data = np.array([item['effort'] for item in topic_data_list])
                state_list.append(extracted_data)
            
            except (TypeError, KeyError):
                # 'effort' がない辞書か、辞書でない場合 (例: グリッパー位置が [0.1] のような数値だった場合)
                print(f"\n[情報] トピック '{topic}' は 'effort' を持たないか辞書ではありません。数値としてそのまま使用します。")
                state_list.append(np.array(topic_data_list))
            except Exception as e:
                print(f"\n[警告] ファイル '{episode_pkl.name}' のトピック '{topic}' 処理中にエラー: {e}。スキップします。")
                continue # このトピックの処理をスキップ

        if not state_list:
            print(f"\n[警告] ファイル '{episode_pkl.name}' で state_obs_topics が処理されませんでした。スキップします。")
            continue

        traj['states'] = np.concatenate(state_list, axis=-1)

        action_list = []  # <-- ★★★ ここで action_list を定義します ★★★
        for topic in action_topics:
            try:
                topic_data_list = traj_data[topic] # 同期済みの辞書のリスト

                # (修正点) トピック名でチェックせず、'position' の抽出を試みる
                actions = np.array([item['position'] for item in topic_data_list])
            
            except (TypeError, KeyError):
                # 'position' がない辞書か、辞書でない場合 (例: グリッパー指令が [0.8] のような数値だった場合)
                print(f"\n[情報] トピック '{topic}' は 'position' を持たないか辞書ではありません。数値としてそのまま使用します。")
                actions = np.array(topic_data_list)
            except Exception as e:
                print(f"\n[警告] ファイル '{episode_pkl.name}' のトピック '{topic}' 処理中にエラー: {e}。スキップします。")
                continue # このトピックの処理をスキップ
                
            action_list.append(actions)

        if not action_list:
            print(f"\n[警告] ファイル '{episode_pkl.name}' で action_config が処理されませんでした。スキップします。")
            continue

        traj['actions'] = np.concatenate(action_list, axis=-1)

        all_states.append(traj['states'])
        all_actions.append(traj["actions"])

        for cam_ind, topic in enumerate(rgb_obs_topics):
            enc_images = traj_data[topic]
            processed_images = [process_image(img_enc) for img_enc in enc_images]
            traj[f'enc_cam_{cam_ind}'] = processed_images
        trajectories.append(traj)
        
    # normalize states and actions
    state_norm_stats = gaussian_norm(all_states)
    action_norm_stats = gaussian_norm(all_actions)
    norm_stats = dict(state=state_norm_stats, action=action_norm_stats)
    
    # dump data buffer
    buffer_name = "buf"
    buffer = generate_robobuf(trajectories)
    with open(output_dir / f"{buffer_name}.pkl", "wb") as f:
        pickle.dump(buffer.to_traj_list(), f)
    
    # dump rollout config
    obs_config = {
        'state_topics': state_obs_topics,
        'camera_topics': rgb_obs_topics,
    }
    rollout_config = {
        'obs_config': obs_config,
        'action_config': action_config,
        'norm_stats': norm_stats
    }
    with open(output_dir / "rollout_config.yaml", "w") as f:
        yaml.dump(rollout_config, f, sort_keys=False)
        
if __name__ == "__main__":
    main()