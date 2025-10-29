import pickle as pkl
import sys
from robobuf import ReplayBuffer as RB
from pathlib import Path

# -----------------------------------------------------------------
# ★ 1. ご自身の buf.pkl へのパスを指定してください
# (実行時のエラーログからパスを特定しました)
BUF_PATH = "processed_data/test_1028/buf.pkl" 
# (↑ もしこのパスが check_buffer.py の場所から見て正しくない場合は、
#   /home/otake/FACTR-pr/FACTR-project/processed_data/test_1028/buf.pkl 
#   のような絶対パスに書き換えてください)
# -----------------------------------------------------------------

buf_file = Path(BUF_PATH)

if not buf_file.exists():
    print(f"エラー: ファイルが見つかりません: {BUF_PATH}")
    print(f"スクリプトの実行場所: {Path.cwd()}")
    sys.exit(1)

print(f"'{BUF_PATH}' を読み込んでいます...")

try:
    with open(buf_file, "rb") as f:
        # process_data.py は .to_traj_list() で保存しています
        traj_list = pkl.load(f)

    # RobobufReplayBuffer のロード処理を模倣
    buf = RB.load_traj_list(traj_list)

    # --- (ここを修正しました) ---
    # ReplayBuffer オブジェクトの総ステップ数は len(buf) で取得します
    total_steps = len(buf)

    print("\n--- buf.pkl の内容 ---")
    print(f"全軌道の総ステップ数: {total_steps} ステップ")

    if total_steps > 0:
        print("\n--- 最初のステップのデータ情報 ---")
        # buf[0] で最初のステップ(Transition)にアクセスできます
        first_step = buf[0]
        print(f"  状態(State)の次元数: {first_step.obs.state.shape}")
        print(f"  行動(Action)の次元数: {first_step.action.shape}")
    # --- (修正ここまで) ---

    print("\n----------------------")
    print("元のデータ（50Hz）の総ステップ数の約半分になっていれば、")
    print("ダウンサンプリング（25Hz）は成功しています。")


except Exception as e:
    print(f"\nエラー: buf.pkl の読み込みに失敗しました。")
    print(f"詳細: {e}")