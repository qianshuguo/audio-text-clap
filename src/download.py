import pandas as pd
import os
import subprocess

# =========================
# 1. 路径设置
# =========================
base_dir = os.path.dirname(os.path.abspath(__file__))   # 当前 download.py 所在目录
project_dir = os.path.dirname(base_dir)                 # 上一级，也就是 FinalGroupWork

input_csv = os.path.join(project_dir, "data", "raw_csv", "test.csv")
audio_dir = os.path.join(project_dir, "data", "audio")
success_csv = os.path.join(project_dir, "data", "metadata_500.csv")
failed_csv = os.path.join(project_dir, "data", "failed.csv")
# =========================
# 2. 参数
# =========================
target_count = 500
clip_duration = 10  # 每条截取10秒
temp_file = "temp.wav"

# =========================
# 3. 读原始CSV
# =========================
df = pd.read_csv(input_csv)

# 创建音频输出文件夹
os.makedirs(audio_dir, exist_ok=True)

# =========================
# 4. 初始化计数和记录
# =========================
saved_count = 0
saved_rows = []
failed_rows = []

# =========================
# 5. 开始遍历
# =========================
for original_index, row in df.iterrows():
    if saved_count >= target_count:
        break

    youtube_id = row["youtube_id"]
    start_time = row["start_time"]

    url = f"https://www.youtube.com/watch?v={youtube_id}"
    output_filename = f"{saved_count}.wav"
    output_file = os.path.join(audio_dir, output_filename)

    print("\n" + "=" * 60)
    print(f"Trying sample #{saved_count}")
    print(f"Original row index: {original_index}")
    print(f"URL: {url}")

    try:
        # 删除旧 temp.wav
        if os.path.exists(temp_file):
            os.remove(temp_file)

        # -------------------------
        # STEP 1: 下载音频
        # -------------------------
        result1 = subprocess.run(
            [
                "yt-dlp",
                "-x",
                "--audio-format", "wav",
                "-o", temp_file,
                url
            ],
            capture_output=True,
            text=True
        )

        print("yt-dlp stdout:\n", result1.stdout)
        print("yt-dlp stderr:\n", result1.stderr)

        if result1.returncode != 0:
            print(f"Skipped: yt-dlp failed for {youtube_id}")

            failed_record = {
                "original_row_index": original_index,
                "youtube_id": youtube_id,
                "start_time": start_time,
                "url": url,
                "reason": "yt-dlp failed"
            }

            for col in df.columns:
                if col not in failed_record:
                    failed_record[col] = row[col]

            failed_rows.append(failed_record)
            continue

        if not os.path.exists(temp_file):
            print(f"Skipped: temp.wav was not created for {youtube_id}")

            failed_record = {
                "original_row_index": original_index,
                "youtube_id": youtube_id,
                "start_time": start_time,
                "url": url,
                "reason": "temp file missing"
            }

            for col in df.columns:
                if col not in failed_record:
                    failed_record[col] = row[col]

            failed_rows.append(failed_record)
            continue

        # -------------------------
        # STEP 2: 裁剪音频
        # -------------------------
        result2 = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-ss", str(start_time),
                "-t", str(clip_duration),
                "-i", temp_file,
                output_file
            ],
            capture_output=True,
            text=True
        )

        print("ffmpeg stdout:\n", result2.stdout)
        print("ffmpeg stderr:\n", result2.stderr)

        if result2.returncode != 0:
            print(f"Skipped: ffmpeg failed for {youtube_id}")

            if os.path.exists(output_file):
                os.remove(output_file)

            failed_record = {
                "original_row_index": original_index,
                "youtube_id": youtube_id,
                "start_time": start_time,
                "url": url,
                "reason": "ffmpeg failed"
            }

            for col in df.columns:
                if col not in failed_record:
                    failed_record[col] = row[col]

            failed_rows.append(failed_record)
            continue

        if not os.path.exists(output_file):
            print(f"Skipped: output file was not created for {youtube_id}")

            failed_record = {
                "original_row_index": original_index,
                "youtube_id": youtube_id,
                "start_time": start_time,
                "url": url,
                "reason": "output file missing"
            }

            for col in df.columns:
                if col not in failed_record:
                    failed_record[col] = row[col]

            failed_rows.append(failed_record)
            continue

        # -------------------------
        # STEP 3: 成功后记录 metadata
        # -------------------------
        print(f"Saved: {output_file}")

        success_record = {
            "saved_id": saved_count,
            "audio_filename": output_filename,
            "audio_path": output_file,
            "original_row_index": original_index,
            "youtube_id": youtube_id,
            "start_time": start_time,
            "url": url
        }

        # 保留原表全部字段
        for col in df.columns:
            if col not in success_record:
                success_record[col] = row[col]

        saved_rows.append(success_record)
        saved_count += 1

    except Exception as e:
        print(f"Exception for {youtube_id}: {e}")

        failed_record = {
            "original_row_index": original_index,
            "youtube_id": youtube_id,
            "start_time": start_time,
            "url": url,
            "reason": f"exception: {str(e)}"
        }

        for col in df.columns:
            if col not in failed_record:
                failed_record[col] = row[col]

        failed_rows.append(failed_record)

    finally:
        # 清理 temp.wav
        if os.path.exists(temp_file):
            os.remove(temp_file)

# =========================
# 6. 保存成功和失败记录
# =========================
metadata_df = pd.DataFrame(saved_rows)
metadata_df.to_csv(success_csv, index=False)

failed_df = pd.DataFrame(failed_rows)
failed_df.to_csv(failed_csv, index=False)

# =========================
# 7. 输出总结
# =========================
print("\n" + "=" * 60)
print("Finished.")
print(f"Target count: {target_count}")
print(f"Successfully saved: {saved_count}")
print(f"Failed: {len(failed_rows)}")
print(f"Success metadata saved to: {success_csv}")
print(f"Failed metadata saved to: {failed_csv}")

if saved_count < target_count:
    print("Warning: Not enough valid audio clips were found in the CSV to reach the target count.")