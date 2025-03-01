import subprocess
import os
import re

def synthesize_with_sinsy_cli(input_xml_path, output_wav_path, language="japanese", voice_bank=1, vibrato=1.0, pitch=0):
    """
    使用 sinsy-cli 合成歌声并自动处理输出文件。

    Args:
        input_xml_path (str): 输入的带有歌词的 MusicXML 文件路径。
        output_wav_path (str): 输出的 WAV 文件路径。
        language (str): 使用的语言 ("japanese", "english", "mandarin")，默认为 "japanese"。
        voice_bank (int): 使用的声库编号，从 0 开始，默认为 1。
        vibrato (float): 颤音强度 (范围：0 到 2)，默认值为 1.0。
        pitch (int): 音高调整（半音，范围 -24 到 24），默认值为 0。

    Returns:
        None
    """
    # 检查输入文件是否存在
    if not os.path.exists(input_xml_path):
        raise FileNotFoundError(f"输入的 MusicXML 文件不存在: {input_xml_path}")

    # 构建 sinsy-cli 命令
    command = [
        "sinsy-cli",
        "-l", language,          # 语言
        "-b", str(voice_bank),   # 声库编号
        "-v", str(vibrato),      # 颤音强度
        "-p", str(pitch),        # 音高调整
        input_xml_path           # 输入的 MusicXML 文件路径
    ]

    try:
        # 执行命令
        print(f"正在调用 Sinsy 合成歌声，命令: {' '.join(command)}")
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # 打印 sinsy-cli 的输出
        print(f"Sinsy 输出:\n{result.stdout}")
        print(f"Sinsy 错误输出:\n{result.stderr}")

        # 从 Sinsy 输出中解析生成的文件路径
        match = re.search(r"Saving .* to (.+\.wav)", result.stderr)
        if match:
            generated_file = match.group(1).strip()
            print(f"找到生成的文件: {generated_file}")

            # 检查生成的文件是否存在并重命名为目标路径
            if os.path.exists(generated_file):
                os.rename(generated_file, output_wav_path)
                print(f"歌声合成成功，已保存到: {output_wav_path}")
            else:
                print("歌声合成失败，生成的文件不存在。")
        else:
            print("无法解析 Sinsy 输出中的文件路径。")

    except subprocess.CalledProcessError as e:
        print(f"Sinsy CLI 调用失败，错误信息: {e.stderr}")

# 示例调用
if __name__ == "__main__":
    input_xml = "E:/output_with_lyrics.xml"  # 带有歌词的 MusicXML 文件路径
    output_wav = "E:/output_song.wav"       # 目标输出的 WAV 文件路径

    synthesize_with_sinsy_cli(
        input_xml_path=input_xml,
        output_wav_path=output_wav,
        language="japanese",  # 根据文件内容选择合适的语言
        voice_bank=1,         # 选择默认声库
        vibrato=1.2,          # 设置颤音强度
        pitch=2               # 调高音高2个半音
    )
