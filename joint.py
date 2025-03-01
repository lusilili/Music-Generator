from pydub import AudioSegment

def mix_audio(vocal_path, accompaniment_path, output_mix_path, vocal_gain=0, accompaniment_gain=0):
    """
    混合歌声和伴奏音频。
    Args:
        vocal_path (str): 歌声 WAV 文件路径。
        accompaniment_path (str): 伴奏 WAV 文件路径。
        output_mix_path (str): 输出混合音频文件路径。
        vocal_gain (float): 歌声音量调整（单位：dB）。
        accompaniment_gain (float): 伴奏音量调整（单位：dB）。
    """
    # 加载音频文件
    vocal = AudioSegment.from_file(vocal_path)
    accompaniment = AudioSegment.from_file(accompaniment_path)

    # 调整音量
    vocal = vocal + vocal_gain
    accompaniment = accompaniment + accompaniment_gain

    # 混合音频
    mixed = vocal.overlay(accompaniment)

    # 导出混合后的音频
    mixed.export(output_mix_path, format="wav")
    print(f"混合音频已保存到：{output_mix_path}")

# 示例
vocal_path = "E:/output_song.wav"  # 歌声文件路径
accompaniment_path = "E:/output5.wav"  # 伴奏文件路径
output_mix_path = "E:/output_vocal_mix.wav"  # 输出混合音频文件路径
mix_audio(vocal_path, accompaniment_path, output_mix_path, vocal_gain=5, accompaniment_gain=-5)
