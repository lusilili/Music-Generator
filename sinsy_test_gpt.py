from music21 import converter, stream, note
from pydub import AudioSegment
import soundfile as sf
import pykakasi
import subprocess
import os
import re
from openai import OpenAI

# 模型文件路径
model_path = r"E:\midi-model-main\midi-model\sinsy-0.92\dic"
musescore_path = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"


def generate_lyrics(song_description):
    """
    根据用户输入的歌曲描述生成日文歌词。
    """

    client = OpenAI(base_url="https://api.gptsapi.net/v1", api_key="sk-acf9a52c5bf75229a834c9c9044f3fdfd163ee5767216kSr")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        store=True,
        messages=[
            {"role": "user", "content": "请根据以下描述生成60字左右日文歌词，停顿的地方用','表示，停顿多久就放多少个',',不要换行：\n{song_description}"}
        ]
    )

    # 获取生成的文本
    generated_text = completion.choices[0].message.content
    print(generated_text)

    # 保存到txt文件
    with open("my_song_lyrics.txt", "w", encoding="utf-8") as file:
        file.write(generated_text)

    return generated_text

# 工具函数
def ensure_file_exist(file_path, description = "文件"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{description}不存在：{file_path}")

def midi_to_musicxml(musescore_path, input_midi, output_musicxml):
    """
    使用 MuseScore 将 MIDI 文件转换为 MusicXML 文件。

    Args:
        musescore_path (str): MuseScore 可执行文件路径。
        input_midi (str): 输入的 MIDI 文件路径。
        output_musicxml (str): 输出的 MusicXML 文件路径。
    """
    ensure_file_exist(input_midi, "输入伴奏midi")
    try:
        subprocess.run(
            [musescore_path, "-o", output_musicxml, input_midi],
            check=True
        )
        print(f"成功将 {input_midi} 转换为 {output_musicxml}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")

def extract_main_melody(input_musicxml, output_musicxml):
    """
    从 MusicXML 文件中提取主旋律并保存为新的文件。

    Args:
        input_musicxml (str): 输入的 MusicXML 文件路径。
        output_musicxml (str): 输出的只包含主旋律的 MusicXML 文件路径。

    Returns:
        str: 输出的主旋律 MusicXML 文件路径。
    """
    ensure_file_exist(input_musicxml, "输入伴奏MusicXML")
    # !!优化策略一：选择音符较多的音轨作为主旋律
    # 解析输入文件
    score = converter.parse(input_musicxml)

    # 初始化变量以存储主旋律
    main_melody = None
    max_notes = 0

    # 遍历所有声部 (parts)
    for part in score.parts:
        # 统计声部中的音符数量
        notes_in_part = len([n for n in part.flat.notes if isinstance(n, note.Note)])

        # 如果当前声部的音符数量最多，则更新主旋律
        if notes_in_part > max_notes:
            max_notes = notes_in_part
            main_melody = part

    # 检查是否找到主旋律
    if main_melody is None:
        raise ValueError("未找到主旋律，可能输入文件不包含音符")

    # 创建一个新的乐谱，只包含主旋律
    new_score = stream.Score()
    new_score.append(main_melody)

    # 保存到新的 MusicXML 文件
    new_score.write('musicxml', output_musicxml)
    print(f"主旋律提取完成，结果保存到 {output_musicxml}")

    return output_musicxml

# 日文转片假名
def convert_to_kana(text):
    """
    将输入文本转换为日语假名。
    """
    kakasi = pykakasi.kakasi()
    kakasi.setMode("H", "H")  # 平假名保持为平假名
    kakasi.setMode("K", "H")  # 片假名转换为平假名
    kakasi.setMode("J", "H")  # 日本汉字转换为平假名
    converter = kakasi.getConverter()
    return converter.do(text)

def split_to_syllables(text):
    """
    将歌词切分为逐字假名。
    """
    kana_text = convert_to_kana("".join(text)) if isinstance(text, list) else convert_to_kana(text)
    return [char for char in kana_text if char.strip()]  # 排除空白字符


def fill_lyrics_to_musicxml(musicxml_path, lyrics, output_path):
    """
    将歌词填充到 MusicXML 文件中的音符上。

    Args:
        musicxml_path (str): 输入的 MusicXML 文件路径。
        lyrics_text (str): 输入的歌词字符串。
        output_path (str): 输出的带歌词的 MusicXML 文件路径。

    Returns:
        str: 填充歌词后的 MusicXML 文件路径。
    """
    # 加载 MusicXML 文件
    score = converter.parse(musicxml_path)

    # 切分歌词
    syllables = split_to_syllables(lyrics)
    print(f"歌词切分后：{syllables}")

    # 遍历音符
    notes = [n for n in score.flat.notes if isinstance(n, note.Note)]
    print(f"总共有 {len(notes)} 个音符")

    # 填充歌词到音符
    lyrics_index = 0
    for n in notes:
        if lyrics_index < len(syllables):
            n.addLyric(syllables[lyrics_index])  # 添加歌词
            lyrics_index += 1
        else:
            break  # 如果歌词不够，停止填充

    # 检查歌词是否分配完成
    if lyrics_index < len(syllables):
        print(f"警告：还有 {len(syllables) - lyrics_index} 个歌词未分配音符")
    else:
        print("歌词分配完成")

    # 保存带歌词的 MusicXML 文件
    score.write("musicxml", output_path)
    print(f"歌词填充完成，结果已保存到 {output_path}")
    return output_path


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
    ensure_file_exist(vocal_path, "vocal文件")
    ensure_file_exist(accompaniment_path, "伴奏文件")

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


if __name__ == "__main__":
    # 用户输入
    input = "描写与友人分别离愁的歌曲"
    lyrics = generate_lyrics(input)  # 替换为你的歌词
    kana_lyrics = split_to_syllables(lyrics)
    #
    input_midi = "E:/sample/output5.mid"  # 输入生成的伴奏midi
    intermediate_musicxml = "E:/output_to_xml.xml"  # 替换为你的 MusicXML 文件路径
    input_musicxml = intermediate_musicxml  # 替换为你的 MusicXML 文件路径
    output_musicxml = "E:/output_get_mainstream.xml"
    output_ii_musicxml = "E:/output_mainstream_with_lyrics.xml"
    output_wav = "E:/output_vocal.wav"
    accompaniment_path = "E:/output5.wav"
    output_mix_path = "E:/output_mix_song.wav"

    try:
        # 步骤 0：导入伴奏
        print("步骤0：将生成伴奏转为MusicXML")
        midi_to_musicxml(musescore_path, input_midi, intermediate_musicxml)

        # 步骤 1：提取主旋律
        print("步骤1：提取伴奏主旋律")
        extract_main_melody(input_musicxml, output_musicxml)

        # 步骤 2: 填充歌词到 MusicXML
        print("步骤2：填充歌词")
        filled_xml = fill_lyrics_to_musicxml(output_musicxml, kana_lyrics, output_ii_musicxml)

        # 步骤 3: 使用 Sinsy 合成歌声
        print("步骤3：Sinsy歌声合成")
        synthesize_with_sinsy_cli(
            input_xml_path=output_ii_musicxml,
            output_wav_path=output_wav,
            language="japanese",  # 根据文件内容选择合适的语言
            voice_bank=1,  # 选择默认声库
            vibrato=1.2,  # 设置颤音强度
            pitch=2  # 调高音高2个半音
        )

        # 步骤 4：混合伴奏和vocal
        print("步骤4：混合伴奏和vocal")
        mix_audio(output_wav, accompaniment_path, output_mix_path, vocal_gain=5, accompaniment_gain=-5)

    except Exception as e:
        print(f"流程失败：{e}")