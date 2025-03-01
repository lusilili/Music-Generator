from music21 import converter, stream, note
import re
import subprocess
import pysinsy
import soundfile as sf
import pykakasi

# 模型文件路径
model_path = r"E:\midi-model-main\midi-model\sinsy-0.92\dic"

def midi_to_musicxml(musescore_path, input_midi, output_musicxml):
    """
    使用 MuseScore 将 MIDI 文件转换为 MusicXML 文件。

    Args:
        musescore_path (str): MuseScore 可执行文件路径。
        input_midi (str): 输入的 MIDI 文件路径。
        output_musicxml (str): 输出的 MusicXML 文件路径。
    """
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

    # !!优化策略二：选择音高变化较为显著的作为主旋律
    # """计算声部的音高变化幅度（标准差）。"""
    # pitches = [n.pitch.midi for n in part.flat.notes if isinstance(n, note.Note)]
    # if len(pitches) > 1:
    #     return sum(abs(pitches[i] - pitches[i - 1]) for i in range(1, len(pitches))) / len(pitches)
    # return 0
    # # 替换原先的音符数量判断逻辑
    # melody_change = melody_variance(part)
    # if melody_change > max_notes:
    #     max_notes = melody_change
    #     main_melody = part

    # 检查是否找到主旋律
    if main_melody is None:
        raise ValueError("未找到主旋律，可能输入文件不包含音符")

    # 创建一个新的乐谱，只包含主旋律
    new_score = stream.Score()
    # 添加小节线，方便对齐
    for element in score.flat.getElementsByClass(['TimeSignature', 'Barline']):
        new_score.append(element)

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

# 中文歌词切分处理
# def split_lyrics(lyrics: str):
#     """
#     对歌词进行切分为逐字歌词。
#     支持自动处理标点符号及空格。
#
#     Args:
#         lyrics_text (str): 输入的歌词字符串。
#
#     Returns:
#         list: 切分后的歌词列表。
#     """
#     # 使用正则表达式将汉字与标点分开
#     pattern = re.compile(r"[\u4e00-\u9fa5]|[\u3000-\u303f\uff00-\uffef]|[\u3040-\u30FF\u4E00-\u9FFF\u3000-\u303F\uFF00-\uFFEF]")
#     characters = pattern.findall(lyrics)
#     return characters


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

def convert_musicxml2midi(musescore_path, input_musicxml, output_midi):
    """
    使用 MuseScore 将 MusicXML 文件转换为 MIDI 文件。

    Args:
        musescore_path (str): MuseScore 可执行文件路径。
        input_musicxml (str): 输入的 MusicXML 文件路径。
        output_midi (str): 输出的 MIDI 文件路径。
    """
    try:
        subprocess.run(
            [musescore_path, "-o", output_midi, input_musicxml],
            check=True
        )
        print(f"成功将 {input_musicxml} 转换为 {output_midi}")
    except subprocess.CalledProcessError as e:
        print(f"转换失败: {e}")

def synthesize_with_pysinsy(musicxml_path, output_wav_path, model_path):
    """
    使用 Sinsy 合成歌声。

    Args:
        musicxml_path (str): 输入的 MusicXML 文件路径。
        output_wav_path (str): 输出的 WAV 文件路径。
        sinsy_instance (Sinsy): 可选的 Sinsy 实例。

    Returns:
        None
    """
    # 方案一：本地合成（效果差）
    sinsy = pysinsy.sinsy.Sinsy()

    # 初始化 Sinsy
    if not sinsy.setLanguages("j", model_path):
        raise RuntimeError("Failed to set Japanese model.")

    # 加载带歌词的 MusicXML 文件
    if not sinsy.loadScoreFromMusicXML(musicxml_path):
        raise RuntimeError(f"Failed to load MusicXML file: {musicxml_path}")

    # 加载默认 HTS voice 模型
    if not sinsy.loadVoices(pysinsy.get_default_htsvoice()):
        raise RuntimeError("Failed to load HTS voices.")

    # 合成歌声
    wav, sr = sinsy.synthesize()
    if len(wav) == 0 or sr != 48000:
        raise RuntimeError("Failed to synthesize.")

    # 保存为 WAV 文件
    sf.write(output_wav_path, wav, sr)
    print(f"歌声生成完成，结果保存到 {output_wav_path}")

    # 清理 Sinsy 的状态
    sinsy.clearScore()

    # 方案二：网页调用



if __name__ == "__main__":
    # 用户输入
    lyrics = "春の風 桜舞う 夢の中 君と歩く 青い空 光揺れる 心響く 愛の歌 夜が明ける 未来を描く 君の声 響く空へ 星の下で,願いを込めて,永遠に続く,この愛を信じて"  # 替换为你的歌词
    kana_lyrics = split_to_syllables(lyrics)
    print(kana_lyrics)

    input_midi = "E:/output5.mid"
    intermediate_musicxml = "E:/output5-all.xml"  # 替换为你的 MusicXML 文件路径
    output_musicxml = "E:/output_main_stream-5all.xml"
    output_ii_musicxml = "E:/output_with_lyrics-5all.xml"
    output_midi = "E:/output_with_lyrics-5all.mid"
    output_wav = "E:/output_song-5all.wav"

    musescore_path = "C:/Program Files/MuseScore 4/bin/MuseScore4.exe"

    # 步骤 1：导入伴奏
    midi_to_musicxml(musescore_path, input_midi, intermediate_musicxml)

    # 步骤 2：提取主旋律
    extract_main_melody(intermediate_musicxml, output_musicxml)

    # 步骤 3：填充歌词到 MusicXML
    filled_xml = fill_lyrics_to_musicxml(output_musicxml, kana_lyrics, output_ii_musicxml)

    # 步骤 4：使用musescore将带歌词xml转换为midi
    convert_musicxml2midi(musescore_path, filled_xml, output_midi)

    # # 步骤 3: 使用 Sinsy 合成歌声
    # synthesize_with_pysinsy(filled_xml, output_wav, model_path)
