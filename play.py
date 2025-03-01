import subprocess
import os


def run_music_generation_directly():
    """
    集成命令到代码中，直接调用可执行文件生成 MIDI 和音频。
    """
    # 可执行文件路径
    executable_path = "E:/app-gpu/app.exe"

    # 参数配置
    model_base_path = "E:/app-gpu/models/default"
    model_config = "auto"
    soundfont_path = "E:/app-gpu/soundfont.sf2"
    batch = 1
    max_gen = 1024
    port = 7860
    share = True
    output_dir = "E:/app-gpu/pla-outputs"

    # 构建命令
    command = [
        executable_path,
        "--model-base-path", model_base_path,
        "--model-config", model_config,
        "--soundfont-path", soundfont_path,
        "--batch", str(batch),
        "--max-gen", str(max_gen),
        "--port", str(port),
    ]

    if share:
        command.append("--share")

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 执行命令
    print(f"Executing command: {' '.join(command)}")
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # 打印输出和错误信息
    print("STDOUT:\n", process.stdout.decode("utf-8", errors="ignore"))
    print("STDERR:\n", process.stderr.decode("gbk", errors="ignore"))

    return process


if __name__ == "__main__":
    # 调用集成的命令逻辑
    result = run_music_generation_directly()

    # 检查运行结果
    if result.returncode == 0:
        print("任务成功完成！")
    else:
        print("任务失败，请检查错误信息。")
