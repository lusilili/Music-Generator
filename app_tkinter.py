import tkinter as tk
from tkinter import filedialog, messagebox
import os
import threading
import torch
import numpy as np

# 假设 MIDI 生成逻辑在 `generate` 函数中
def generate_midi(description, bpm, time_sig, key_sig, gen_events, seed, temp, top_p, top_k, output_path):
    """
    调用 MIDI 生成逻辑，并将生成的 MIDI 文件保存到指定路径。
    """
    # 模拟生成逻辑（需要替换为真实逻辑）
    print(f"Generating MIDI with description: {description}, BPM: {bpm}, Time Sig: {time_sig}, Key Sig: {key_sig}")
    # 示例生成内容，实际应调用生成函数
    with open(output_path, 'w') as f:
        f.write(f"MIDI generated: {description}, BPM: {bpm}, Time Sig: {time_sig}, Key Sig: {key_sig}")
    return output_path


def run_gui():
    def select_output_path():
        path = filedialog.asksaveasfilename(
            defaultextension=".mid",
            filetypes=[("MIDI files", "*.mid"), ("All files", "*.*")]
        )
        if path:
            output_path_entry.delete(0, tk.END)
            output_path_entry.insert(0, path)

    def generate():
        # 获取用户输入
        description = description_entry.get()
        bpm = bpm_entry.get()
        time_sig = time_sig_var.get()
        key_sig = key_sig_var.get()
        gen_events = gen_events_entry.get()
        seed = seed_entry.get()
        temp = temp_entry.get()
        top_p = top_p_entry.get()
        top_k = top_k_entry.get()
        output_path = output_path_entry.get()

        # 输入校验
        if not description or not output_path:
            messagebox.showerror("Error", "Please provide a description and output path.")
            return

        # 后台生成
        def background_task():
            try:
                result = generate_midi(description, bpm, time_sig, key_sig, gen_events, seed, temp, top_p, top_k, output_path)
                messagebox.showinfo("Success", f"MIDI file generated successfully: {result}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate MIDI: {e}")

        threading.Thread(target=background_task).start()

    # 主窗口
    root = tk.Tk()
    root.title("MIDI Generator")
    root.geometry("600x400")

    # 描述输入
    tk.Label(root, text="MIDI Description:").grid(row=0, column=0, sticky="w", padx=10, pady=5)
    description_entry = tk.Entry(root, width=40)
    description_entry.grid(row=0, column=1, padx=10, pady=5)

    # BPM 输入
    tk.Label(root, text="BPM:").grid(row=1, column=0, sticky="w", padx=10, pady=5)
    bpm_entry = tk.Entry(root, width=10)
    bpm_entry.grid(row=1, column=1, sticky="w", padx=10, pady=5)

    # 时间签名选择
    tk.Label(root, text="Time Signature:").grid(row=2, column=0, sticky="w", padx=10, pady=5)
    time_sig_var = tk.StringVar(value="4/4")
    time_sig_menu = tk.OptionMenu(root, time_sig_var, "4/4", "3/4", "6/8")
    time_sig_menu.grid(row=2, column=1, sticky="w", padx=10, pady=5)

    # 调号选择
    tk.Label(root, text="Key Signature:").grid(row=3, column=0, sticky="w", padx=10, pady=5)
    key_sig_var = tk.StringVar(value="C")
    key_sig_menu = tk.OptionMenu(root, key_sig_var, "C", "D", "E", "F", "G", "A", "B")
    key_sig_menu.grid(row=3, column=1, sticky="w", padx=10, pady=5)

    # 生成事件数
    tk.Label(root, text="Max Events:").grid(row=4, column=0, sticky="w", padx=10, pady=5)
    gen_events_entry = tk.Entry(root, width=10)
    gen_events_entry.grid(row=4, column=1, sticky="w", padx=10, pady=5)

    # 随机种子
    tk.Label(root, text="Seed:").grid(row=5, column=0, sticky="w", padx=10, pady=5)
    seed_entry = tk.Entry(root, width=10)
    seed_entry.grid(row=5, column=1, sticky="w", padx=10, pady=5)

    # Temperature
    tk.Label(root, text="Temperature:").grid(row=6, column=0, sticky="w", padx=10, pady=5)
    temp_entry = tk.Entry(root, width=10)
    temp_entry.grid(row=6, column=1, sticky="w", padx=10, pady=5)

    # Top-p
    tk.Label(root, text="Top-p:").grid(row=7, column=0, sticky="w", padx=10, pady=5)
    top_p_entry = tk.Entry(root, width=10)
    top_p_entry.grid(row=7, column=1, sticky="w", padx=10, pady=5)

    # Top-k
    tk.Label(root, text="Top-k:").grid(row=8, column=0, sticky="w", padx=10, pady=5)
    top_k_entry = tk.Entry(root, width=10)
    top_k_entry.grid(row=8, column=1, sticky="w", padx=10, pady=5)

    # 输出路径
    tk.Label(root, text="Output Path:").grid(row=9, column=0, sticky="w", padx=10, pady=5)
    output_path_entry = tk.Entry(root, width=40)
    output_path_entry.grid(row=9, column=1, padx=10, pady=5)
    tk.Button(root, text="Browse", command=select_output_path).grid(row=9, column=2, padx=10, pady=5)

    # 生成按钮
    tk.Button(root, text="Generate MIDI", command=generate, bg="green", fg="white").grid(row=10, column=1, pady=20)

    root.mainloop()


if __name__ == "__main__":
    run_gui()
