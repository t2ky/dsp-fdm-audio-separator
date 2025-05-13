import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg
from scipy.io import wavfile


def main():
    # 音声ファイルを読み込み
    fs, data = wavfile.read("raggvoice.wav")  # 40kHzでサンプリングされたファイル

    # モノラルに変換（ステレオの場合）
    if len(data.shape) > 1:
        data = data[:, 0]

    # 信号を正規化
    data = data.astype(float) / np.max(np.abs(data))

    # FFTを計算
    t = np.arange(0, len(data)) / fs
    f = np.arange(0, fs, fs / len(data))
    fftx = np.fft.fft(data)

    # バンドパスフィルタを作成
    bp1_sos = sg.iirfilter(
        4,
        [6000 / (fs / 2), 10000 / (fs / 2)],
        btype="band",
        ftype="butter",
        output="sos",
    )
    filtered1 = sg.sosfilt(bp1_sos, data)

    # 8kHzで復調
    t_demod = np.arange(0, len(filtered1)) / fs
    demod1 = filtered1 * np.cos(2 * np.pi * 8000 * t_demod)

    # ローパスフィルタで音声成分を抽出（4kHz以下）
    lpf_demod1 = sg.iirfilter(
        4, 4000 / (fs / 2), btype="low", ftype="butter", output="sos"
    )
    voice1 = sg.sosfilt(lpf_demod1, demod1)

    # 16kHzキャリアの音声を分離（帯域: 14kHz-18kHz）
    bp2_sos = sg.iirfilter(
        4,
        [14000 / (fs / 2), 18000 / (fs / 2)],
        btype="band",
        ftype="butter",
        output="sos",
    )
    filtered2 = sg.sosfilt(bp2_sos, data)

    # 16kHzで復調
    demod2 = filtered2 * np.cos(2 * np.pi * 16000 * t_demod)

    # ローパスフィルタで音声成分を抽出（4kHz以下）
    lpf_demod2 = sg.iirfilter(
        4, 4000 / (fs / 2), btype="low", ftype="butter", output="sos"
    )
    voice2 = sg.sosfilt(lpf_demod2, demod2)

    # 時間領域信号をプロット
    plt.figure(figsize=(12, 8))

    # 元の信号
    plt.subplot(3, 1, 1)
    plt.plot(t, data)
    plt.title("Original FDM Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 音声1（8kHzキャリア）
    plt.subplot(3, 1, 2)
    plt.plot(t, voice1)
    plt.title("Demultiplexed Voice 1 (8kHz carrier)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    # 音声2（16kHzキャリア）
    plt.subplot(3, 1, 3)
    plt.plot(t, voice2)
    plt.title("Demultiplexed Voice 2 (16kHz carrier)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 周波数スペクトラムをプロット
    plt.figure(figsize=(12, 8))

    # 元の信号のスペクトラム
    plt.subplot(3, 1, 1)
    plt.plot(f[: len(f) // 2], np.abs(fftx[: len(fftx) // 2]))
    plt.title("Original FDM Signal Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 20000)
    plt.grid(True)

    # 音声1のスペクトラム
    fft_voice1 = np.fft.fft(voice1)
    plt.subplot(3, 1, 2)
    plt.plot(f[: len(f) // 2], np.abs(fft_voice1[: len(fft_voice1) // 2]))
    plt.title("Voice 1 Spectrum (after demodulation)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 20000)
    plt.grid(True)

    # 音声2のスペクトラム
    fft_voice2 = np.fft.fft(voice2)
    plt.subplot(3, 1, 3)
    plt.plot(f[: len(f) // 2], np.abs(fft_voice2[: len(fft_voice2) // 2]))
    plt.title("Voice 2 Spectrum (after demodulation)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 20000)
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # 分離された音声を保存
    wavfile.write("voice1.wav", int(fs), (voice1 * 32767).astype(np.int16))
    wavfile.write("voice2.wav", int(fs), (voice2 * 32767).astype(np.int16))

    print(f"Sampling rate: {fs} Hz")
    print(f"Signal length: {len(data)} samples")
    print(f"Duration: {len(data)/fs:.2f} seconds")
    print("Voice signals saved as voice1.wav and voice2.wav")


if __name__ == "__main__":
    main()
