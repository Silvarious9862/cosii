import numpy as np
from scipy.io import wavfile
from typing import Callable


def generate_signal(name: str,
                    func: Callable[[np.ndarray], np.ndarray],
                    fs: int = 44100,
                    duration: float = 3.0):
    """
    name  – имя WAV-файла
    func  – функция сигнала: принимает массив t, возвращает массив s(t)
    fs    – частота дискретизации
    duration – длительность в секундах
    пример вызова:
    generate_signal('sig1.wav', lambda t: 2*np.sin(3*t) + np.cos(2*t + 1))
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)

    s = func(t)                      

    s_norm = s / np.max(np.abs(s))   # нормализация
    s_int16 = (s_norm * 32767).astype(np.int16)

    wavfile.write(name, fs, s_int16)


def read_signal(name: str):
    """
    пример вызова:
    fs1, x_list = read_signal('sig1.wav')
    """
    fs, data = wavfile.read(name)

    if data.ndim > 1:
        data = data[:, 0]

    x = data.astype(np.float32) / 32767.0
    
    return fs, x
