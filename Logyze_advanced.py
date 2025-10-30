# -*- coding: utf-8 -*-
"""
Logyze Advanced - Анализ затухающих колебаний с ДВУМЯ датчиками RF603HS
Версия 3.0 - Синхронная запись с двух датчиков
"""

import sys
import serial
import serial.tools.list_ports
import struct
import time
import csv
from datetime import datetime
from pathlib import Path
import threading

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QMessageBox, QSplitter,
    QTabWidget, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QCursor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================================
# КЛАСС АНАЛИЗАТОРА
# ============================================================================

class RF603OscillationAnalyzer:
    """Класс для анализа затухающих колебаний"""

    def __init__(self):
        self.data = None
        self.processed_data = None
        self.original_processed_data = None
        self.oscillation_start = 0
        self.oscillation_end = 0
        self.corrected_peaks = None
        self.current_period = None
        self.current_frequency = None
        self.log_decrement = None
        self.loss_factor = None
        self.damping_ratio = None

    def load_csv_column(self, filename, column_name):
        """Загрузка CSV файла с выбором колонки"""
        try:
            for delimiter in [',', ';']:
                try:
                    self.data = pd.read_csv(filename, delimiter=delimiter, encoding='utf-8')
                    if len(self.data.columns) >= 2:
                        break
                except:
                    continue

            if column_name in self.data.columns:
                distance_col = column_name
            else:
                return False

            if 'Time_s' in self.data.columns:
                time_col = 'Time_s'
            elif 'Временная_метка' in self.data.columns:
                time_col = 'Временная_метка'
            else:
                return False

            self.data = pd.DataFrame({
                'Расстояние_мм': self.data[distance_col],
                'Временная_метка': self.data[time_col],
                'Point': range(len(self.data))
            })

            print(f"✅ Данные загружены для {column_name}: {len(self.data)} строк")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки {column_name}: {e}")
            return False

    def normalize_data(self):
        """Нормировка данных"""
        if self.data is None:
            return False

        try:
            self.processed_data = self.data.copy()
            first_distance = self.processed_data.iloc[0]['Расстояние_мм']
            self.processed_data['Расстояние_норм'] = (
                self.processed_data['Расстояние_мм'] - first_distance
            )
            self.original_processed_data = self.processed_data.copy()

            self.corrected_peaks = None
            self.current_period = None
            self.current_frequency = None
            self.log_decrement = None
            self.loss_factor = None
            self.damping_ratio = None

            return True
        except Exception as e:
            print(f"❌ Ошибка нормировки: {e}")
            return False

    def reset_to_original(self):
        """Сброс к исходным данным"""
        if self.original_processed_data is not None:
            self.processed_data = self.original_processed_data.copy()
            self.oscillation_start = 0
            self.oscillation_end = len(self.processed_data) - 1

            self.corrected_peaks = None
            self.current_period = None
            self.current_frequency = None
            self.log_decrement = None
            self.loss_factor = None
            self.damping_ratio = None

            return True
        return False

    def crop_by_time(self, start_time, end_time):
        """Обрезка по времени"""
        if self.processed_data is None:
            return False

        try:
            start_mask = self.processed_data['Временная_метка'] >= start_time
            end_mask = self.processed_data['Временная_метка'] <= end_time

            if not any(start_mask) or not any(end_mask):
                return False

            start_idx = self.processed_data[start_mask].index[0]
            end_idx = self.processed_data[end_mask].index[-1]

            return self.crop_by_points(start_idx, end_idx)
        except:
            return False

    def crop_by_points(self, start_point, end_point):
        """Обрезка по точкам"""
        if self.processed_data is None:
            return False

        try:
            start_idx = max(0, min(start_point, len(self.processed_data) - 1))
            end_idx = max(start_idx + 1, min(end_point, len(self.processed_data) - 1))

            self.processed_data = self.processed_data.iloc[start_idx:end_idx + 1].reset_index(drop=True)
            self.oscillation_start = start_idx
            self.oscillation_end = end_idx

            self.corrected_peaks = None
            self.current_period = None
            self.current_frequency = None
            self.log_decrement = None
            self.loss_factor = None
            self.damping_ratio = None

            return True
        except Exception as e:
            print(f"❌ Ошибка обрезки: {e}")
            return False

    def find_release_point(self, threshold=0.5):
        """Поиск начала колебаний"""
        if self.processed_data is None:
            return 0

        distances = self.processed_data['Расстояние_норм'].values

        for i in range(1, len(distances) - 10):
            window = distances[i:i+10]
            if np.max(window) - np.min(window) > threshold:
                return max(0, i - 5)

        return 0

    def auto_crop_oscillations(self, duration_after_start=1.0):
        """Автоматическая обрезка"""
        if self.processed_data is None:
            return False, None, None, None

        try:
            self.processed_data = self.original_processed_data.copy()

            start_idx = self.find_release_point(threshold=0.5)
            start_time = self.processed_data.iloc[start_idx]['Временная_метка']
            end_time = start_time + duration_after_start

            mask = ((self.processed_data['Временная_метка'] >= start_time) &
                    (self.processed_data['Временная_метка'] <= end_time))
            self.processed_data = self.processed_data[mask].reset_index(drop=True)

            period, frequency, peaks = self.calculate_period_frequency_improved()

            return True, period, frequency, peaks
        except Exception as e:
            print(f"❌ Ошибка автообрезки: {e}")
            return False, None, None, None

    def _find_peaks_adaptive(self, distances, amplitude):
        """Адаптивный поиск пиков"""
        height_threshold = amplitude * 0.3
        peaks1, _ = find_peaks(distances, height=height_threshold, distance=5)

        if len(peaks1) >= 2:
            return peaks1

        peaks2, _ = find_peaks(distances, height=amplitude * 0.1, distance=3)
        if len(peaks2) >= 2:
            return peaks2

        peaks3, _ = find_peaks(distances)
        return peaks3

    def set_manual_peaks(self, peaks):
        """Установка пиков вручную"""
        self.corrected_peaks = np.array(peaks)

    def calculate_period_frequency_improved(self):
        """Расчет периода и частоты"""
        if self.processed_data is None:
            return None, None, None

        try:
            distances = self.processed_data['Расстояние_норм'].values
            timestamps = self.processed_data['Временная_метка'].values

            if self.corrected_peaks is not None and len(self.corrected_peaks) >= 2:
                peaks = self.corrected_peaks
            else:
                if len(distances) > 11:
                    distances_smooth = savgol_filter(distances, 11, 3)
                else:
                    distances_smooth = distances

                amplitude = np.max(distances_smooth) - np.min(distances_smooth)
                peaks = self._find_peaks_adaptive(distances_smooth, amplitude)

            if len(peaks) < 2:
                return None, None, None

            time_intervals = []
            for i in range(len(peaks) - 1):
                t1 = timestamps[peaks[i]]
                t2 = timestamps[peaks[i + 1]]
                time_intervals.append(t2 - t1)

            avg_period = np.mean(time_intervals)
            frequency = 1.0 / avg_period if avg_period > 0 else 0

            self.current_period = avg_period
            self.current_frequency = frequency

            return avg_period, frequency, peaks

        except Exception as e:
            print(f"❌ Ошибка расчета: {e}")
            return None, None, None

    def calculate_logarithmic_decrement(self, peaks):
        """Расчет логарифмического декремента"""
        if peaks is None or len(peaks) < 2:
            return None, None, None

        try:
            distances = self.processed_data['Расстояние_норм'].values
            amplitudes = distances[peaks]

            decrements = []
            for i in range(len(peaks) - 1):
                A_i = abs(amplitudes[i])
                A_i_plus_1 = abs(amplitudes[i + 1])

                if A_i_plus_1 > 0:
                    delta = np.log(A_i / A_i_plus_1)
                    decrements.append(delta)

            if not decrements:
                return None, None, None

            avg_decrement = np.mean(decrements)
            damping_ratio = avg_decrement / np.sqrt(4 * np.pi**2 + avg_decrement**2)
            loss_factor = 2 * damping_ratio

            self.log_decrement = avg_decrement
            self.damping_ratio = damping_ratio
            self.loss_factor = loss_factor

            return avg_decrement, damping_ratio, loss_factor

        except Exception as e:
            print(f"❌ Ошибка расчета декремента: {e}")
            return None, None, None


# ============================================================================
# КЛАСС ДЛЯ РАБОТЫ С ДАТЧИКОМ
# ============================================================================

class RF603Sensor:
    """Класс для работы с датчиком RF603HS"""

    BAUDRATES = [9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600]

    def __init__(self):
        self.serial_port = None
        self.is_connected = False

    def list_ports(self):
        ports = serial.tools.list_ports.comports()
        return [port.device for port in ports]

    def connect(self, port, baudrate=9600, address=1, timeout=1.0):
        try:
            self.serial_port = serial.Serial(
                port=port,
                baudrate=baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=timeout
            )
            time.sleep(0.1)
            self.is_connected = True
            return True
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        self.is_connected = False

    def change_baudrate(self, new_baudrate):
        """Смена baudrate без отключения"""
        if self.serial_port and self.serial_port.is_open:
            try:
                self.serial_port.baudrate = new_baudrate
                time.sleep(0.1)
                return True
            except:
                return False
        return False

    def send_request(self, address, code):
        inc0 = address & 0x7F
        inc1 = 0x80 | (code & 0x0F)
        self.serial_port.write(bytes([inc0, inc1]))

    def identify(self, address=1):
        try:
            self.serial_port.reset_input_buffer()
            self.send_request(address, 0x01)
            time.sleep(0.1)

            response = self.serial_port.read(16)
            if len(response) >= 16:
                decoded_bytes = []
                i = 0
                while i < len(response) - 1:
                    if (response[i] & 0x80) and (response[i+1] & 0x80):
                        low = response[i] & 0x0F
                        high = response[i+1] & 0x0F
                        decoded_bytes.append(low | (high << 4))
                        i += 2
                    else:
                        i += 1

                if len(decoded_bytes) >= 8:
                    device_type = decoded_bytes[0]
                    firmware_ver = decoded_bytes[1]
                    serial_num = decoded_bytes[2] | (decoded_bytes[3] << 8)
                    base_distance = decoded_bytes[4] | (decoded_bytes[5] << 8)
                    measurement_range = decoded_bytes[6] | (decoded_bytes[7] << 8)

                    print(f"✅ Датчик: тип {device_type}, прошивка {firmware_ver}, S/N {serial_num}")
                    print(f"   Базовое расстояние: {base_distance} мм, Диапазон: {measurement_range} мм")
                    return True
            return False
        except Exception as e:
            print(f"❌ Ошибка идентификации: {e}")
            return False

    def start_stream(self, address=1):
        try:
            self.serial_port.reset_input_buffer()
            time.sleep(0.05)
            self.send_request(address, 0x07)
            time.sleep(0.1)
            return True
        except:
            return False

    def stop_stream(self, address=1):
        try:
            self.send_request(address, 0x08)
            time.sleep(0.05)
            return True
        except:
            return False

    def find_packet_start(self, timeout=2.0):
        """Поиск начала пакета данных"""
        start_time = time.time()
        byte_buffer = []

        while (time.time() - start_time) < timeout:
            if self.serial_port.in_waiting > 0:
                byte = self.serial_port.read(1)
                if len(byte) > 0:
                    byte_buffer.append(byte[0])

                    if len(byte_buffer) >= 2:
                        if (byte_buffer[-2] & 0x80) and (byte_buffer[-1] & 0x80):
                            sb = (byte_buffer[-2] >> 6) & 0x01
                            cnt = (byte_buffer[-2] >> 4) & 0x03
                            return bytes(byte_buffer[-2:])

                        byte_buffer = byte_buffer[-1:]
            else:
                time.sleep(0.001)

        return None

    def read_measurement(self):
        """Чтение измерения с проверкой формата пакета"""
        try:
            if self.serial_port.in_waiting >= 2:
                data = self.serial_port.read(2)
                if len(data) == 2:
                    value, status_bit, counter, valid = self._decode_measurement_packet(data[0], data[1])
                    if valid:
                        return value
            return None
        except:
            return None

    def _decode_measurement_packet(self, byte1, byte2):
        """Декодирование пакета измерения"""
        if not ((byte1 & 0x80) and (byte2 & 0x80)):
            return None, None, None, False

        sb1 = (byte1 >> 6) & 0x01
        cnt1 = (byte1 >> 4) & 0x03
        low_nibble = byte1 & 0x0F

        sb2 = (byte2 >> 6) & 0x01
        cnt2 = (byte2 >> 4) & 0x03
        high_nibble = byte2 & 0x0F

        if sb1 != sb2 or cnt1 != cnt2:
            return None, None, None, False

        value = low_nibble | (high_nibble << 4)

        return value, sb1, cnt1, True


# ============================================================================
# ПОТОК ЗАПИСИ ДВУХ ДАТЧИКОВ
# ============================================================================

class DualSensorRecordingThread(QThread):
    """Поток для синхронной записи данных с двух датчиков"""

    data_received = pyqtSignal(float, float, int, float)  # distance1, distance2, point, time
    recording_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, sensor1, sensor2, base_distance, measurement_range):
        super().__init__()
        self.sensor1 = sensor1
        self.sensor2 = sensor2
        self.base_distance = base_distance
        self.measurement_range = measurement_range
        self.running = False
        self.data_list = []

    def run(self):
        """Основной цикл записи"""
        self.running = True
        self.data_list = []

        try:
            # Синхронизация обоих датчиков
            print("🔍 Синхронизация датчика 1...")
            packet1 = self.sensor1.find_packet_start(timeout=3.0)

            print("🔍 Синхронизация датчика 2...")
            packet2 = self.sensor2.find_packet_start(timeout=3.0)

            if packet1 is None or packet2 is None:
                self.error_occurred.emit("Не удалось синхронизироваться с датчиками")
                return

            print("✅ Оба датчика синхронизированы, начинаем запись...")

            start_time = time.perf_counter()
            point = 0

            # Обработка первых пакетов
            byte1_s1, byte2_s1 = packet1[0], packet1[1]
            value1, _, _, valid1 = self.sensor1._decode_measurement_packet(byte1_s1, byte2_s1)

            byte1_s2, byte2_s2 = packet2[0], packet2[1]
            value2, _, _, valid2 = self.sensor2._decode_measurement_packet(byte1_s2, byte2_s2)

            if valid1 and valid2 and value1 is not None and value2 is not None:
                distance1 = self.base_distance + (value1 * self.measurement_range / 16384.0)
                distance2 = self.base_distance + (value2 * self.measurement_range / 16384.0)
                current_time = time.perf_counter() - start_time

                self.data_list.append({
                    'Time_s': current_time,
                    'Distance_Sensor1_mm': distance1,
                    'Distance_Sensor2_mm': distance2
                })

                self.data_received.emit(distance1, distance2, point, current_time)
                point += 1

            # Продолжаем читать данные
            while self.running:
                value1 = self.sensor1.read_measurement()
                value2 = self.sensor2.read_measurement()

                if value1 is not None and value2 is not None and value1 > 0 and value2 > 0:
                    distance1 = self.base_distance + (value1 * self.measurement_range / 16384.0)
                    distance2 = self.base_distance + (value2 * self.measurement_range / 16384.0)
                    current_time = time.perf_counter() - start_time

                    self.data_list.append({
                        'Time_s': current_time,
                        'Distance_Sensor1_mm': distance1,
                        'Distance_Sensor2_mm': distance2
                    })

                    self.data_received.emit(distance1, distance2, point, current_time)
                    point += 1

                time.sleep(0.0001)

            # Сохранение
            if self.data_list:
                filename = f"dual_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['Time_s', 'Distance_Sensor1_mm', 'Distance_Sensor2_mm'])
                    writer.writeheader()
                    writer.writerows(self.data_list)

                self.recording_finished.emit(filename)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def stop(self):
        """Остановка записи"""
        self.running = False


# ============================================================================
# ВИДЖЕТ ГРАФИКА ДЛЯ ДВУХ ДАТЧИКОВ
# ============================================================================

class DualSensorPlotWidget(QWidget):
    """Виджет с графиком для двух датчиков"""

    def __init__(self, plot_title='График'):
        super().__init__()

        self.times = []
        self.distances1 = []
        self.distances2 = []
        self.peaks1 = []
        self.peaks2 = []
        self.analyzer1 = None
        self.analyzer2 = None

        self.crop_mode_sensor = None  # 1 или 2
        self.add_peak_mode_sensor = None
        self.remove_peak_mode_sensor = None

        self.crop_start_val = None
        self.crop_end_val = None
        self.crop_start_line = None
        self.crop_end_line = None
        self.dragging_line = None

        self.peak_markers1 = []
        self.peak_texts1 = []
        self.peak_markers2 = []
        self.peak_texts2 = []

        self.init_ui(plot_title)

    def init_ui(self, plot_title):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.figure = Figure(figsize=(10, 5))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)

        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.coord_label = QLabel("Координаты: -")
        layout.addWidget(self.coord_label)

        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

        self.ax.set_title(plot_title, fontweight='bold')
        self.ax.grid(True, alpha=0.3)

    def on_click(self, event):
        """Обработка клика"""
        if event.inaxes != self.ax:
            return

        x, y = event.xdata, event.ydata

        # Режим обрезки
        if self.crop_mode_sensor is not None:
            if self.crop_start_val is None:
                self.crop_start_val = x
                self.crop_start_line = self.ax.axvline(x, color='green', linestyle='--', linewidth=2, label='Начало')
                self.ax.legend()
                self.canvas.draw()
            elif self.crop_end_val is None:
                self.crop_end_val = x
                self.crop_end_line = self.ax.axvline(x, color='red', linestyle='--', linewidth=2, label='Конец')
                self.ax.legend()
                self.canvas.draw()
            return

        # Режим добавления пика
        if self.add_peak_mode_sensor is not None:
            if len(self.times) > 0:
                idx = np.argmin(np.abs(np.array(self.times) - x))

                if self.add_peak_mode_sensor == 1:
                    if idx not in self.peaks1:
                        self.peaks1.append(idx)
                        self.peaks1.sort()
                        self.update_peaks_display()
                        print(f"✅ Пик добавлен для Датчик 1. Всего пиков: {len(self.peaks1)}")
                elif self.add_peak_mode_sensor == 2:
                    if idx not in self.peaks2:
                        self.peaks2.append(idx)
                        self.peaks2.sort()
                        self.update_peaks_display()
                        print(f"✅ Пик добавлен для Датчик 2. Всего пиков: {len(self.peaks2)}")
            return

        # Режим удаления пика
        if self.remove_peak_mode_sensor is not None:
            if len(self.times) > 0:
                idx = np.argmin(np.abs(np.array(self.times) - x))

                if self.remove_peak_mode_sensor == 1 and idx in self.peaks1:
                    self.peaks1.remove(idx)
                    self.update_peaks_display()
                    print(f"✅ Пик удален для Датчик 1. Осталось: {len(self.peaks1)}")
                elif self.remove_peak_mode_sensor == 2 and idx in self.peaks2:
                    self.peaks2.remove(idx)
                    self.update_peaks_display()
                    print(f"✅ Пик удален для Датчик 2. Осталось: {len(self.peaks2)}")
            return

        # Перетаскивание границ
        if self.crop_start_line is not None:
            if abs(x - self.crop_start_val) < 0.02 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]):
                self.dragging_line = 'start'
                return

        if self.crop_end_line is not None:
            if abs(x - self.crop_end_val) < 0.02 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0]):
                self.dragging_line = 'end'
                return

    def on_release(self, event):
        """Отпускание кнопки мыши"""
        self.dragging_line = None

    def on_motion(self, event):
        """Движение мыши"""
        if event.inaxes != self.ax:
            self.coord_label.setText("Координаты: -")
            return

        x, y = event.xdata, event.ydata
        self.coord_label.setText(f"Время: {x:.6f} с, Расстояние: {y:.6f} мм")

        # Перетаскивание границ
        if self.dragging_line == 'start' and self.crop_start_line is not None:
            self.crop_start_val = x
            self.crop_start_line.set_xdata([x, x])
            self.canvas.draw()
        elif self.dragging_line == 'end' and self.crop_end_line is not None:
            self.crop_end_val = x
            self.crop_end_line.set_xdata([x, x])
            self.canvas.draw()

        # Изменение курсора
        near_start = self.crop_start_line is not None and abs(x - self.crop_start_val) < 0.02 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])
        near_end = self.crop_end_line is not None and abs(x - self.crop_end_val) < 0.02 * (self.ax.get_xlim()[1] - self.ax.get_xlim()[0])

        if near_start or near_end:
            self.canvas.setCursor(QCursor(Qt.SizeHorCursor))
        elif self.crop_mode_sensor or self.add_peak_mode_sensor or self.remove_peak_mode_sensor:
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))

    def set_crop_mode(self, enabled, sensor_num=None):
        """Включение режима обрезки"""
        self.crop_mode_sensor = sensor_num if enabled else None
        if enabled:
            self.add_peak_mode_sensor = None
            self.remove_peak_mode_sensor = None
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))

    def set_add_peak_mode(self, enabled, sensor_num=None):
        """Режим добавления пика"""
        self.add_peak_mode_sensor = sensor_num if enabled else None
        if enabled:
            self.crop_mode_sensor = None
            self.remove_peak_mode_sensor = None
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))

    def set_remove_peak_mode(self, enabled, sensor_num=None):
        """Режим удаления пика"""
        self.remove_peak_mode_sensor = sensor_num if enabled else None
        if enabled:
            self.crop_mode_sensor = None
            self.add_peak_mode_sensor = None
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))

    def clear_crop_lines(self):
        """Очистка линий обрезки"""
        if self.crop_start_line:
            self.crop_start_line.remove()
            self.crop_start_line = None
        if self.crop_end_line:
            self.crop_end_line.remove()
            self.crop_end_line = None
        self.crop_start_val = None
        self.crop_end_val = None
        self.canvas.draw()

    def update_plot(self, time_val, distance1, distance2):
        """Обновление графика в реальном времени"""
        self.times.append(time_val)
        self.distances1.append(distance1)
        self.distances2.append(distance2)

        max_points = 5000
        if len(self.times) > max_points:
            self.times = self.times[-max_points:]
            self.distances1 = self.distances1[-max_points:]
            self.distances2 = self.distances2[-max_points:]

        self.ax.clear()
        self.ax.plot(self.times, self.distances1, 'b-', linewidth=1, label='Датчик 1')
        self.ax.plot(self.times, self.distances2, 'r-', linewidth=1, label='Датчик 2')
        self.ax.set_xlabel('Время (с)')
        self.ax.set_ylabel('Расстояние (мм)')
        self.ax.set_title(f'Запись ({len(self.times)} точек)', fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()

    def clear_plot(self):
        """Очистка графика"""
        self.times = []
        self.distances1 = []
        self.distances2 = []
        self.peaks1 = []
        self.peaks2 = []
        self.clear_crop_lines()

        self.ax.clear()
        self.ax.set_xlabel('Время (с)')
        self.ax.set_ylabel('Расстояние (мм)')
        self.ax.set_title('Готов к записи', fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def plot_analysis(self, analyzer1, analyzer2, show_peaks=False):
        """Отображение результатов анализа"""
        if analyzer1.processed_data is None or analyzer2.processed_data is None:
            return

        self.analyzer1 = analyzer1
        self.analyzer2 = analyzer2

        self.times = analyzer1.processed_data['Временная_метка'].values.tolist()
        self.distances1 = analyzer1.processed_data['Расстояние_норм'].values.tolist()
        self.distances2 = analyzer2.processed_data['Расстояние_норм'].values.tolist()

        self.ax.clear()

        self.ax.plot(self.times, self.distances1, 'b-', linewidth=1.5, label='Датчик 1')
        self.ax.plot(self.times, self.distances2, 'r-', linewidth=1.5, label='Датчик 2')

        if show_peaks:
            if analyzer1.corrected_peaks is not None:
                self.peaks1 = analyzer1.corrected_peaks.tolist()
            if analyzer2.corrected_peaks is not None:
                self.peaks2 = analyzer2.corrected_peaks.tolist()

            self.update_peaks_display()
            self._add_results_text(analyzer1, analyzer2)

        self.ax.set_xlabel('Время (с)')
        self.ax.set_ylabel('Расстояние (мм)')
        self.ax.set_title('Анализ двух датчиков', fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()

    def _add_results_text(self, analyzer1, analyzer2):
        """Добавление текстовых блоков с результатами"""
        # Датчик 1
        info1 = '=== ДАТЧИК 1 ===\n'
        if analyzer1.current_period:
            info1 += f'T: {analyzer1.current_period:.6f} с\n'
            info1 += f'f: {analyzer1.current_frequency:.2f} Гц\n'
        if analyzer1.log_decrement:
            info1 += f'δ: {analyzer1.log_decrement:.6f}\n'

        # Датчик 2
        info2 = '=== ДАТЧИК 2 ===\n'
        if analyzer2.current_period:
            info2 += f'T: {analyzer2.current_period:.6f} с\n'
            info2 += f'f: {analyzer2.current_frequency:.2f} Гц\n'
        if analyzer2.log_decrement:
            info2 += f'δ: {analyzer2.log_decrement:.6f}\n'

        # Отображение
        self.ax.text(0.02, 0.98, info1.strip(),
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.85),
                    fontsize=8, family='monospace')

        self.ax.text(0.98, 0.98, info2.strip(),
                    transform=self.ax.transAxes,
                    verticalalignment='top',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.85),
                    fontsize=8, family='monospace')

    def update_peaks_display(self):
        """Обновление отображения пиков"""
        self.peak_markers1 = []
        self.peak_texts1 = []
        self.peak_markers2 = []
        self.peak_texts2 = []

        if len(self.times) == 0:
            return

        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self.ax.clear()

        self.ax.plot(self.times, self.distances1, 'b-', linewidth=1.5, label='Датчик 1')
        self.ax.plot(self.times, self.distances2, 'r-', linewidth=1.5, label='Датчик 2')

        times_arr = np.array(self.times)
        distances1_arr = np.array(self.distances1)
        distances2_arr = np.array(self.distances2)

        # Пики датчика 1
        for i, peak_idx in enumerate(self.peaks1):
            if peak_idx < len(times_arr):
                x = times_arr[peak_idx]
                y = distances1_arr[peak_idx]
                marker, = self.ax.plot(x, y, 'bo', markersize=8)
                text = self.ax.text(x, y, f'  1.{i+1}', fontsize=8, color='blue')
                self.peak_markers1.append(marker)
                self.peak_texts1.append(text)

        # Пики датчика 2
        for i, peak_idx in enumerate(self.peaks2):
            if peak_idx < len(times_arr):
                x = times_arr[peak_idx]
                y = distances2_arr[peak_idx]
                marker, = self.ax.plot(x, y, 'rs', markersize=8)
                text = self.ax.text(x, y, f'  2.{i+1}', fontsize=8, color='red')
                self.peak_markers2.append(marker)
                self.peak_texts2.append(text)

        if self.analyzer1 and self.analyzer2:
            self._add_results_text(self.analyzer1, self.analyzer2)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_xlabel('Время (с)')
        self.ax.set_ylabel('Расстояние (мм)')
        self.ax.set_title('Анализ двух датчиков', fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        self.canvas.draw()

    def get_crop_values(self):
        """Получить значения границ обрезки"""
        return self.crop_start_val, self.crop_end_val

    def save_figure(self, filename):
        """Сохранение графика"""
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False


# ============================================================================
# ГЛАВНОЕ ОКНО
# ============================================================================

class MainWindow(QMainWindow):
    """Главное окно Logyze Advanced"""

    def __init__(self):
        super().__init__()

        self.sensor1 = RF603Sensor()
        self.sensor2 = RF603Sensor()
        self.analyzer1 = RF603OscillationAnalyzer()
        self.analyzer2 = RF603OscillationAnalyzer()
        self.recording_thread = None
        self.is_recording = False
        self.current_file = None

        self.init_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot_buffer)
        self.plot_buffer = []

    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle('Logyze Advanced - Анализатор с двумя датчиками RF603HS v3.0')
        self.setGeometry(100, 100, 1800, 900)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Левая панель
        left_panel = self.create_control_panel()

        # Правая панель
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)

        # График для записи
        self.plot_recording = DualSensorPlotWidget('Синхронная запись двух датчиков')
        self.right_layout.addWidget(self.plot_recording)

        # График для анализа
        self.plot_analysis = DualSensorPlotWidget('Анализ двух датчиков')
        self.plot_analysis.hide()

        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout.addWidget(splitter)

        self.refresh_ports()

    def create_control_panel(self):
        """Создание панели управления"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        tabs = QTabWidget()

        connection_tab = self.create_connection_tab()
        tabs.addTab(connection_tab, "Подключение")

        recording_tab = self.create_recording_tab()
        tabs.addTab(recording_tab, "Запись")

        analysis_tab = self.create_analysis_tab()
        tabs.addTab(analysis_tab, "Анализ")

        editing_tab = self.create_editing_tab()
        tabs.addTab(editing_tab, "Редактирование")

        layout.addWidget(tabs)

        # Консоль
        console_group = QGroupBox("Консоль")
        console_layout = QVBoxLayout()

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumHeight(120)
        console_layout.addWidget(self.console)

        btn_clear = QPushButton("Очистить")
        btn_clear.clicked.connect(self.console.clear)
        console_layout.addWidget(btn_clear)

        console_group.setLayout(console_layout)
        layout.addWidget(console_group)

        return panel

    def create_connection_tab(self):
        """Вкладка подключения"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Датчик 1
        sensor1_group = QGroupBox("🔵 Датчик 1")
        sensor1_layout = QVBoxLayout()

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("COM-порт:"))
        self.combo_port1 = QComboBox()
        h1.addWidget(self.combo_port1)
        btn_refresh = QPushButton("🔄")
        btn_refresh.setMaximumWidth(40)
        btn_refresh.clicked.connect(self.refresh_ports)
        h1.addWidget(btn_refresh)
        sensor1_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Baudrate:"))
        self.combo_baudrate1 = QComboBox()
        self.combo_baudrate1.addItems([str(b) for b in RF603Sensor.BAUDRATES])
        self.combo_baudrate1.setCurrentText("9600")
        h2.addWidget(self.combo_baudrate1)
        sensor1_layout.addLayout(h2)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Адрес:"))
        self.spin_address1 = QSpinBox()
        self.spin_address1.setRange(1, 127)
        self.spin_address1.setValue(1)
        h3.addWidget(self.spin_address1)
        sensor1_layout.addLayout(h3)

        btn_layout1 = QHBoxLayout()
        self.btn_connect1 = QPushButton("Подключить")
        self.btn_connect1.clicked.connect(lambda: self.connect_sensor(1))
        btn_layout1.addWidget(self.btn_connect1)

        self.btn_identify1 = QPushButton("ID")
        self.btn_identify1.clicked.connect(lambda: self.identify_sensor(1))
        self.btn_identify1.setEnabled(False)
        btn_layout1.addWidget(self.btn_identify1)
        sensor1_layout.addLayout(btn_layout1)

        # Смена baudrate
        h4 = QHBoxLayout()
        h4.addWidget(QLabel("Новый Baudrate:"))
        self.combo_new_baudrate1 = QComboBox()
        self.combo_new_baudrate1.addItems([str(b) for b in RF603Sensor.BAUDRATES])
        self.combo_new_baudrate1.setCurrentText("9600")
        h4.addWidget(self.combo_new_baudrate1)
        sensor1_layout.addLayout(h4)

        self.btn_change_baudrate1 = QPushButton("Сменить Baudrate")
        self.btn_change_baudrate1.clicked.connect(lambda: self.change_baudrate(1))
        self.btn_change_baudrate1.setEnabled(False)
        sensor1_layout.addWidget(self.btn_change_baudrate1)

        sensor1_group.setLayout(sensor1_layout)
        layout.addWidget(sensor1_group)

        # Датчик 2
        sensor2_group = QGroupBox("🔴 Датчик 2")
        sensor2_layout = QVBoxLayout()

        h5 = QHBoxLayout()
        h5.addWidget(QLabel("COM-порт:"))
        self.combo_port2 = QComboBox()
        h5.addWidget(self.combo_port2)
        sensor2_layout.addLayout(h5)

        h6 = QHBoxLayout()
        h6.addWidget(QLabel("Baudrate:"))
        self.combo_baudrate2 = QComboBox()
        self.combo_baudrate2.addItems([str(b) for b in RF603Sensor.BAUDRATES])
        self.combo_baudrate2.setCurrentText("9600")
        h6.addWidget(self.combo_baudrate2)
        sensor2_layout.addLayout(h6)

        h7 = QHBoxLayout()
        h7.addWidget(QLabel("Адрес:"))
        self.spin_address2 = QSpinBox()
        self.spin_address2.setRange(1, 127)
        self.spin_address2.setValue(1)
        h7.addWidget(self.spin_address2)
        sensor2_layout.addLayout(h7)

        btn_layout2 = QHBoxLayout()
        self.btn_connect2 = QPushButton("Подключить")
        self.btn_connect2.clicked.connect(lambda: self.connect_sensor(2))
        btn_layout2.addWidget(self.btn_connect2)

        self.btn_identify2 = QPushButton("ID")
        self.btn_identify2.clicked.connect(lambda: self.identify_sensor(2))
        self.btn_identify2.setEnabled(False)
        btn_layout2.addWidget(self.btn_identify2)
        sensor2_layout.addLayout(btn_layout2)

        # Смена baudrate
        h8 = QHBoxLayout()
        h8.addWidget(QLabel("Новый Baudrate:"))
        self.combo_new_baudrate2 = QComboBox()
        self.combo_new_baudrate2.addItems([str(b) for b in RF603Sensor.BAUDRATES])
        self.combo_new_baudrate2.setCurrentText("9600")
        h8.addWidget(self.combo_new_baudrate2)
        sensor2_layout.addLayout(h8)

        self.btn_change_baudrate2 = QPushButton("Сменить Baudrate")
        self.btn_change_baudrate2.clicked.connect(lambda: self.change_baudrate(2))
        self.btn_change_baudrate2.setEnabled(False)
        sensor2_layout.addWidget(self.btn_change_baudrate2)

        sensor2_group.setLayout(sensor2_layout)
        layout.addWidget(sensor2_group)

        # Общее отключение
        self.btn_disconnect_all = QPushButton("Отключить ОБА датчика")
        self.btn_disconnect_all.clicked.connect(self.disconnect_all)
        self.btn_disconnect_all.setEnabled(False)
        layout.addWidget(self.btn_disconnect_all)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def create_recording_tab(self):
        """Вкладка записи"""
        tab = QWidget()
        layout = QVBoxLayout()

        sensor_group = QGroupBox("Параметры датчиков")
        sensor_layout = QVBoxLayout()

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("Базовое расстояние (мм):"))
        self.spin_base_distance = QSpinBox()
        self.spin_base_distance.setRange(15, 260)
        self.spin_base_distance.setValue(80)
        h1.addWidget(self.spin_base_distance)
        sensor_layout.addLayout(h1)

        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Диапазон (мм):"))
        self.spin_range = QSpinBox()
        self.spin_range.setRange(2, 750)
        self.spin_range.setValue(100)
        h2.addWidget(self.spin_range)
        sensor_layout.addLayout(h2)

        sensor_group.setLayout(sensor_layout)
        layout.addWidget(sensor_group)

        record_group = QGroupBox("Запись данных")
        record_layout = QVBoxLayout()

        self.label_status = QLabel("Статус: Не записывается")
        record_layout.addWidget(self.label_status)

        self.label_points = QLabel("Точек: 0")
        record_layout.addWidget(self.label_points)

        record_group.setLayout(record_layout)
        layout.addWidget(record_group)

        btn_layout = QVBoxLayout()

        self.btn_start_record = QPushButton("▶ Начать синхронную запись")
        self.btn_start_record.clicked.connect(self.start_recording)
        self.btn_start_record.setEnabled(False)
        self.btn_start_record.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.btn_start_record)

        self.btn_stop_record = QPushButton("⏹ Остановить запись")
        self.btn_stop_record.clicked.connect(self.stop_recording)
        self.btn_stop_record.setEnabled(False)
        self.btn_stop_record.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        btn_layout.addWidget(self.btn_stop_record)

        layout.addLayout(btn_layout)

        self.check_auto_analyze = QCheckBox("Автоанализ после записи")
        self.check_auto_analyze.setChecked(True)
        layout.addWidget(self.check_auto_analyze)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_analysis_tab(self):
        """Вкладка анализа"""
        tab = QWidget()
        layout = QVBoxLayout()

        file_group = QGroupBox("Файл данных")
        file_layout = QVBoxLayout()

        h1 = QHBoxLayout()
        self.edit_file = QLineEdit()
        self.edit_file.setPlaceholderText("Выберите файл...")
        h1.addWidget(self.edit_file)
        btn_browse = QPushButton("Обзор...")
        btn_browse.clicked.connect(self.browse_file)
        h1.addWidget(btn_browse)
        file_layout.addLayout(h1)

        btn_load = QPushButton("Загрузить и проанализировать")
        btn_load.clicked.connect(self.load_and_analyze)
        file_layout.addWidget(btn_load)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        results_group = QGroupBox("Результаты анализа")
        results_layout = QVBoxLayout()

        self.table_results = QTableWidget()
        self.table_results.setColumnCount(3)
        self.table_results.setHorizontalHeaderLabels(['Параметр', 'Датчик 1', 'Датчик 2'])
        self.table_results.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.table_results)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)

        btn_export = QPushButton("Экспорт результатов")
        btn_export.clicked.connect(self.export_results)
        layout.addWidget(btn_export)

        btn_save_plot = QPushButton("💾 Сохранить график")
        btn_save_plot.clicked.connect(self.save_plot)
        layout.addWidget(btn_save_plot)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def create_editing_tab(self):
        """Вкладка редактирования"""
        tab = QWidget()
        layout = QVBoxLayout()

        # Пересчет
        recalc_group = QGroupBox("⚡ Обновление результатов")
        recalc_layout = QVBoxLayout()

        btn_recalculate = QPushButton("🔄 ПЕРЕСЧИТАТЬ ХАРАКТЕРИСТИКИ")
        btn_recalculate.clicked.connect(self.manual_recalculate)
        btn_recalculate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 12pt;
                font-weight: bold;
                padding: 12px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        recalc_layout.addWidget(btn_recalculate)

        recalc_group.setLayout(recalc_layout)
        layout.addWidget(recalc_group)

        # Обрезка
        crop_group = QGroupBox("Обрезка данных")
        crop_layout = QVBoxLayout()

        sensor_select_layout = QHBoxLayout()
        sensor_select_layout.addWidget(QLabel("Датчик:"))
        self.combo_sensor_select = QComboBox()
        self.combo_sensor_select.addItems(['Оба', 'Датчик 1', 'Датчик 2'])
        sensor_select_layout.addWidget(self.combo_sensor_select)
        crop_layout.addLayout(sensor_select_layout)

        btn_crop_visual = QPushButton("Выбрать границы на графике")
        btn_crop_visual.setCheckable(True)
        btn_crop_visual.clicked.connect(self.toggle_crop_mode)
        crop_layout.addWidget(btn_crop_visual)
        self.btn_crop_visual = btn_crop_visual

        h1 = QHBoxLayout()
        h1.addWidget(QLabel("От (с):"))
        self.spin_crop_start_time = QDoubleSpinBox()
        self.spin_crop_start_time.setDecimals(6)
        self.spin_crop_start_time.setRange(0, 10000)
        h1.addWidget(self.spin_crop_start_time)
        h1.addWidget(QLabel("До (с):"))
        self.spin_crop_end_time = QDoubleSpinBox()
        self.spin_crop_end_time.setDecimals(6)
        self.spin_crop_end_time.setRange(0, 10000)
        h1.addWidget(self.spin_crop_end_time)
        crop_layout.addLayout(h1)

        btn_apply_crop = QPushButton("Применить обрезку")
        btn_apply_crop.clicked.connect(self.apply_crop)
        crop_layout.addWidget(btn_apply_crop)

        btn_reset_crop = QPushButton("Сбросить к исходным")
        btn_reset_crop.clicked.connect(self.reset_data)
        crop_layout.addWidget(btn_reset_crop)

        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)

        # Пики
        peaks_group = QGroupBox("Управление пиками")
        peaks_layout = QVBoxLayout()

        sensor_select2_layout = QHBoxLayout()
        sensor_select2_layout.addWidget(QLabel("Датчик:"))
        self.combo_peak_sensor = QComboBox()
        self.combo_peak_sensor.addItems(['Датчик 1', 'Датчик 2'])
        sensor_select2_layout.addWidget(self.combo_peak_sensor)
        peaks_layout.addLayout(sensor_select2_layout)

        h3 = QHBoxLayout()
        btn_add_peak = QPushButton("Добавить пик")
        btn_add_peak.setCheckable(True)
        btn_add_peak.clicked.connect(self.toggle_add_peak_mode)
        h3.addWidget(btn_add_peak)
        self.btn_add_peak = btn_add_peak

        btn_remove_peak = QPushButton("Удалить пик")
        btn_remove_peak.setCheckable(True)
        btn_remove_peak.clicked.connect(self.toggle_remove_peak_mode)
        h3.addWidget(btn_remove_peak)
        self.btn_remove_peak = btn_remove_peak

        peaks_layout.addLayout(h3)

        btn_show_auto_peaks = QPushButton("Показать автопики")
        btn_show_auto_peaks.clicked.connect(self.show_auto_peaks)
        peaks_layout.addWidget(btn_show_auto_peaks)

        self.label_peaks_count1 = QLabel("Датчик 1 - Пиков: 0")
        peaks_layout.addWidget(self.label_peaks_count1)

        self.label_peaks_count2 = QLabel("Датчик 2 - Пиков: 0")
        peaks_layout.addWidget(self.label_peaks_count2)

        peaks_group.setLayout(peaks_layout)
        layout.addWidget(peaks_group)

        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def log(self, message):
        """Вывод в консоль"""
        self.console.append(message)
        print(message)

    def refresh_ports(self):
        """Обновление списка портов"""
        self.combo_port1.clear()
        self.combo_port2.clear()

        ports = self.sensor1.list_ports()

        if ports:
            self.combo_port1.addItems(ports)
            self.combo_port2.addItems(ports)
            self.log(f"📡 Найдено портов: {len(ports)}")
        else:
            self.log("⚠️ COM-порты не найдены")

    def connect_sensor(self, sensor_num):
        """Подключение датчика"""
        if sensor_num == 1:
            port = self.combo_port1.currentText()
            baudrate = int(self.combo_baudrate1.currentText())
            address = self.spin_address1.value()
            sensor = self.sensor1
        else:
            port = self.combo_port2.currentText()
            baudrate = int(self.combo_baudrate2.currentText())
            address = self.spin_address2.value()
            sensor = self.sensor2

        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите COM-порт")
            return

        self.log(f"🔌 Подключение Датчик {sensor_num} к {port} ({baudrate} бод)...")

        if sensor.connect(port, baudrate):
            self.log(f"✅ Датчик {sensor_num} подключен")

            if sensor_num == 1:
                self.btn_connect1.setEnabled(False)
                self.btn_identify1.setEnabled(True)
                self.btn_change_baudrate1.setEnabled(True)
                self.combo_port1.setEnabled(False)
                self.combo_baudrate1.setEnabled(False)
            else:
                self.btn_connect2.setEnabled(False)
                self.btn_identify2.setEnabled(True)
                self.btn_change_baudrate2.setEnabled(True)
                self.combo_port2.setEnabled(False)
                self.combo_baudrate2.setEnabled(False)

            if self.sensor1.is_connected and self.sensor2.is_connected:
                self.btn_start_record.setEnabled(True)
                self.btn_disconnect_all.setEnabled(True)
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось подключиться к Датчик {sensor_num}")

    def identify_sensor(self, sensor_num):
        """Идентификация датчика"""
        sensor = self.sensor1 if sensor_num == 1 else self.sensor2
        address = self.spin_address1.value() if sensor_num == 1 else self.spin_address2.value()

        self.log(f"🔍 Идентификация Датчик {sensor_num}...")

        if sensor.identify(address):
            self.log(f"✅ Датчик {sensor_num} идентифицирован")
        else:
            QMessageBox.warning(self, "Ошибка", f"Не удалось идентифицировать Датчик {sensor_num}")

    def change_baudrate(self, sensor_num):
        """Смена baudrate"""
        if sensor_num == 1:
            new_baudrate = int(self.combo_new_baudrate1.currentText())
            sensor = self.sensor1
        else:
            new_baudrate = int(self.combo_new_baudrate2.currentText())
            sensor = self.sensor2

        self.log(f"🔄 Смена baudrate Датчик {sensor_num} на {new_baudrate}...")

        if sensor.change_baudrate(new_baudrate):
            self.log(f"✅ Baudrate изменен для Датчик {sensor_num}")
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось изменить baudrate для Датчик {sensor_num}")

    def disconnect_all(self):
        """Отключение обоих датчиков"""
        self.sensor1.disconnect()
        self.sensor2.disconnect()

        self.log("🔌 Оба датчика отключены")

        self.btn_connect1.setEnabled(True)
        self.btn_identify1.setEnabled(False)
        self.btn_change_baudrate1.setEnabled(False)
        self.combo_port1.setEnabled(True)
        self.combo_baudrate1.setEnabled(True)

        self.btn_connect2.setEnabled(True)
        self.btn_identify2.setEnabled(False)
        self.btn_change_baudrate2.setEnabled(False)
        self.combo_port2.setEnabled(True)
        self.combo_baudrate2.setEnabled(True)

        self.btn_start_record.setEnabled(False)
        self.btn_disconnect_all.setEnabled(False)

    def start_recording(self):
        """Начало синхронной записи"""
        if not (self.sensor1.is_connected and self.sensor2.is_connected):
            QMessageBox.warning(self, "Ошибка", "Оба датчика должны быть подключены")
            return

        address1 = self.spin_address1.value()
        address2 = self.spin_address2.value()

        if not self.sensor1.start_stream(address1):
            QMessageBox.critical(self, "Ошибка", "Не удалось запустить поток Датчик 1")
            return

        if not self.sensor2.start_stream(address2):
            self.sensor1.stop_stream(address1)
            QMessageBox.critical(self, "Ошибка", "Не удалось запустить поток Датчик 2")
            return

        base_distance = self.spin_base_distance.value()
        measurement_range = self.spin_range.value()

        self.show_recording_plot()
        self.plot_recording.clear_plot()
        self.plot_buffer = []

        self.recording_thread = DualSensorRecordingThread(
            self.sensor1, self.sensor2, base_distance, measurement_range
        )
        self.recording_thread.data_received.connect(self.on_data_received)
        self.recording_thread.recording_finished.connect(self.on_recording_finished)
        self.recording_thread.error_occurred.connect(self.on_recording_error)

        self.recording_thread.start()
        self.is_recording = True

        self.update_timer.start(100)

        self.log("🔴 Синхронная запись начата")
        self.label_status.setText("Статус: Идет запись")
        self.btn_start_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)
        self.btn_disconnect_all.setEnabled(False)

    def stop_recording(self):
        """Остановка записи"""
        if self.recording_thread:
            self.recording_thread.stop()
            self.recording_thread.wait()

        address1 = self.spin_address1.value()
        address2 = self.spin_address2.value()
        self.sensor1.stop_stream(address1)
        self.sensor2.stop_stream(address2)

        self.update_timer.stop()
        self.is_recording = False

        self.log("⏹️ Запись остановлена")
        self.label_status.setText("Статус: Остановлена")
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        self.btn_disconnect_all.setEnabled(True)

    def on_data_received(self, distance1, distance2, point, time_val):
        """Обработка данных"""
        self.plot_buffer.append((time_val, distance1, distance2))
        self.label_points.setText(f"Точек: {point}")

    def update_plot_buffer(self):
        """Обновление графика"""
        if self.plot_buffer:
            time_val, distance1, distance2 = self.plot_buffer[-1]
            self.plot_recording.update_plot(time_val, distance1, distance2)

    def on_recording_finished(self, filename):
        """Завершение записи"""
        self.current_file = filename
        self.log(f"✅ Сохранено: {filename}")

        if self.check_auto_analyze.isChecked():
            self.edit_file.setText(filename)
            QTimer.singleShot(500, self.load_and_analyze)

    def on_recording_error(self, error):
        """Ошибка записи"""
        self.log(f"❌ Ошибка: {error}")
        QMessageBox.critical(self, "Ошибка", f"Ошибка записи:\n{error}")

    def browse_file(self):
        """Выбор файла"""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Выберите файл", "", "CSV Files (*.csv);;All Files (*)"
        )
        if filename:
            self.edit_file.setText(filename)

    def load_and_analyze(self):
        """Загрузка и анализ"""
        filename = self.edit_file.text()

        if not filename or not Path(filename).exists():
            QMessageBox.warning(self, "Ошибка", "Выберите файл")
            return

        self.current_file = filename
        self.log(f"📂 Загрузка: {filename}")

        # Загрузка для обоих датчиков
        if not self.analyzer1.load_csv_column(filename, 'Distance_Sensor1_mm'):
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить данные Датчик 1")
            return

        if not self.analyzer2.load_csv_column(filename, 'Distance_Sensor2_mm'):
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить данные Датчик 2")
            return

        self.analyzer1.normalize_data()
        self.analyzer2.normalize_data()

        self.show_analysis_plot()

        # Автообрезка
        self.log("✂️ Автообрезка...")
        success1, period1, frequency1, peaks1 = self.analyzer1.auto_crop_oscillations(1.0)
        success2, period2, frequency2, peaks2 = self.analyzer2.auto_crop_oscillations(1.0)

        if not (success1 and success2):
            QMessageBox.warning(self, "Предупреждение", "Автообрезка не удалась")
            return

        if peaks1 is not None:
            self.analyzer1.calculate_logarithmic_decrement(peaks1)

        if peaks2 is not None:
            self.analyzer2.calculate_logarithmic_decrement(peaks2)

        self.update_analysis_results()
        self.update_plots()

        self.log("✅ Анализ завершен")

    def show_recording_plot(self):
        """Показать график записи"""
        self.plot_recording.show()
        self.plot_analysis.hide()

        while self.right_layout.count():
            self.right_layout.takeAt(0)

        self.right_layout.addWidget(self.plot_recording)

    def show_analysis_plot(self):
        """Показать график анализа"""
        self.plot_recording.hide()
        self.plot_analysis.show()

        while self.right_layout.count():
            self.right_layout.takeAt(0)

        self.right_layout.addWidget(self.plot_analysis)

    def update_plots(self):
        """Обновление графиков анализа"""
        self.plot_analysis.plot_analysis(self.analyzer1, self.analyzer2, show_peaks=True)

        if self.analyzer1.corrected_peaks is not None:
            self.label_peaks_count1.setText(f"Датчик 1 - Пиков: {len(self.analyzer1.corrected_peaks)}")

        if self.analyzer2.corrected_peaks is not None:
            self.label_peaks_count2.setText(f"Датчик 2 - Пиков: {len(self.analyzer2.corrected_peaks)}")

    def update_analysis_results(self):
        """Обновление таблицы результатов"""
        self.table_results.setRowCount(0)

        params = [
            ("Период (T), с", 'current_period', '.6f'),
            ("Частота (f), Гц", 'current_frequency', '.2f'),
            ("Лог. декремент (δ)", 'log_decrement', '.6f'),
            ("Коэф. демпфирования (ζ)", 'damping_ratio', '.6f'),
            ("Коэф. потерь (η)", 'loss_factor', '.6f'),
        ]

        for i, (param_name, attr, fmt) in enumerate(params):
            self.table_results.insertRow(i)
            self.table_results.setItem(i, 0, QTableWidgetItem(param_name))

            val1 = getattr(self.analyzer1, attr)
            val2 = getattr(self.analyzer2, attr)

            text1 = f"{val1:{fmt}}" if val1 is not None else "—"
            text2 = f"{val2:{fmt}}" if val2 is not None else "—"

            self.table_results.setItem(i, 1, QTableWidgetItem(text1))
            self.table_results.setItem(i, 2, QTableWidgetItem(text2))

        self.table_results.resizeColumnsToContents()

    def toggle_crop_mode(self, checked):
        """Переключение режима обрезки"""
        sensor = self.combo_sensor_select.currentText()

        if sensor == 'Датчик 1':
            sensor_num = 1
        elif sensor == 'Датчик 2':
            sensor_num = 2
        else:
            sensor_num = 0  # Оба

        self.plot_analysis.set_crop_mode(checked, sensor_num if checked else None)

        if checked:
            self.log(f"✂️ Режим обрезки для {sensor}")
        else:
            self.log("✂️ Режим обрезки выключен")

    def toggle_add_peak_mode(self, checked):
        """Режим добавления пика"""
        sensor = self.combo_peak_sensor.currentText()
        sensor_num = 1 if sensor == 'Датчик 1' else 2

        self.plot_analysis.set_add_peak_mode(checked, sensor_num if checked else None)

        if checked:
            self.btn_remove_peak.setChecked(False)
            self.plot_analysis.set_remove_peak_mode(False, None)
            self.log(f"📍 Добавление пиков для {sensor}")
        else:
            self.log("📍 Режим добавления выключен")

    def toggle_remove_peak_mode(self, checked):
        """Режим удаления пика"""
        sensor = self.combo_peak_sensor.currentText()
        sensor_num = 1 if sensor == 'Датчик 1' else 2

        self.plot_analysis.set_remove_peak_mode(checked, sensor_num if checked else None)

        if checked:
            self.btn_add_peak.setChecked(False)
            self.plot_analysis.set_add_peak_mode(False, None)
            self.log(f"🗑️ Удаление пиков для {sensor}")
        else:
            self.log("🗑️ Режим удаления выключен")

    def show_auto_peaks(self):
        """Показать автоматически найденные пики"""
        if self.analyzer1.processed_data is None or self.analyzer2.processed_data is None:
            return

        sensor = self.combo_peak_sensor.currentText()

        if sensor == 'Датчик 1':
            period, frequency, peaks = self.analyzer1.calculate_period_frequency_improved()
            if peaks is not None:
                self.plot_analysis.peaks1 = peaks.tolist()
                self.label_peaks_count1.setText(f"Датчик 1 - Пиков: {len(peaks)}")
                self.log(f"✅ Найдено пиков для Датчик 1: {len(peaks)}")
        else:
            period, frequency, peaks = self.analyzer2.calculate_period_frequency_improved()
            if peaks is not None:
                self.plot_analysis.peaks2 = peaks.tolist()
                self.label_peaks_count2.setText(f"Датчик 2 - Пиков: {len(peaks)}")
                self.log(f"✅ Найдено пиков для Датчик 2: {len(peaks)}")

        self.plot_analysis.update_peaks_display()
        self.log("💡 Нажмите '🔄 ПЕРЕСЧИТАТЬ' для обновления результатов")

    def apply_crop(self):
        """Применить обрезку"""
        if self.analyzer1.processed_data is None or self.analyzer2.processed_data is None:
            return

        sensor = self.combo_sensor_select.currentText()
        start_val, end_val = self.plot_analysis.get_crop_values()

        if start_val is not None and end_val is not None:
            self.log(f"✂️ Обрезка {sensor}: {start_val:.6f} - {end_val:.6f}")

            if sensor == 'Оба' or sensor == 'Датчик 1':
                self.analyzer1.crop_by_time(start_val, end_val)

            if sensor == 'Оба' or sensor == 'Датчик 2':
                self.analyzer2.crop_by_time(start_val, end_val)

        elif self.spin_crop_start_time.value() > 0 or self.spin_crop_end_time.value() > 0:
            start_t = self.spin_crop_start_time.value()
            end_t = self.spin_crop_end_time.value()

            self.log(f"✂️ Обрезка {sensor}: {start_t} - {end_t} с")

            if sensor == 'Оба' or sensor == 'Датчик 1':
                self.analyzer1.crop_by_time(start_t, end_t)

            if sensor == 'Оба' or sensor == 'Датчик 2':
                self.analyzer2.crop_by_time(start_t, end_t)
        else:
            QMessageBox.warning(self, "Ошибка", "Задайте границы обрезки")
            return

        self.plot_analysis.clear_crop_lines()
        self.plot_analysis.peaks1 = []
        self.plot_analysis.peaks2 = []

        self.update_plots()
        self.log("💡 Пики сброшены. Нажмите 'Показать автопики'")

    def reset_data(self):
        """Сброс к исходным данным"""
        if self.analyzer1.reset_to_original() and self.analyzer2.reset_to_original():
            self.plot_analysis.clear_crop_lines()
            self.plot_analysis.peaks1 = []
            self.plot_analysis.peaks2 = []
            self.update_plots()
            self.log("✅ Данные сброшены")

    def manual_recalculate(self):
        """Ручной пересчет"""
        if self.analyzer1.processed_data is None or self.analyzer2.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для пересчета")
            return

        if len(self.plot_analysis.peaks1) < 2 and len(self.plot_analysis.peaks2) < 2:
            QMessageBox.warning(self, "Ошибка", "Нужно минимум 2 пика для каждого датчика")
            return

        self.log("🔄 Пересчет характеристик...")

        if len(self.plot_analysis.peaks1) >= 2:
            self.analyzer1.set_manual_peaks(self.plot_analysis.peaks1)
            period1, frequency1, peaks1 = self.analyzer1.calculate_period_frequency_improved()
            if peaks1 is not None:
                self.analyzer1.calculate_logarithmic_decrement(peaks1)

        if len(self.plot_analysis.peaks2) >= 2:
            self.analyzer2.set_manual_peaks(self.plot_analysis.peaks2)
            period2, frequency2, peaks2 = self.analyzer2.calculate_period_frequency_improved()
            if peaks2 is not None:
                self.analyzer2.calculate_logarithmic_decrement(peaks2)

        self.update_analysis_results()
        self.update_plots()

        self.log("✅ Характеристики пересчитаны!")
        QMessageBox.information(self, "Успех", "Характеристики успешно пересчитаны!")

    def save_plot(self):
        """Сохранение графика"""
        if self.analyzer1.processed_data is None or self.analyzer2.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if filename:
            if self.plot_analysis.save_figure(filename):
                self.log(f"✅ График сохранен: {filename}")
                QMessageBox.information(self, "Успех", f"График сохранен:\n{filename}")
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось сохранить график")

    def export_results(self):
        """Экспорт результатов"""
        if self.analyzer1.processed_data is None or self.analyzer2.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для экспорта")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "", "Text Files (*.txt);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("LOGYZE ADVANCED - Результаты анализа двух датчиков RF603HS\n")
                    f.write("=" * 80 + "\n\n")

                    f.write(f"Файл данных: {self.current_file}\n")
                    f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    f.write("-" * 80 + "\n")
                    f.write(f"{'Параметр':<30} {'Датчик 1':<20} {'Датчик 2':<20}\n")
                    f.write("-" * 80 + "\n")

                    if self.analyzer1.current_period:
                        f.write(f"{'Период (T), с':<30} {self.analyzer1.current_period:<20.6f} {self.analyzer2.current_period:<20.6f}\n")

                    if self.analyzer1.current_frequency:
                        f.write(f"{'Частота (f), Гц':<30} {self.analyzer1.current_frequency:<20.2f} {self.analyzer2.current_frequency:<20.2f}\n")

                    if self.analyzer1.log_decrement:
                        f.write(f"{'Лог. декремент (δ)':<30} {self.analyzer1.log_decrement:<20.6f} {self.analyzer2.log_decrement:<20.6f}\n")

                    if self.analyzer1.damping_ratio:
                        f.write(f"{'Коэф. демпфирования (ζ)':<30} {self.analyzer1.damping_ratio:<20.6f} {self.analyzer2.damping_ratio:<20.6f}\n")

                    if self.analyzer1.loss_factor:
                        f.write(f"{'Коэф. потерь (η)':<30} {self.analyzer1.loss_factor:<20.6f} {self.analyzer2.loss_factor:<20.6f}\n")

                    f.write("\n" + "=" * 80 + "\n")

                self.log(f"✅ Результаты сохранены: {filename}")
                QMessageBox.information(self, "Успех", f"Результаты сохранены:\n{filename}")

            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения:\n{e}")


# ============================================================================
# ЗАПУСК
# ============================================================================

def main():
    app = QApplication(sys.argv)

    font = QFont("Segoe UI", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
