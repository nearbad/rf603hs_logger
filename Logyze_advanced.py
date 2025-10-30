# -*- coding: utf-8 -*-
"""
Logyze Advanced - Анализ затухающих колебаний с ДВУМЯ датчиками RF603HS
Версия 3.1 - Синхронная работа с двумя датчиками, два независимых графика анализа
"""

import sys
import serial
import serial.tools.list_ports
import struct
import time
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QTextEdit, QFileDialog,
    QGroupBox, QSpinBox, QDoubleSpinBox, QMessageBox, QSplitter,
    QTabWidget, QCheckBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QRadioButton
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
# КЛАСС АНАЛИЗАТОРА (из dekrement.py)
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
        
    def load_csv(self, filename, sensor_column=None):
        """
        Загрузка CSV файла с данными
        sensor_column: None для обычных файлов, 'Distance_Sensor1_mm' или 'Distance_Sensor2_mm' для dual файлов
        """
        try:
            for delimiter in [',', ';']:
                try:
                    self.data = pd.read_csv(filename, delimiter=delimiter, encoding='utf-8')
                    if len(self.data.columns) >= 2:
                        break
                except:
                    continue

            # Если указана конкретная колонка датчика (для dual sensor файлов)
            if sensor_column and sensor_column in self.data.columns:
                # Dual sensor файл
                distance_col = sensor_column
                time_col = 'Time_s' if 'Time_s' in self.data.columns else 'Временная_метка'

                self.data = pd.DataFrame({
                    'Расстояние_мм': self.data[distance_col],
                    'Временная_метка': self.data[time_col],
                    'Point': range(len(self.data))
                })
            else:
                # Обычный single sensor файл
                if 'Расстояние_мм' not in self.data.columns:
                    if 'Distance_mm' in self.data.columns:
                        self.data.rename(columns={'Distance_mm': 'Расстояние_мм'}, inplace=True)
                    elif len(self.data.columns) >= 1:
                        self.data.rename(columns={self.data.columns[0]: 'Расстояние_мм'}, inplace=True)

                if 'Временная_метка' not in self.data.columns:
                    if 'Time_s' in self.data.columns:
                        self.data.rename(columns={'Time_s': 'Временная_метка'}, inplace=True)
                    elif len(self.data.columns) >= 3:
                        self.data.rename(columns={self.data.columns[2]: 'Временная_метка'}, inplace=True)

                if 'Point' not in self.data.columns:
                    if len(self.data.columns) >= 2:
                        col_name = self.data.columns[1]
                        if col_name not in ['Расстояние_мм', 'Временная_метка']:
                            self.data.rename(columns={col_name: 'Point'}, inplace=True)
                        else:
                            self.data['Point'] = range(len(self.data))
                    else:
                        self.data['Point'] = range(len(self.data))

            print(f"✅ Данные загружены: {len(self.data)} строк")
            print(f"📏 Диапазон: {self.data['Расстояние_мм'].min():.3f} - {self.data['Расстояние_мм'].max():.3f} мм")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            import traceback
            traceback.print_exc()
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
            
            print(f"✅ Нормировка выполнена")
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
            
            print("✅ Данные сброшены к исходным")
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
            
            original_count = len(self.processed_data)
            self.processed_data = self.processed_data.iloc[start_idx:end_idx + 1].reset_index(drop=True)
            self.oscillation_start = start_idx
            self.oscillation_end = end_idx
            
            # Сбрасываем пики и характеристики при обрезке
            self.corrected_peaks = None
            self.current_period = None
            self.current_frequency = None
            self.log_decrement = None
            self.loss_factor = None
            self.damping_ratio = None
            
            print(f"✅ Обрезка: точки {start_idx}-{end_idx} ({len(self.processed_data)} точек)")
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
                release_point = max(0, i - 5)
                print(f"📍 Начало колебаний: точка {release_point}")
                return release_point
        
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
            
            print(f"📍 Обрезка: {start_time:.3f} - {end_time:.3f} сек")
            
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
        print(f"✅ Установлено пиков: {len(peaks)}")
    
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
                print("❌ Недостаточно пиков")
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
            
            print(f"⏱️ Период: {avg_period:.6f} с, Частота: {frequency:.2f} Гц, Пиков: {len(peaks)}")
            
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
            
            print(f"📉 Декремент: {avg_decrement:.6f}, Демпфирование: {damping_ratio:.6f}, Потери: {loss_factor:.6f}")
            
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
                # Декодируем ответ идентификации
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
            # Очистка буфера перед запуском потока
            self.serial_port.reset_input_buffer()
            time.sleep(0.05)

            # Запуск потока данных
            self.send_request(address, 0x07)
            time.sleep(0.1)  # Даем датчику время начать передачу

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
        """
        Поиск начала пакета данных
        Ищет два последовательных байта со старшим битом = 1
        """
        start_time = time.time()
        byte_buffer = []

        while (time.time() - start_time) < timeout:
            if self.serial_port.in_waiting > 0:
                byte = self.serial_port.read(1)
                if len(byte) > 0:
                    byte_buffer.append(byte[0])

                    # Ищем последовательность из двух байтов с битом 0x80
                    if len(byte_buffer) >= 2:
                        if (byte_buffer[-2] & 0x80) and (byte_buffer[-1] & 0x80):
                            # Проверяем, что это похоже на начало пакета данных
                            sb = (byte_buffer[-2] >> 6) & 0x01  # Status bit
                            cnt = (byte_buffer[-2] >> 4) & 0x03  # Counter
                            # Если это выглядит как данные, возвращаем эти два байта
                            return bytes(byte_buffer[-2:])

                        # Сохраняем только последний байт для следующей проверки
                        byte_buffer = byte_buffer[-1:]
            else:
                time.sleep(0.001)

        return None

    def read_measurement(self):
        """
        Чтение измерения с проверкой формата пакета
        Возвращает (value, valid) где valid - True если пакет корректный
        """
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
        """
        Декодирование пакета измерения из двух байтов
        Формат согласно документации (п. 11.5.5):
        Байт 0: 1 SB CNT(1:0) DAT(3:0)
        Байт 1: 1 SB CNT(1:0) DAT(7:4)

        Возвращает: (value, status_bit, counter, valid)
        """
        # Проверка, что оба байта имеют старший бит
        if not ((byte1 & 0x80) and (byte2 & 0x80)):
            return None, None, None, False

        # Извлечение полей из первого байта
        sb1 = (byte1 >> 6) & 0x01  # Status bit
        cnt1 = (byte1 >> 4) & 0x03  # Counter
        low_nibble = byte1 & 0x0F  # Младшие 4 бита значения

        # Извлечение полей из второго байта
        sb2 = (byte2 >> 6) & 0x01
        cnt2 = (byte2 >> 4) & 0x03
        high_nibble = byte2 & 0x0F  # Старшие 4 бита значения

        # Проверка согласованности
        if sb1 != sb2 or cnt1 != cnt2:
            return None, None, None, False

        # Сборка значения
        value = low_nibble | (high_nibble << 4)

        return value, sb1, cnt1, True

    def _decode_byte(self, data):
        """Декодирование одного байта из двух байтов протокола"""
        if len(data) >= 2:
            if (data[0] & 0x80) and (data[1] & 0x80):
                low = data[0] & 0x0F
                high = data[1] & 0x0F
                return low | (high << 4)
        return 0

    def _decode_word(self, data):
        """Декодирование слова (word) из четырех байтов протокола"""
        if len(data) >= 4:
            byte1 = self._decode_byte(data[0:2])
            byte2 = self._decode_byte(data[2:4])
            return byte1 | (byte2 << 8)
        return 0


# ============================================================================
# ПОТОК ЗАПИСИ
# ============================================================================

class RecordingThread(QThread):
    """Поток для записи данных"""
    
    data_received = pyqtSignal(float, int, float)
    recording_finished = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, sensor, base_distance, measurement_range):
        super().__init__()
        self.sensor = sensor
        self.base_distance = base_distance
        self.measurement_range = measurement_range
        self.running = False
        self.data_list = []
        
    def run(self):
        """Основной цикл записи"""
        self.running = True
        self.data_list = []

        try:
            # === СИНХРОНИЗАЦИЯ ===
            # Ищем начало пакета перед началом записи
            print("🔍 Поиск начала пакета...")
            initial_packet = self.sensor.find_packet_start(timeout=3.0)

            if initial_packet is None:
                self.error_occurred.emit("Не удалось синхронизироваться с потоком данных")
                return

            print("✅ Синхронизация успешна, начинаем запись...")

            # === ОСНОВНОЙ ЦИКЛ ЗАПИСИ ===
            start_time = time.perf_counter()
            point = 0

            # Обрабатываем первый пакет, полученный при синхронизации
            byte1, byte2 = initial_packet[0], initial_packet[1]
            value, status_bit, counter, valid = self.sensor._decode_measurement_packet(byte1, byte2)

            if valid and value is not None and value > 0:
                distance = self.base_distance + (value * self.measurement_range / 16384.0)
                current_time = time.perf_counter() - start_time

                self.data_list.append({
                    'Distance_mm': distance,
                    'Point': point,
                    'Time_s': current_time
                })

                self.data_received.emit(distance, point, current_time)
                point += 1

            # Продолжаем читать данные
            while self.running:
                value = self.sensor.read_measurement()

                if value is not None and value > 0:
                    distance = self.base_distance + (value * self.measurement_range / 16384.0)
                    current_time = time.perf_counter() - start_time

                    self.data_list.append({
                        'Distance_mm': distance,
                        'Point': point,
                        'Time_s': current_time
                    })

                    self.data_received.emit(distance, point, current_time)
                    point += 1

                time.sleep(0.0001)
            
            # Сохранение
            if self.data_list:
                filename = f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['Distance_mm', 'Point', 'Time_s'])
                    writer.writeheader()
                    writer.writerows(self.data_list)
                
                self.recording_finished.emit(filename)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def stop(self):
        """Остановка записи"""
        self.running = False


# ============================================================================
# ПОТОК ЗАПИСИ ДВУХ ДАТЧИКОВ (СИНХРОННО)
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
        """Основной цикл синхронной записи"""
        self.running = True
        self.data_list = []

        try:
            # Синхронизация обоих датчиков
            print("🔍 Синхронизация датчика 1...")
            packet1 = self.sensor1.find_packet_start(timeout=3.0)

            print("🔍 Синхронизация датчика 2...")
            packet2 = self.sensor2.find_packet_start(timeout=3.0)

            if packet1 is None or packet2 is None:
                self.error_occurred.emit("Не удалось синхронизироваться с одним или обоими датчиками")
                return

            print("✅ Оба датчика синхронизированы, начинаем запись...")

            start_time = time.perf_counter()
            point = 0

            # Обработка первых пакетов
            byte1_s1, byte2_s1 = packet1[0], packet1[1]
            value1, _, _, valid1 = self.sensor1._decode_measurement_packet(byte1_s1, byte2_s1)

            byte1_s2, byte2_s2 = packet2[0], packet2[1]
            value2, _, _, valid2 = self.sensor2._decode_measurement_packet(byte1_s2, byte2_s2)

            if valid1 and valid2 and value1 is not None and value2 is not None and value1 > 0 and value2 > 0:
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
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))

    def stop(self):
        """Остановка записи"""
        self.running = False


# ============================================================================
# ВИДЖЕТ ГРАФИКА
# ============================================================================

class InteractivePlotWidget(QWidget):
    """Виджет с интерактивным графиком"""
    
    def __init__(self, plot_type='time'):
        super().__init__()
        
        self.plot_type = plot_type
        self.times = []
        self.distances = []
        self.points = []
        self.peaks = []
        self.analyzer = None  # Сохраняем analyzer для обновления результатов
        
        self.crop_mode = False
        self.add_peak_mode = False
        self.remove_peak_mode = False
        
        self.crop_start_val = None
        self.crop_end_val = None
        self.crop_start_line = None
        self.crop_end_line = None
        self.dragging_line = None
        
        self.peak_markers = []
        self.peak_texts = []
        
        self.init_ui()
        
    def init_ui(self):
        """Инициализация интерфейса"""
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # График
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        
        # Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        
        # Метка координат
        self.coord_label = QLabel("Координаты: -")
        layout.addWidget(self.coord_label)
        
        # События
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        
        self.ax.grid(True, alpha=0.3)
        
    def on_click(self, event):
        """Обработка клика"""
        if event.inaxes != self.ax:
            return
        
        x, y = event.xdata, event.ydata
        
        # Режим обрезки
        if self.crop_mode:
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
        if self.add_peak_mode and self.plot_type == 'time':
            if len(self.times) > 0:
                idx = np.argmin(np.abs(np.array(self.times) - x))
                if idx not in self.peaks:
                    self.peaks.append(idx)
                    self.peaks.sort()
                    self.update_peaks_display()
                    print(f"✅ Пик добавлен. Всего пиков: {len(self.peaks)}")
                    print("💡 Нажмите '🔄 ПЕРЕСЧИТАТЬ' для обновления результатов")
            return
        
        # Режим удаления пика
        if self.remove_peak_mode and self.plot_type == 'time':
            if len(self.peaks) > 0 and len(self.times) > 0:
                idx = np.argmin(np.abs(np.array(self.times) - x))
                if idx in self.peaks:
                    self.peaks.remove(idx)
                    self.update_peaks_display()
                    print(f"✅ Пик удален. Осталось пиков: {len(self.peaks)}")
                    print("💡 Нажмите '🔄 ПЕРЕСЧИТАТЬ' для обновления результатов")
            return
        
        # Проверка на перетаскивание границ
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
        
        # Отображение координат
        if self.plot_type == 'time':
            self.coord_label.setText(f"Время: {x:.6f} с, Расстояние: {y:.6f} мм")
        else:
            self.coord_label.setText(f"Точка: {int(x)}, Расстояние: {y:.6f} мм")
        
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
        elif self.crop_mode or self.add_peak_mode or self.remove_peak_mode:
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))
    
    def set_crop_mode(self, enabled):
        """Включение/выключение режима обрезки"""
        self.crop_mode = enabled
        if enabled:
            self.add_peak_mode = False
            self.remove_peak_mode = False
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))
    
    def set_add_peak_mode(self, enabled):
        """Режим добавления пика"""
        self.add_peak_mode = enabled
        if enabled:
            self.crop_mode = False
            self.remove_peak_mode = False
            self.canvas.setCursor(QCursor(Qt.CrossCursor))
        else:
            self.canvas.setCursor(QCursor(Qt.ArrowCursor))
    
    def set_remove_peak_mode(self, enabled):
        """Режим удаления пика"""
        self.remove_peak_mode = enabled
        if enabled:
            self.crop_mode = False
            self.add_peak_mode = False
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
    
    def update_plot(self, time_val, distance):
        """Обновление графика в реальном времени"""
        self.times.append(time_val)
        self.distances.append(distance)
        self.points.append(len(self.points))
        
        max_points = 5000
        if len(self.times) > max_points:
            self.times = self.times[-max_points:]
            self.distances = self.distances[-max_points:]
            self.points = self.points[-max_points:]
        
        self.ax.clear()
        
        if self.plot_type == 'time':
            self.ax.plot(self.times, self.distances, 'b-', linewidth=0.5)
            self.ax.set_xlabel('Время (с)')
            self.ax.set_title(f'Расстояние от времени ({len(self.times)} точек)')
        else:
            self.ax.plot(self.points, self.distances, 'b-', linewidth=0.5)
            self.ax.set_xlabel('Номер точки')
            self.ax.set_title(f'Расстояние от номера точки ({len(self.points)} точек)')
        
        self.ax.set_ylabel('Расстояние (мм)')
        self.ax.grid(True, alpha=0.3)
        
        self.canvas.draw()
    
    def clear_plot(self):
        """Очистка графика"""
        self.times = []
        self.distances = []
        self.points = []
        self.peaks = []
        self.clear_crop_lines()
        
        self.ax.clear()
        if self.plot_type == 'time':
            self.ax.set_xlabel('Время (с)')
            self.ax.set_title('Расстояние от времени')
        else:
            self.ax.set_xlabel('Номер точки')
            self.ax.set_title('Расстояние от номера точки')
        self.ax.set_ylabel('Расстояние (мм)')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def plot_analysis(self, analyzer, show_peaks=False):
        """Отображение результатов анализа"""
        if analyzer.processed_data is None:
            return
        
        # Сохраняем analyzer для последующих обновлений
        self.analyzer = analyzer
        
        self.times = analyzer.processed_data['Временная_метка'].values.tolist()
        self.distances = analyzer.processed_data['Расстояние_норм'].values.tolist()
        self.points = analyzer.processed_data['Point'].values.tolist()
        
        self.ax.clear()
        
        if self.plot_type == 'time':
            self.ax.plot(self.times, self.distances, 'b-', linewidth=1, label='Данные')
            self.ax.set_xlabel('Время (с)')
            self.ax.set_title('Расстояние от времени')
            
            # Пики только на графике времени
            if show_peaks and analyzer.corrected_peaks is not None:
                self.peaks = analyzer.corrected_peaks.tolist()
                self.update_peaks_display()
            
            # ДОБАВЛЯЕМ ТЕКСТОВЫЙ БЛОК С РЕЗУЛЬТАТАМИ
            self._add_results_text(analyzer)
            
        else:
            self.ax.plot(self.points, self.distances, 'b-', linewidth=1, label='Данные')
            self.ax.set_xlabel('Номер точки')
            self.ax.set_title('Расстояние от номера точки')
        
        self.ax.set_ylabel('Расстояние (мм)')
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        self.canvas.draw()
    
    def _add_results_text(self, analyzer):
        """Добавление текстового блока с результатами на график"""
        info_text = ''
        
        # Период
        if analyzer.current_period is not None:
            info_text += f'Период (T): {analyzer.current_period:.6f} с\n'
        
        # Частота
        if analyzer.current_frequency is not None:
            info_text += f'Частота (f): {analyzer.current_frequency:.2f} Гц\n'
        
        # Логарифмический декремент
        if analyzer.log_decrement is not None:
            info_text += f'Лог. декремент (δ): {analyzer.log_decrement:.6f}\n'
        
        # Коэффициент демпфирования
        if analyzer.damping_ratio is not None:
            info_text += f'Коэф. демпфирования (ζ): {analyzer.damping_ratio:.6f}\n'
        
        # Коэффициент потерь
        if analyzer.loss_factor is not None:
            info_text += f'Коэф. потерь (η): {analyzer.loss_factor:.6f}\n'
        
        # Примечание об исправленных пиках
        if analyzer.corrected_peaks is not None:
            info_text += '\n(исправленные пики)'
        
        # Отображение текста на графике
        if info_text:
            # Используем светло-голубой фон с прозрачностью
            self.ax.text(0.02, 0.98, info_text.strip(), 
                        transform=self.ax.transAxes, 
                        verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', 
                                 facecolor='lightblue', 
                                 alpha=0.85,
                                 edgecolor='gray',
                                 linewidth=1.5),
                        fontsize=9,
                        family='monospace')
    
    def update_peaks_display(self):
        """Обновление отображения пиков"""
        # Очищаем старые маркеры (не используем remove() - вызывает ошибку)
        self.peak_markers = []
        self.peak_texts = []
        
        # Перерисовываем весь график заново
        if len(self.peaks) > 0 and len(self.times) > 0:
            # Сохраняем текущие лимиты
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            
            # Очищаем и перерисовываем
            self.ax.clear()
            
            # Перерисовываем основную линию
            if self.plot_type == 'time':
                self.ax.plot(self.times, self.distances, 'b-', linewidth=1, label='Данные')
                self.ax.set_xlabel('Время (с)')
                self.ax.set_title('Расстояние от времени')
            else:
                self.ax.plot(self.points, self.distances, 'b-', linewidth=1, label='Данные')
                self.ax.set_xlabel('Номер точки')
                self.ax.set_title('Расстояние от номера точки')
            
            self.ax.set_ylabel('Расстояние (мм)')
            self.ax.grid(True, alpha=0.3)
            
            # Добавляем пики
            times_arr = np.array(self.times)
            distances_arr = np.array(self.distances)
            
            for i, peak_idx in enumerate(self.peaks):
                if peak_idx < len(times_arr):
                    x = times_arr[peak_idx]
                    y = distances_arr[peak_idx]
                    
                    marker, = self.ax.plot(x, y, 'ro', markersize=8, label='Пики' if i == 0 else '')
                    text = self.ax.text(x, y, f'  {i+1}', fontsize=9, color='red')
                    
                    self.peak_markers.append(marker)
                    self.peak_texts.append(text)
            
            # ВАЖНО: Восстанавливаем текстовый блок с результатами
            if self.analyzer is not None and self.plot_type == 'time':
                self._add_results_text(self.analyzer)
            
            # Восстанавливаем лимиты
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
            self.ax.legend()
        
        self.canvas.draw()
    
    def get_crop_values(self):
        """Получить значения границ обрезки"""
        return self.crop_start_val, self.crop_end_val
    
    def set_crop_lines(self, start, end):
        """Установить линии обрезки"""
        self.clear_crop_lines()
        
        if start is not None:
            self.crop_start_val = start
            self.crop_start_line = self.ax.axvline(start, color='green', linestyle='--', linewidth=2, label='Начало')
        
        if end is not None:
            self.crop_end_val = end
            self.crop_end_line = self.ax.axvline(end, color='red', linestyle='--', linewidth=2, label='Конец')
        
        if start is not None or end is not None:
            self.ax.legend()
        
        self.canvas.draw()
    
    def save_figure(self, filename):
        """Сохранение графика в файл"""
        try:
            self.figure.savefig(filename, dpi=300, bbox_inches='tight')
            return True
        except Exception as e:
            print(f"❌ Ошибка сохранения графика: {e}")
            return False


# ============================================================================
# ГЛАВНОЕ ОКНО
# ============================================================================

class MainWindow(QMainWindow):
    """Главное окно Logyze"""
    
    def __init__(self):
        super().__init__()

        # ДВА датчика
        self.sensor1 = RF603Sensor()
        self.sensor2 = RF603Sensor()

        # ДВА анализатора
        self.analyzer1 = RF603OscillationAnalyzer()
        self.analyzer2 = RF603OscillationAnalyzer()

        self.recording_thread = None
        self.is_recording = False
        self.current_file = None

        # Активный датчик для редактирования (1 или 2)
        self.active_sensor = 1

        self.init_ui()

        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_plot_buffer)
        self.plot_buffer = []
        
    def init_ui(self):
        """Инициализация интерфейса"""
        self.setWindowTitle('Logyze Advanced - Анализатор с ДВУМЯ датчиками RF603HS v3.1')
        self.setGeometry(100, 100, 1800, 900)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Левая панель
        left_panel = self.create_control_panel()
        
        # Правая панель (графики)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout()
        self.right_panel.setLayout(self.right_layout)
        
        # График для записи (один, показывает среднее значение)
        self.plot_recording = InteractivePlotWidget('time')
        self.right_layout.addWidget(self.plot_recording)

        # Графики для анализа - ДВА графика времени для двух датчиков (скрыты по умолчанию)
        self.plot_time_sensor1 = InteractivePlotWidget('time')
        self.plot_time_sensor2 = InteractivePlotWidget('time')
        self.plot_time_sensor1.hide()
        self.plot_time_sensor2.hide()
        
        # Splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        
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
        self.console.setMaximumHeight(150)
        console_layout.addWidget(self.console)
        
        btn_clear = QPushButton("Очистить")
        btn_clear.clicked.connect(self.console.clear)
        console_layout.addWidget(btn_clear)
        
        console_group.setLayout(console_layout)
        layout.addWidget(console_group)
        
        return panel
    
    def create_connection_tab(self):
        """Вкладка подключения для двух датчиков"""
        tab = QWidget()
        main_layout = QVBoxLayout()

        # Кнопка обновления портов (общая для обоих датчиков)
        btn_refresh = QPushButton("🔄 Обновить список портов")
        btn_refresh.clicked.connect(self.refresh_ports)
        main_layout.addWidget(btn_refresh)

        # ДАТЧИК 1
        sensor1_group = QGroupBox("⚙️ ДАТЧИК 1")
        sensor1_layout = QVBoxLayout()

        h1_1 = QHBoxLayout()
        h1_1.addWidget(QLabel("COM-порт:"))
        self.combo_port1 = QComboBox()
        h1_1.addWidget(self.combo_port1)
        sensor1_layout.addLayout(h1_1)

        h1_2 = QHBoxLayout()
        h1_2.addWidget(QLabel("Baud Rate:"))
        self.combo_baudrate1 = QComboBox()
        self.combo_baudrate1.addItems([str(b) for b in RF603Sensor.BAUDRATES])
        self.combo_baudrate1.setCurrentText("9600")
        h1_2.addWidget(self.combo_baudrate1)
        sensor1_layout.addLayout(h1_2)

        h1_3 = QHBoxLayout()
        h1_3.addWidget(QLabel("Адрес:"))
        self.spin_address1 = QSpinBox()
        self.spin_address1.setRange(1, 127)
        self.spin_address1.setValue(1)
        h1_3.addWidget(self.spin_address1)
        sensor1_layout.addLayout(h1_3)

        self.btn_connect1 = QPushButton("Подключить датчик 1")
        self.btn_connect1.clicked.connect(lambda: self.connect_sensor(1))
        sensor1_layout.addWidget(self.btn_connect1)

        self.btn_identify1 = QPushButton("Идентификация датчика 1")
        self.btn_identify1.clicked.connect(lambda: self.identify_sensor(1))
        self.btn_identify1.setEnabled(False)
        sensor1_layout.addWidget(self.btn_identify1)

        self.btn_change_baudrate1 = QPushButton("Изменить Baud Rate")
        self.btn_change_baudrate1.clicked.connect(lambda: self.change_baudrate(1))
        self.btn_change_baudrate1.setEnabled(False)
        sensor1_layout.addWidget(self.btn_change_baudrate1)

        self.btn_disconnect1 = QPushButton("Отключить датчик 1")
        self.btn_disconnect1.clicked.connect(lambda: self.disconnect_sensor(1))
        self.btn_disconnect1.setEnabled(False)
        sensor1_layout.addWidget(self.btn_disconnect1)

        self.label_status1 = QLabel("Статус: Не подключен")
        sensor1_layout.addWidget(self.label_status1)

        sensor1_group.setLayout(sensor1_layout)
        main_layout.addWidget(sensor1_group)

        # ДАТЧИК 2
        sensor2_group = QGroupBox("⚙️ ДАТЧИК 2")
        sensor2_layout = QVBoxLayout()

        h2_1 = QHBoxLayout()
        h2_1.addWidget(QLabel("COM-порт:"))
        self.combo_port2 = QComboBox()
        h2_1.addWidget(self.combo_port2)
        sensor2_layout.addLayout(h2_1)

        h2_2 = QHBoxLayout()
        h2_2.addWidget(QLabel("Baud Rate:"))
        self.combo_baudrate2 = QComboBox()
        self.combo_baudrate2.addItems([str(b) for b in RF603Sensor.BAUDRATES])
        self.combo_baudrate2.setCurrentText("9600")
        h2_2.addWidget(self.combo_baudrate2)
        sensor2_layout.addLayout(h2_2)

        h2_3 = QHBoxLayout()
        h2_3.addWidget(QLabel("Адрес:"))
        self.spin_address2 = QSpinBox()
        self.spin_address2.setRange(1, 127)
        self.spin_address2.setValue(1)
        h2_3.addWidget(self.spin_address2)
        sensor2_layout.addLayout(h2_3)

        self.btn_connect2 = QPushButton("Подключить датчик 2")
        self.btn_connect2.clicked.connect(lambda: self.connect_sensor(2))
        sensor2_layout.addWidget(self.btn_connect2)

        self.btn_identify2 = QPushButton("Идентификация датчика 2")
        self.btn_identify2.clicked.connect(lambda: self.identify_sensor(2))
        self.btn_identify2.setEnabled(False)
        sensor2_layout.addWidget(self.btn_identify2)

        self.btn_change_baudrate2 = QPushButton("Изменить Baud Rate")
        self.btn_change_baudrate2.clicked.connect(lambda: self.change_baudrate(2))
        self.btn_change_baudrate2.setEnabled(False)
        sensor2_layout.addWidget(self.btn_change_baudrate2)

        self.btn_disconnect2 = QPushButton("Отключить датчик 2")
        self.btn_disconnect2.clicked.connect(lambda: self.disconnect_sensor(2))
        self.btn_disconnect2.setEnabled(False)
        sensor2_layout.addWidget(self.btn_disconnect2)

        self.label_status2 = QLabel("Статус: Не подключен")
        sensor2_layout.addWidget(self.label_status2)

        sensor2_group.setLayout(sensor2_layout)
        main_layout.addWidget(sensor2_group)

        # Кнопка отключения обоих датчиков
        self.btn_disconnect_all = QPushButton("❌ Отключить ОБА датчика")
        self.btn_disconnect_all.clicked.connect(self.disconnect_all_sensors)
        self.btn_disconnect_all.setEnabled(False)
        self.btn_disconnect_all.setStyleSheet("background-color: #ff6b6b; color: white; font-weight: bold; padding: 10px;")
        main_layout.addWidget(self.btn_disconnect_all)

        main_layout.addStretch()

        tab.setLayout(main_layout)
        return tab
    
    def create_recording_tab(self):
        """Вкладка записи"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        sensor_group = QGroupBox("Параметры датчика")
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

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Имя файла:"))
        self.edit_filename = QLineEdit()
        self.edit_filename.setPlaceholderText("Авто (дата-время)")
        h3.addWidget(self.edit_filename)
        record_layout.addLayout(h3)

        self.label_rec_status = QLabel("Статус: Не записывается")
        record_layout.addWidget(self.label_rec_status)

        self.label_points = QLabel("Точек: 0")
        record_layout.addWidget(self.label_points)
        
        record_group.setLayout(record_layout)
        layout.addWidget(record_group)
        
        btn_layout = QVBoxLayout()
        
        self.btn_start_record = QPushButton("Начать запись")
        self.btn_start_record.clicked.connect(self.start_recording)
        self.btn_start_record.setEnabled(False)
        btn_layout.addWidget(self.btn_start_record)
        
        self.btn_stop_record = QPushButton("Остановить запись")
        self.btn_stop_record.clicked.connect(self.stop_recording)
        self.btn_stop_record.setEnabled(False)
        btn_layout.addWidget(self.btn_stop_record)
        
        layout.addLayout(btn_layout)
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
        
        self.check_auto_analyze = QCheckBox("Автоматический анализ после записи")
        self.check_auto_analyze.setChecked(True)
        file_layout.addWidget(self.check_auto_analyze)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        results_group = QGroupBox("Результаты анализа (ДВА датчика)")
        results_layout = QVBoxLayout()

        # Таблица с двумя колонками для двух датчиков
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Датчик 1", "Датчик 2"])
        self.results_table.setRowCount(5)
        self.results_table.setVerticalHeaderLabels([
            "Период (T), с",
            "Частота (f), Гц",
            "Лог. декремент (δ)",
            "Коэфф. демпфирования (ζ)",
            "Коэфф. потерь (η)"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setMaximumHeight(200)

        # Заполняем пустыми значениями
        for row in range(5):
            for col in range(2):
                self.results_table.setItem(row, col, QTableWidgetItem("-"))

        results_layout.addWidget(self.results_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        btn_export = QPushButton("Экспорт результатов")
        btn_export.clicked.connect(self.export_results)
        layout.addWidget(btn_export)
        
        # НОВАЯ КНОПКА: Сохранить график
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

        # ВЫБОР АКТИВНОГО ДАТЧИКА
        sensor_select_group = QGroupBox("🎯 Выбор датчика для редактирования")
        sensor_select_layout = QHBoxLayout()

        self.radio_sensor1 = QRadioButton("Датчик 1")
        self.radio_sensor1.setChecked(True)
        self.radio_sensor1.clicked.connect(lambda: self.set_active_sensor(1))

        self.radio_sensor2 = QRadioButton("Датчик 2")
        self.radio_sensor2.clicked.connect(lambda: self.set_active_sensor(2))

        sensor_select_layout.addWidget(self.radio_sensor1)
        sensor_select_layout.addWidget(self.radio_sensor2)

        sensor_select_group.setLayout(sensor_select_layout)
        layout.addWidget(sensor_select_group)

        # БОЛЬШАЯ КНОПКА ПЕРЕСЧЕТА
        recalc_group = QGroupBox("⚡ Обновление результатов")
        recalc_layout = QVBoxLayout()
        
        btn_recalculate = QPushButton("🔄 ПЕРЕСЧИТАТЬ ХАРАКТЕРИСТИКИ")
        btn_recalculate.clicked.connect(self.manual_recalculate)
        btn_recalculate.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14pt;
                font-weight: bold;
                padding: 15px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
        """)
        recalc_layout.addWidget(btn_recalculate)
        
        info_label = QLabel("💡 Нажмите после ручной коррекции пиков")
        info_label.setStyleSheet("color: gray; font-style: italic;")
        recalc_layout.addWidget(info_label)
        
        recalc_group.setLayout(recalc_layout)
        layout.addWidget(recalc_group)
        
        # Обрезка данных
        crop_group = QGroupBox("Обрезка данных")
        crop_layout = QVBoxLayout()
        
        # Визуальный выбор
        btn_crop_visual = QPushButton("Выбрать границы на графике")
        btn_crop_visual.setCheckable(True)
        btn_crop_visual.clicked.connect(self.toggle_crop_mode)
        crop_layout.addWidget(btn_crop_visual)
        self.btn_crop_visual = btn_crop_visual
        
        # По времени
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
        
        # По точкам
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("От точки:"))
        self.spin_crop_start_point = QSpinBox()
        self.spin_crop_start_point.setRange(0, 1000000)
        h2.addWidget(self.spin_crop_start_point)
        h2.addWidget(QLabel("До точки:"))
        self.spin_crop_end_point = QSpinBox()
        self.spin_crop_end_point.setRange(0, 1000000)
        h2.addWidget(self.spin_crop_end_point)
        crop_layout.addLayout(h2)
        
        btn_apply_crop = QPushButton("Применить обрезку")
        btn_apply_crop.clicked.connect(self.apply_crop)
        crop_layout.addWidget(btn_apply_crop)
        
        btn_reset_crop = QPushButton("Сбросить к исходным данным")
        btn_reset_crop.clicked.connect(self.reset_data)
        crop_layout.addWidget(btn_reset_crop)
        
        crop_group.setLayout(crop_layout)
        layout.addWidget(crop_group)
        
        # Пики
        peaks_group = QGroupBox("Управление пиками")
        peaks_layout = QVBoxLayout()
        
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
        
        self.label_peaks_count = QLabel("Пиков: 0")
        peaks_layout.addWidget(self.label_peaks_count)
        
        peaks_group.setLayout(peaks_layout)
        layout.addWidget(peaks_group)
        
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def log(self, message):
        """Вывод в консоль"""
        self.console.append(message)
        print(message)

    def set_active_sensor(self, sensor_num):
        """Установка активного датчика для редактирования"""
        self.active_sensor = sensor_num
        self.log(f"🎯 Активный датчик для редактирования: {sensor_num}")

        # Обновляем счетчик пиков
        active_analyzer = self.analyzer1 if sensor_num == 1 else self.analyzer2
        if active_analyzer.corrected_peaks is not None:
            self.label_peaks_count.setText(f"Пиков (датчик {sensor_num}): {len(active_analyzer.corrected_peaks)}")
        else:
            self.label_peaks_count.setText(f"Пиков (датчик {sensor_num}): 0")
    
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

    def change_baudrate(self, sensor_num):
        """Изменение baudrate после подключения"""
        sensor = self.sensor1 if sensor_num == 1 else self.sensor2
        combo_baudrate = self.combo_baudrate1 if sensor_num == 1 else self.combo_baudrate2
        combo_port = self.combo_port1 if sensor_num == 1 else self.combo_port2
        spin_address = self.spin_address1 if sensor_num == 1 else self.spin_address2

        if not sensor.is_connected:
            QMessageBox.warning(self, "Ошибка", f"Датчик {sensor_num} не подключен")
            return

        new_baudrate = int(combo_baudrate.currentText())
        port = combo_port.currentText()
        address = spin_address.value()

        self.log(f"🔄 Изменение baudrate датчика {sensor_num} на {new_baudrate}...")

        try:
            # Отключаемся
            sensor.disconnect()
            time.sleep(0.2)

            # Подключаемся с новым baudrate
            if sensor.connect(port, new_baudrate, address):
                self.log(f"✅ Датчик {sensor_num}: baudrate изменен на {new_baudrate}")
                QMessageBox.information(self, "Успех", f"Baudrate датчика {sensor_num} изменен на {new_baudrate}")
            else:
                self.log(f"❌ Датчик {sensor_num}: не удалось переподключиться")
                QMessageBox.critical(self, "Ошибка", f"Не удалось переподключить датчик {sensor_num}")
        except Exception as e:
            self.log(f"❌ Ошибка изменения baudrate: {e}")
            QMessageBox.critical(self, "Ошибка", f"Ошибка изменения baudrate:\n{e}")

    def connect_sensor(self, sensor_num):
        """Подключение к датчику"""
        sensor = self.sensor1 if sensor_num == 1 else self.sensor2
        combo_port = self.combo_port1 if sensor_num == 1 else self.combo_port2
        combo_baudrate = self.combo_baudrate1 if sensor_num == 1 else self.combo_baudrate2
        btn_connect = self.btn_connect1 if sensor_num == 1 else self.btn_connect2
        btn_identify = self.btn_identify1 if sensor_num == 1 else self.btn_identify2
        btn_disconnect = self.btn_disconnect1 if sensor_num == 1 else self.btn_disconnect2
        btn_change_baudrate = self.btn_change_baudrate1 if sensor_num == 1 else self.btn_change_baudrate2
        label_status = self.label_status1 if sensor_num == 1 else self.label_status2

        port = combo_port.currentText()
        baudrate = int(combo_baudrate.currentText())

        if not port:
            QMessageBox.warning(self, "Ошибка", f"Выберите COM-порт для датчика {sensor_num}")
            return

        self.log(f"🔌 Подключение датчика {sensor_num} к {port} ({baudrate} бод)...")

        if sensor.connect(port, baudrate):
            self.log(f"✅ Датчик {sensor_num} подключен")
            label_status.setText("Статус: Подключен")
            btn_connect.setEnabled(False)
            btn_identify.setEnabled(True)
            btn_disconnect.setEnabled(True)
            btn_change_baudrate.setEnabled(True)
            combo_port.setEnabled(False)

            # Проверяем оба датчика для включения кнопки записи
            if self.sensor1.is_connected and self.sensor2.is_connected:
                self.btn_start_record.setEnabled(True)
                self.btn_disconnect_all.setEnabled(True)
                self.log("✅ Оба датчика подключены! Можно начинать запись.")
        else:
            QMessageBox.critical(self, "Ошибка", f"Не удалось подключить датчик {sensor_num}")
            label_status.setText("Статус: Ошибка подключения")

    def identify_sensor(self, sensor_num):
        """Идентификация датчика"""
        sensor = self.sensor1 if sensor_num == 1 else self.sensor2
        spin_address = self.spin_address1 if sensor_num == 1 else self.spin_address2
        address = spin_address.value()

        self.log(f"🔍 Идентификация датчика {sensor_num} (адрес {address})...")

        if sensor.identify(address):
            self.log(f"✅ Датчик {sensor_num}: идентификация успешна")
        else:
            QMessageBox.warning(self, "Ошибка", f"Не удалось идентифицировать датчик {sensor_num}")

    def disconnect_sensor(self, sensor_num):
        """Отключение датчика"""
        sensor = self.sensor1 if sensor_num == 1 else self.sensor2
        combo_port = self.combo_port1 if sensor_num == 1 else self.combo_port2
        btn_connect = self.btn_connect1 if sensor_num == 1 else self.btn_connect2
        btn_identify = self.btn_identify1 if sensor_num == 1 else self.btn_identify2
        btn_disconnect = self.btn_disconnect1 if sensor_num == 1 else self.btn_disconnect2
        btn_change_baudrate = self.btn_change_baudrate1 if sensor_num == 1 else self.btn_change_baudrate2
        label_status = self.label_status1 if sensor_num == 1 else self.label_status2

        sensor.disconnect()
        self.log(f"🔌 Датчик {sensor_num} отключен")
        label_status.setText("Статус: Не подключен")

        btn_connect.setEnabled(True)
        btn_identify.setEnabled(False)
        btn_disconnect.setEnabled(False)
        btn_change_baudrate.setEnabled(False)
        combo_port.setEnabled(True)

        # Отключаем запись если хотя бы один датчик отключен
        if not (self.sensor1.is_connected and self.sensor2.is_connected):
            self.btn_start_record.setEnabled(False)
            self.btn_disconnect_all.setEnabled(False)

    def disconnect_all_sensors(self):
        """Отключение обоих датчиков"""
        self.log("🔌 Отключение обоих датчиков...")

        if self.sensor1.is_connected:
            self.disconnect_sensor(1)

        if self.sensor2.is_connected:
            self.disconnect_sensor(2)

        self.log("✅ Все датчики отключены")
    
    def start_recording(self):
        """Начало записи с ДВУХ датчиков синхронно"""
        if not self.sensor1.is_connected or not self.sensor2.is_connected:
            QMessageBox.warning(self, "Ошибка", "Оба датчика должны быть подключены для записи")
            return

        address1 = self.spin_address1.value()
        address2 = self.spin_address2.value()

        # Запускаем потоки данных на обоих датчиках
        if not self.sensor1.start_stream(address1):
            QMessageBox.critical(self, "Ошибка", "Не удалось запустить поток на датчике 1")
            return

        if not self.sensor2.start_stream(address2):
            self.sensor1.stop_stream(address1)
            QMessageBox.critical(self, "Ошибка", "Не удалось запустить поток на датчике 2")
            return

        base_distance = self.spin_base_distance.value()
        measurement_range = self.spin_range.value()

        # Переключение на график записи
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

        self.log("🔴 Запись ДВУХ датчиков начата")
        self.label_rec_status.setText("Статус: Идет запись с ДВУХ датчиков")
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
        self.label_rec_status.setText("Статус: Остановлена")
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        self.btn_disconnect_all.setEnabled(True)
    
    def on_data_received(self, distance1, distance2, point, time_val):
        """Обработка данных от ДВУХ датчиков"""
        # Для графика записи показываем среднее или первый датчик
        avg_distance = (distance1 + distance2) / 2.0
        self.plot_buffer.append((time_val, avg_distance))
        self.label_points.setText(f"Точек: {point} | Датчик 1: {distance1:.2f} мм | Датчик 2: {distance2:.2f} мм")
    
    def update_plot_buffer(self):
        """Обновление графика"""
        if self.plot_buffer:
            time_val, distance = self.plot_buffer[-1]
            self.plot_recording.update_plot(time_val, distance)
    
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
        """Загрузка и анализ для ДВУХ датчиков"""
        filename = self.edit_file.text()

        if not filename or not Path(filename).exists():
            QMessageBox.warning(self, "Ошибка", "Выберите файл")
            return

        self.current_file = filename
        self.log(f"📂 Загрузка: {filename}")

        # Проверяем тип файла (dual sensor или single sensor)
        try:
            test_df = pd.read_csv(filename, nrows=1)
            is_dual = 'Distance_Sensor1_mm' in test_df.columns and 'Distance_Sensor2_mm' in test_df.columns
        except:
            is_dual = False

        if is_dual:
            self.log("✅ Обнаружен файл с ДВУМЯ датчиками")

            # Загружаем данные для датчика 1
            if not self.analyzer1.load_csv(filename, sensor_column='Distance_Sensor1_mm'):
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить данные датчика 1")
                return

            # Загружаем данные для датчика 2
            if not self.analyzer2.load_csv(filename, sensor_column='Distance_Sensor2_mm'):
                QMessageBox.critical(self, "Ошибка", "Не удалось загрузить данные датчика 2")
                return

            # Нормировка для обоих датчиков
            if not self.analyzer1.normalize_data():
                QMessageBox.critical(self, "Ошибка", "Ошибка нормировки датчика 1")
                return

            if not self.analyzer2.normalize_data():
                QMessageBox.critical(self, "Ошибка", "Ошибка нормировки датчика 2")
                return

            # Переключение на графики анализа
            self.show_analysis_plots()

            # Автообрезка для обоих датчиков
            self.log("✂️ Автообрезка датчика 1...")
            success1, period1, frequency1, peaks1 = self.analyzer1.auto_crop_oscillations(1.0)

            self.log("✂️ Автообрезка датчика 2...")
            success2, period2, frequency2, peaks2 = self.analyzer2.auto_crop_oscillations(1.0)

            if not success1 or not success2:
                QMessageBox.warning(self, "Предупреждение", "Автообрезка не удалась для одного или обоих датчиков")

            if peaks1 is not None:
                self.analyzer1.calculate_logarithmic_decrement(peaks1)

            if peaks2 is not None:
                self.analyzer2.calculate_logarithmic_decrement(peaks2)

            self.update_analysis_results()
            self.update_plots()

            self.log("✅ Анализ ДВУХ датчиков завершен")

        else:
            QMessageBox.information(self, "Информация",
                                   "Это файл с одним датчиком. Используйте logyze_ui_final.py для анализа.")
            self.log("⚠️ Файл содержит данные только одного датчика")
    
    def show_recording_plot(self):
        """Показать график записи"""
        self.plot_recording.show()
        self.plot_time_sensor1.hide()
        self.plot_time_sensor2.hide()

        # Удаляем из layout
        while self.right_layout.count():
            self.right_layout.takeAt(0)

        self.right_layout.addWidget(self.plot_recording)

    def show_analysis_plots(self):
        """Показать графики анализа для ДВУХ датчиков"""
        self.plot_recording.hide()
        self.plot_time_sensor1.show()
        self.plot_time_sensor2.show()

        while self.right_layout.count():
            self.right_layout.takeAt(0)

        # Добавляем заголовки
        label1 = QLabel("📊 ДАТЧИК 1")
        label1.setStyleSheet("font-size: 14pt; font-weight: bold; color: #2196F3; padding: 5px;")
        label1.setAlignment(Qt.AlignCenter)

        label2 = QLabel("📊 ДАТЧИК 2")
        label2.setStyleSheet("font-size: 14pt; font-weight: bold; color: #4CAF50; padding: 5px;")
        label2.setAlignment(Qt.AlignCenter)

        self.right_layout.addWidget(label1)
        self.right_layout.addWidget(self.plot_time_sensor1)
        self.right_layout.addWidget(label2)
        self.right_layout.addWidget(self.plot_time_sensor2)
    
    def update_plots(self):
        """Обновление графиков анализа для ДВУХ датчиков"""
        self.plot_time_sensor1.plot_analysis(self.analyzer1, show_peaks=True)
        self.plot_time_sensor2.plot_analysis(self.analyzer2, show_peaks=True)

        # Обновление счетчика пиков для активного датчика
        active_analyzer = self.analyzer1 if self.active_sensor == 1 else self.analyzer2
        if active_analyzer.corrected_peaks is not None:
            self.label_peaks_count.setText(f"Пиков (датчик {self.active_sensor}): {len(active_analyzer.corrected_peaks)}")
    
    def update_analysis_results(self):
        """Обновление результатов для ДВУХ датчиков"""
        # Датчик 1 (колонка 0)
        if self.analyzer1.current_period:
            self.results_table.setItem(0, 0, QTableWidgetItem(f"{self.analyzer1.current_period:.6f}"))
        else:
            self.results_table.setItem(0, 0, QTableWidgetItem("-"))

        if self.analyzer1.current_frequency:
            self.results_table.setItem(1, 0, QTableWidgetItem(f"{self.analyzer1.current_frequency:.2f}"))
        else:
            self.results_table.setItem(1, 0, QTableWidgetItem("-"))

        if self.analyzer1.log_decrement:
            self.results_table.setItem(2, 0, QTableWidgetItem(f"{self.analyzer1.log_decrement:.6f}"))
        else:
            self.results_table.setItem(2, 0, QTableWidgetItem("-"))

        if self.analyzer1.damping_ratio:
            self.results_table.setItem(3, 0, QTableWidgetItem(f"{self.analyzer1.damping_ratio:.6f}"))
        else:
            self.results_table.setItem(3, 0, QTableWidgetItem("-"))

        if self.analyzer1.loss_factor:
            self.results_table.setItem(4, 0, QTableWidgetItem(f"{self.analyzer1.loss_factor:.6f}"))
        else:
            self.results_table.setItem(4, 0, QTableWidgetItem("-"))

        # Датчик 2 (колонка 1)
        if self.analyzer2.current_period:
            self.results_table.setItem(0, 1, QTableWidgetItem(f"{self.analyzer2.current_period:.6f}"))
        else:
            self.results_table.setItem(0, 1, QTableWidgetItem("-"))

        if self.analyzer2.current_frequency:
            self.results_table.setItem(1, 1, QTableWidgetItem(f"{self.analyzer2.current_frequency:.2f}"))
        else:
            self.results_table.setItem(1, 1, QTableWidgetItem("-"))

        if self.analyzer2.log_decrement:
            self.results_table.setItem(2, 1, QTableWidgetItem(f"{self.analyzer2.log_decrement:.6f}"))
        else:
            self.results_table.setItem(2, 1, QTableWidgetItem("-"))

        if self.analyzer2.damping_ratio:
            self.results_table.setItem(3, 1, QTableWidgetItem(f"{self.analyzer2.damping_ratio:.6f}"))
        else:
            self.results_table.setItem(3, 1, QTableWidgetItem("-"))

        if self.analyzer2.loss_factor:
            self.results_table.setItem(4, 1, QTableWidgetItem(f"{self.analyzer2.loss_factor:.6f}"))
        else:
            self.results_table.setItem(4, 1, QTableWidgetItem("-"))
    
    def toggle_crop_mode(self, checked):
        """Переключение режима обрезки"""
        active_plot = self.plot_time_sensor1 if self.active_sensor == 1 else self.plot_time_sensor2
        active_plot.set_crop_mode(checked)

        if checked:
            self.log(f"✂️ Режим обрезки для датчика {self.active_sensor}: кликните начало и конец на графике")
        else:
            self.log(f"✂️ Режим обрезки для датчика {self.active_sensor} выключен")

    def toggle_add_peak_mode(self, checked):
        """Режим добавления пика"""
        active_plot = self.plot_time_sensor1 if self.active_sensor == 1 else self.plot_time_sensor2

        active_plot.set_add_peak_mode(checked)

        if checked:
            self.btn_remove_peak.setChecked(False)
            active_plot.set_remove_peak_mode(False)
            self.log(f"📍 Датчик {self.active_sensor}: кликайте на графике для добавления пиков")
        else:
            self.log(f"📍 Режим добавления для датчика {self.active_sensor} выключен")

    def toggle_remove_peak_mode(self, checked):
        """Режим удаления пика"""
        active_plot = self.plot_time_sensor1 if self.active_sensor == 1 else self.plot_time_sensor2

        active_plot.set_remove_peak_mode(checked)

        if checked:
            self.btn_add_peak.setChecked(False)
            active_plot.set_add_peak_mode(False)
            self.log(f"🗑️ Датчик {self.active_sensor}: кликайте на пики для удаления")
        else:
            self.log(f"🗑️ Режим удаления для датчика {self.active_sensor} выключен")

    def show_auto_peaks(self):
        """Показать автоматически найденные пики"""
        active_analyzer = self.analyzer1 if self.active_sensor == 1 else self.analyzer2
        active_plot = self.plot_time_sensor1 if self.active_sensor == 1 else self.plot_time_sensor2

        if active_analyzer.processed_data is None:
            return

        period, frequency, peaks = active_analyzer.calculate_period_frequency_improved()

        if peaks is not None:
            active_plot.peaks = peaks.tolist()
            active_plot.update_peaks_display()
            self.label_peaks_count.setText(f"Пиков (датчик {self.active_sensor}): {len(active_plot.peaks)}")
            self.log(f"✅ Датчик {self.active_sensor}: найдено пиков: {len(peaks)}")
            self.log("💡 Нажмите '🔄 ПЕРЕСЧИТАТЬ' для обновления результатов")
    
    def apply_crop(self):
        """Применить обрезку"""
        active_analyzer = self.analyzer1 if self.active_sensor == 1 else self.analyzer2
        active_plot = self.plot_time_sensor1 if self.active_sensor == 1 else self.plot_time_sensor2

        if active_analyzer.processed_data is None:
            return

        # Проверяем источник
        start_val, end_val = active_plot.get_crop_values()

        if start_val is not None and end_val is not None:
            # Визуальный выбор
            self.log(f"✂️ Датчик {self.active_sensor}: обрезка по графику: {start_val:.6f} - {end_val:.6f}")
            active_analyzer.crop_by_time(start_val, end_val)
        elif self.spin_crop_start_time.value() > 0 or self.spin_crop_end_time.value() > 0:
            # По времени
            start_t = self.spin_crop_start_time.value()
            end_t = self.spin_crop_end_time.value()
            self.log(f"✂️ Датчик {self.active_sensor}: обрезка по времени: {start_t} - {end_t} с")
            active_analyzer.crop_by_time(start_t, end_t)
        elif self.spin_crop_start_point.value() > 0 or self.spin_crop_end_point.value() > 0:
            # По точкам
            start_p = self.spin_crop_start_point.value()
            end_p = self.spin_crop_end_point.value()
            self.log(f"✂️ Датчик {self.active_sensor}: обрезка по точкам: {start_p} - {end_p}")
            active_analyzer.crop_by_points(start_p, end_p)
        else:
            QMessageBox.warning(self, "Ошибка", "Задайте границы обрезки")
            return

        # Очистка линий
        active_plot.clear_crop_lines()

        # Сброс пиков при обрезке
        active_plot.peaks = []

        # Обновление графиков
        self.update_plots()

        self.log("💡 Пики сброшены. Нажмите 'Показать автопики' или добавьте вручную")
        self.log("💡 После коррекции пиков нажмите '🔄 ПЕРЕСЧИТАТЬ'")

    def reset_data(self):
        """Сброс к исходным данным"""
        active_analyzer = self.analyzer1 if self.active_sensor == 1 else self.analyzer2
        active_plot = self.plot_time_sensor1 if self.active_sensor == 1 else self.plot_time_sensor2

        if active_analyzer.reset_to_original():
            active_plot.clear_crop_lines()
            active_plot.peaks = []
            self.update_plots()
            self.log(f"✅ Данные датчика {self.active_sensor} сброшены")

    def manual_recalculate(self):
        """РУЧНОЙ пересчет по нажатию кнопки"""
        active_analyzer = self.analyzer1 if self.active_sensor == 1 else self.analyzer2
        active_plot = self.plot_time_sensor1 if self.active_sensor == 1 else self.plot_time_sensor2

        if active_analyzer.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для пересчета")
            return

        if len(active_plot.peaks) < 2:
            QMessageBox.warning(self, "Ошибка", "Нужно минимум 2 пика для расчета")
            return

        self.log(f"🔄 Пересчет характеристик датчика {self.active_sensor}...")

        # Устанавливаем пики в анализатор
        active_analyzer.set_manual_peaks(active_plot.peaks)

        # Пересчитываем все характеристики
        period, frequency, peaks = active_analyzer.calculate_period_frequency_improved()

        if peaks is not None:
            active_analyzer.calculate_logarithmic_decrement(peaks)

        # Обновляем интерфейс
        self.update_analysis_results()
        self.update_plots()
        self.label_peaks_count.setText(f"Пиков (датчик {self.active_sensor}): {len(active_plot.peaks)}")

        self.log(f"✅ Характеристики датчика {self.active_sensor} пересчитаны!")
        QMessageBox.information(self, "Успех", f"Характеристики датчика {self.active_sensor} успешно пересчитаны!")

    
    def save_plot(self):
        """Сохранение графиков в файлы"""
        if self.analyzer1.processed_data is None and self.analyzer2.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return

        base_filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить графики", "",
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )

        if base_filename:
            try:
                from pathlib import Path
                base = Path(base_filename)
                ext = base.suffix

                # Сохраняем график датчика 1
                if self.analyzer1.processed_data is not None:
                    filename1 = str(base.with_name(f"{base.stem}_sensor1{ext}"))
                    if self.plot_time_sensor1.save_figure(filename1):
                        self.log(f"✅ График датчика 1 сохранен: {filename1}")

                # Сохраняем график датчика 2
                if self.analyzer2.processed_data is not None:
                    filename2 = str(base.with_name(f"{base.stem}_sensor2{ext}"))
                    if self.plot_time_sensor2.save_figure(filename2):
                        self.log(f"✅ График датчика 2 сохранен: {filename2}")

                QMessageBox.information(self, "Успех", "Графики сохранены")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения:\n{e}")
    
    def export_results(self):
        """Экспорт результатов для ДВУХ датчиков"""
        if self.analyzer1.processed_data is None and self.analyzer2.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для экспорта")
            return

        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "", "Text Files (*.txt);;All Files (*)"
        )

        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=" * 80 + "\n")
                    f.write("LOGYZE ADVANCED - Результаты анализа с ДВУХ датчиков RF603HS\n")
                    f.write("=" * 80 + "\n\n")

                    f.write(f"Файл данных: {self.current_file}\n")
                    f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    # ДАТЧИК 1
                    if self.analyzer1.processed_data is not None:
                        f.write("=" * 80 + "\n")
                        f.write("ДАТЧИК 1\n")
                        f.write("=" * 80 + "\n\n")

                        f.write("-" * 80 + "\n")
                        f.write("ПАРАМЕТРЫ КОЛЕБАНИЙ\n")
                        f.write("-" * 80 + "\n")

                        if self.analyzer1.current_period:
                            f.write(f"Период (T):                    {self.analyzer1.current_period:.6f} с\n")

                        if self.analyzer1.current_frequency:
                            f.write(f"Частота (f):                   {self.analyzer1.current_frequency:.2f} Гц\n")

                        if self.analyzer1.log_decrement:
                            f.write(f"Логарифмический декремент (δ): {self.analyzer1.log_decrement:.6f}\n")

                        if self.analyzer1.damping_ratio:
                            f.write(f"Коэффициент демпфирования (ζ): {self.analyzer1.damping_ratio:.6f}\n")

                        if self.analyzer1.loss_factor:
                            f.write(f"Коэффициент потерь (η):        {self.analyzer1.loss_factor:.6f}\n")

                        f.write("\n")
                        f.write("-" * 80 + "\n")
                        f.write("СТАТИСТИКА ДАННЫХ\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"Количество точек:              {len(self.analyzer1.processed_data)}\n")

                        if self.analyzer1.corrected_peaks is not None:
                            f.write(f"Количество пиков:              {len(self.analyzer1.corrected_peaks)}\n")

                        f.write(f"Длительность записи:           {self.analyzer1.processed_data['Временная_метка'].iloc[-1]:.3f} с\n\n")

                    # ДАТЧИК 2
                    if self.analyzer2.processed_data is not None:
                        f.write("=" * 80 + "\n")
                        f.write("ДАТЧИК 2\n")
                        f.write("=" * 80 + "\n\n")

                        f.write("-" * 80 + "\n")
                        f.write("ПАРАМЕТРЫ КОЛЕБАНИЙ\n")
                        f.write("-" * 80 + "\n")

                        if self.analyzer2.current_period:
                            f.write(f"Период (T):                    {self.analyzer2.current_period:.6f} с\n")

                        if self.analyzer2.current_frequency:
                            f.write(f"Частота (f):                   {self.analyzer2.current_frequency:.2f} Гц\n")

                        if self.analyzer2.log_decrement:
                            f.write(f"Логарифмический декремент (δ): {self.analyzer2.log_decrement:.6f}\n")

                        if self.analyzer2.damping_ratio:
                            f.write(f"Коэффициент демпфирования (ζ): {self.analyzer2.damping_ratio:.6f}\n")

                        if self.analyzer2.loss_factor:
                            f.write(f"Коэффициент потерь (η):        {self.analyzer2.loss_factor:.6f}\n")

                        f.write("\n")
                        f.write("-" * 80 + "\n")
                        f.write("СТАТИСТИКА ДАННЫХ\n")
                        f.write("-" * 80 + "\n")
                        f.write(f"Количество точек:              {len(self.analyzer2.processed_data)}\n")

                        if self.analyzer2.corrected_peaks is not None:
                            f.write(f"Количество пиков:              {len(self.analyzer2.corrected_peaks)}\n")

                        f.write(f"Длительность записи:           {self.analyzer2.processed_data['Временная_метка'].iloc[-1]:.3f} с\n\n")

                    f.write("=" * 80 + "\n")

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
