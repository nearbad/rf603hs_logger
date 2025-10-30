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
        
        # График для записи (один)
        self.plot_recording = InteractivePlotWidget('time')
        self.right_layout.addWidget(self.plot_recording)
        
        # Графики для анализа (два) - скрыты по умолчанию
        self.plot_time = InteractivePlotWidget('time')
        self.plot_points = InteractivePlotWidget('points')
        self.plot_time.hide()
        self.plot_points.hide()
        
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
        """Вкладка подключения"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        port_group = QGroupBox("Настройки подключения")
        port_layout = QVBoxLayout()
        
        h1 = QHBoxLayout()
        h1.addWidget(QLabel("COM-порт:"))
        self.combo_port = QComboBox()
        h1.addWidget(self.combo_port)
        btn_refresh = QPushButton("🔄")
        btn_refresh.setMaximumWidth(40)
        btn_refresh.clicked.connect(self.refresh_ports)
        h1.addWidget(btn_refresh)
        port_layout.addLayout(h1)
        
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Baud Rate:"))
        self.combo_baudrate = QComboBox()
        self.combo_baudrate.addItems([str(b) for b in RF603Sensor.BAUDRATES])
        self.combo_baudrate.setCurrentText("9600")
        h2.addWidget(self.combo_baudrate)
        port_layout.addLayout(h2)
        
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Адрес:"))
        self.spin_address = QSpinBox()
        self.spin_address.setRange(1, 127)
        self.spin_address.setValue(1)
        h3.addWidget(self.spin_address)
        port_layout.addLayout(h3)
        
        port_group.setLayout(port_layout)
        layout.addWidget(port_group)
        
        btn_layout = QVBoxLayout()
        
        self.btn_connect = QPushButton("Подключиться")
        self.btn_connect.clicked.connect(self.connect_sensor)
        btn_layout.addWidget(self.btn_connect)
        
        self.btn_identify = QPushButton("Идентификация")
        self.btn_identify.clicked.connect(self.identify_sensor)
        self.btn_identify.setEnabled(False)
        btn_layout.addWidget(self.btn_identify)
        
        self.btn_disconnect = QPushButton("Отключиться")
        self.btn_disconnect.clicked.connect(self.disconnect_sensor)
        self.btn_disconnect.setEnabled(False)
        btn_layout.addWidget(self.btn_disconnect)
        
        layout.addLayout(btn_layout)
        layout.addStretch()
        
        tab.setLayout(layout)
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
        
        self.label_status = QLabel("Статус: Не записывается")
        record_layout.addWidget(self.label_status)
        
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
        
        results_group = QGroupBox("Результаты анализа")
        results_layout = QVBoxLayout()
        
        self.label_period = QLabel("Период: -")
        results_layout.addWidget(self.label_period)
        
        self.label_frequency = QLabel("Частота: -")
        results_layout.addWidget(self.label_frequency)
        
        self.label_decrement = QLabel("Лог. декремент: -")
        results_layout.addWidget(self.label_decrement)
        
        self.label_damping = QLabel("Коэфф. демпфирования: -")
        results_layout.addWidget(self.label_damping)
        
        self.label_loss = QLabel("Коэфф. потерь: -")
        results_layout.addWidget(self.label_loss)
        
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
    
    def refresh_ports(self):
        """Обновление списка портов"""
        self.combo_port.clear()
        ports = self.sensor.list_ports()
        
        if ports:
            self.combo_port.addItems(ports)
            self.log(f"📡 Найдено портов: {len(ports)}")
        else:
            self.log("⚠️ COM-порты не найдены")
    
    def connect_sensor(self):
        """Подключение к датчику"""
        port = self.combo_port.currentText()
        baudrate = int(self.combo_baudrate.currentText())
        
        if not port:
            QMessageBox.warning(self, "Ошибка", "Выберите COM-порт")
            return
        
        self.log(f"🔌 Подключение к {port} ({baudrate} бод)...")
        
        if self.sensor.connect(port, baudrate):
            self.log("✅ Подключено")
            self.btn_connect.setEnabled(False)
            self.btn_identify.setEnabled(True)
            self.btn_disconnect.setEnabled(True)
            self.btn_start_record.setEnabled(True)
            self.combo_port.setEnabled(False)
            self.combo_baudrate.setEnabled(False)
        else:
            QMessageBox.critical(self, "Ошибка", "Не удалось подключиться")
    
    def identify_sensor(self):
        """Идентификация датчика"""
        address = self.spin_address.value()
        self.log(f"🔍 Идентификация (адрес {address})...")
        
        if self.sensor.identify(address):
            self.log("✅ Идентификация успешна")
        else:
            QMessageBox.warning(self, "Ошибка", "Не удалось идентифицировать")
    
    def disconnect_sensor(self):
        """Отключение"""
        self.sensor.disconnect()
        self.log("🔌 Отключено")
        
        self.btn_connect.setEnabled(True)
        self.btn_identify.setEnabled(False)
        self.btn_disconnect.setEnabled(False)
        self.btn_start_record.setEnabled(False)
        self.combo_port.setEnabled(True)
        self.combo_baudrate.setEnabled(True)
    
    def start_recording(self):
        """Начало записи"""
        if not self.sensor.is_connected:
            QMessageBox.warning(self, "Ошибка", "Датчик не подключен")
            return
        
        address = self.spin_address.value()
        if not self.sensor.start_stream(address):
            QMessageBox.critical(self, "Ошибка", "Не удалось запустить поток")
            return
        
        base_distance = self.spin_base_distance.value()
        measurement_range = self.spin_range.value()
        
        # Переключение на график записи
        self.show_recording_plot()
        self.plot_recording.clear_plot()
        self.plot_buffer = []
        
        self.recording_thread = RecordingThread(
            self.sensor, base_distance, measurement_range
        )
        self.recording_thread.data_received.connect(self.on_data_received)
        self.recording_thread.recording_finished.connect(self.on_recording_finished)
        self.recording_thread.error_occurred.connect(self.on_recording_error)
        
        self.recording_thread.start()
        self.is_recording = True
        
        self.update_timer.start(100)
        
        self.log("🔴 Запись начата")
        self.label_status.setText("Статус: Идет запись")
        self.btn_start_record.setEnabled(False)
        self.btn_stop_record.setEnabled(True)
        self.btn_disconnect.setEnabled(False)
    
    def stop_recording(self):
        """Остановка записи"""
        if self.recording_thread:
            self.recording_thread.stop()
            self.recording_thread.wait()
        
        address = self.spin_address.value()
        self.sensor.stop_stream(address)
        
        self.update_timer.stop()
        self.is_recording = False
        
        self.log("⏹️ Запись остановлена")
        self.label_status.setText("Статус: Остановлена")
        self.btn_start_record.setEnabled(True)
        self.btn_stop_record.setEnabled(False)
        self.btn_disconnect.setEnabled(True)
    
    def on_data_received(self, distance, point, time_val):
        """Обработка данных"""
        self.plot_buffer.append((time_val, distance))
        self.label_points.setText(f"Точек: {point}")
    
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
        """Загрузка и анализ"""
        filename = self.edit_file.text()
        
        if not filename or not Path(filename).exists():
            QMessageBox.warning(self, "Ошибка", "Выберите файл")
            return
        
        self.current_file = filename
        self.log(f"📂 Загрузка: {filename}")
        
        if not self.analyzer.load_csv(filename):
            QMessageBox.critical(self, "Ошибка", "Не удалось загрузить")
            return
        
        if not self.analyzer.normalize_data():
            QMessageBox.critical(self, "Ошибка", "Ошибка нормировки")
            return
        
        # Переключение на графики анализа
        self.show_analysis_plots()
        
        self.log("✂️ Автообрезка...")
        success, period, frequency, peaks = self.analyzer.auto_crop_oscillations(1.0)
        
        if not success:
            QMessageBox.warning(self, "Предупреждение", "Автообрезка не удалась")
            return
        
        if peaks is not None:
            self.analyzer.calculate_logarithmic_decrement(peaks)
        
        self.update_analysis_results()
        self.update_plots()
        
        self.log("✅ Анализ завершен")
    
    def show_recording_plot(self):
        """Показать график записи"""
        self.plot_recording.show()
        self.plot_time.hide()
        self.plot_points.hide()
        
        # Удаляем из layout
        while self.right_layout.count():
            self.right_layout.takeAt(0)
        
        self.right_layout.addWidget(self.plot_recording)
    
    def show_analysis_plots(self):
        """Показать графики анализа"""
        self.plot_recording.hide()
        self.plot_time.show()
        self.plot_points.show()
        
        while self.right_layout.count():
            self.right_layout.takeAt(0)
        
        self.right_layout.addWidget(self.plot_time)
        self.right_layout.addWidget(self.plot_points)
    
    def update_plots(self):
        """Обновление графиков анализа"""
        self.plot_time.plot_analysis(self.analyzer, show_peaks=True)
        self.plot_points.plot_analysis(self.analyzer, show_peaks=False)
        
        # Обновление счетчика пиков
        if self.analyzer.corrected_peaks is not None:
            self.label_peaks_count.setText(f"Пиков: {len(self.analyzer.corrected_peaks)}")
    
    def update_analysis_results(self):
        """Обновление результатов"""
        if self.analyzer.current_period:
            self.label_period.setText(f"Период: {self.analyzer.current_period:.6f} с")
        else:
            self.label_period.setText("Период: -")
        
        if self.analyzer.current_frequency:
            self.label_frequency.setText(f"Частота: {self.analyzer.current_frequency:.2f} Гц")
        else:
            self.label_frequency.setText("Частота: -")
        
        if self.analyzer.log_decrement:
            self.label_decrement.setText(f"Лог. декремент: {self.analyzer.log_decrement:.6f}")
        else:
            self.label_decrement.setText("Лог. декремент: -")
        
        if self.analyzer.damping_ratio:
            self.label_damping.setText(f"Коэфф. демпфирования: {self.analyzer.damping_ratio:.6f}")
        else:
            self.label_damping.setText("Коэфф. демпфирования: -")
        
        if self.analyzer.loss_factor:
            self.label_loss.setText(f"Коэфф. потерь: {self.analyzer.loss_factor:.6f}")
        else:
            self.label_loss.setText("Коэфф. потерь: -")
    
    def toggle_crop_mode(self, checked):
        """Переключение режима обрезки"""
        self.plot_time.set_crop_mode(checked)
        self.plot_points.set_crop_mode(checked)
        
        if checked:
            self.log("✂️ Режим обрезки: кликните начало и конец на графике")
        else:
            self.log("✂️ Режим обрезки выключен")
    
    def toggle_add_peak_mode(self, checked):
        """Режим добавления пика"""
        self.plot_time.set_add_peak_mode(checked)
        
        if checked:
            self.btn_remove_peak.setChecked(False)
            self.plot_time.set_remove_peak_mode(False)
            self.log("📍 Кликайте на графике для добавления пиков")
        else:
            self.log("📍 Режим добавления выключен")
    
    def toggle_remove_peak_mode(self, checked):
        """Режим удаления пика"""
        self.plot_time.set_remove_peak_mode(checked)
        
        if checked:
            self.btn_add_peak.setChecked(False)
            self.plot_time.set_add_peak_mode(False)
            self.log("🗑️ Кликайте на пики для удаления")
        else:
            self.log("🗑️ Режим удаления выключен")
    
    def show_auto_peaks(self):
        """Показать автоматически найденные пики"""
        if self.analyzer.processed_data is None:
            return
        
        period, frequency, peaks = self.analyzer.calculate_period_frequency_improved()
        
        if peaks is not None:
            self.plot_time.peaks = peaks.tolist()
            self.plot_time.update_peaks_display()
            self.label_peaks_count.setText(f"Пиков: {len(self.plot_time.peaks)}")
            self.log(f"✅ Найдено пиков: {len(peaks)}")
            self.log("💡 Нажмите '🔄 ПЕРЕСЧИТАТЬ' для обновления результатов")
    
    def apply_crop(self):
        """Применить обрезку"""
        if self.analyzer.processed_data is None:
            return
        
        # Проверяем источник
        start_val, end_val = self.plot_time.get_crop_values()
        
        if start_val is not None and end_val is not None:
            # Визуальный выбор
            self.log(f"✂️ Обрезка по графику: {start_val:.6f} - {end_val:.6f}")
            self.analyzer.crop_by_time(start_val, end_val)
        elif self.spin_crop_start_time.value() > 0 or self.spin_crop_end_time.value() > 0:
            # По времени
            start_t = self.spin_crop_start_time.value()
            end_t = self.spin_crop_end_time.value()
            self.log(f"✂️ Обрезка по времени: {start_t} - {end_t} с")
            self.analyzer.crop_by_time(start_t, end_t)
        elif self.spin_crop_start_point.value() > 0 or self.spin_crop_end_point.value() > 0:
            # По точкам
            start_p = self.spin_crop_start_point.value()
            end_p = self.spin_crop_end_point.value()
            self.log(f"✂️ Обрезка по точкам: {start_p} - {end_p}")
            self.analyzer.crop_by_points(start_p, end_p)
        else:
            QMessageBox.warning(self, "Ошибка", "Задайте границы обрезки")
            return
        
        # Очистка линий
        self.plot_time.clear_crop_lines()
        self.plot_points.clear_crop_lines()
        
        # Сброс пиков при обрезке
        self.plot_time.peaks = []
        
        # Обновление графиков
        self.update_plots()
        
        self.log("💡 Пики сброшены. Нажмите 'Показать автопики' или добавьте вручную")
        self.log("💡 После коррекции пиков нажмите '🔄 ПЕРЕСЧИТАТЬ'")
    
    def reset_data(self):
        """Сброс к исходным данным"""
        if self.analyzer.reset_to_original():
            self.plot_time.clear_crop_lines()
            self.plot_points.clear_crop_lines()
            self.plot_time.peaks = []
            self.update_plots()
            self.log("✅ Данные сброшены")
    
    def recalculate_characteristics(self):
        """Пересчет характеристик"""
        if len(self.plot_time.peaks) >= 2:
            # Устанавливаем пики в анализатор
            self.analyzer.set_manual_peaks(self.plot_time.peaks)
            
            # Пересчитываем
            period, frequency, peaks = self.analyzer.calculate_period_frequency_improved()
            
            if peaks is not None:
                self.analyzer.calculate_logarithmic_decrement(peaks)
            
            # ВАЖНО: Обновляем результаты И графики
            self.update_analysis_results()
            self.update_plots()  # Добавлено обновление графиков!
            self.label_peaks_count.setText(f"Пиков: {len(self.plot_time.peaks)}")
    
    def manual_recalculate(self):
        """РУЧНОЙ пересчет по нажатию кнопки"""
        if self.analyzer.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для пересчета")
            return
        
        if len(self.plot_time.peaks) < 2:
            QMessageBox.warning(self, "Ошибка", "Нужно минимум 2 пика для расчета")
            return
        
        self.log("🔄 Пересчет характеристик...")
        
        # Устанавливаем пики в анализатор
        self.analyzer.set_manual_peaks(self.plot_time.peaks)
        
        # Пересчитываем все характеристики
        period, frequency, peaks = self.analyzer.calculate_period_frequency_improved()
        
        if peaks is not None:
            self.analyzer.calculate_logarithmic_decrement(peaks)
        
        # Обновляем интерфейс
        self.update_analysis_results()
        self.update_plots()
        self.label_peaks_count.setText(f"Пиков: {len(self.plot_time.peaks)}")
        
        self.log("✅ Характеристики пересчитаны!")
        QMessageBox.information(self, "Успех", "Характеристики успешно пересчитаны!")

    
    def save_plot(self):
        """Сохранение графика в файл"""
        if self.analyzer.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для сохранения")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить график", "", 
            "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if filename:
            try:
                # Сохраняем график времени (основной)
                if self.plot_time.save_figure(filename):
                    self.log(f"✅ График сохранен: {filename}")
                    QMessageBox.information(self, "Успех", f"График сохранен:\n{filename}")
                else:
                    QMessageBox.critical(self, "Ошибка", "Не удалось сохранить график")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка сохранения:\n{e}")
    
    def export_results(self):
        """Экспорт результатов"""
        if self.analyzer.processed_data is None:
            QMessageBox.warning(self, "Ошибка", "Нет данных для экспорта")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить результаты", "", "Text Files (*.txt);;All Files (*)"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("=" * 60 + "\n")
                    f.write("LOGYZE - Результаты анализа затухающих колебаний\n")
                    f.write("=" * 60 + "\n\n")
                    
                    f.write(f"Файл данных: {self.current_file}\n")
                    f.write(f"Дата анализа: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    f.write("-" * 60 + "\n")
                    f.write("ПАРАМЕТРЫ КОЛЕБАНИЙ\n")
                    f.write("-" * 60 + "\n")
                    
                    if self.analyzer.current_period:
                        f.write(f"Период (T):                    {self.analyzer.current_period:.6f} с\n")
                    
                    if self.analyzer.current_frequency:
                        f.write(f"Частота (f):                   {self.analyzer.current_frequency:.2f} Гц\n")
                    
                    if self.analyzer.log_decrement:
                        f.write(f"Логарифмический декремент (δ): {self.analyzer.log_decrement:.6f}\n")
                    
                    if self.analyzer.damping_ratio:
                        f.write(f"Коэффициент демпфирования (ζ): {self.analyzer.damping_ratio:.6f}\n")
                    
                    if self.analyzer.loss_factor:
                        f.write(f"Коэффициент потерь (η):        {self.analyzer.loss_factor:.6f}\n")
                    
                    f.write("\n")
                    f.write("-" * 60 + "\n")
                    f.write("СТАТИСТИКА ДАННЫХ\n")
                    f.write("-" * 60 + "\n")
                    f.write(f"Количество точек:              {len(self.analyzer.processed_data)}\n")
                    
                    if self.analyzer.corrected_peaks is not None:
                        f.write(f"Количество пиков:              {len(self.analyzer.corrected_peaks)}\n")
                        f.write("Примечание:                    Использованы исправленные пики\n")
                    
                    f.write(f"Длительность записи:           {self.analyzer.processed_data['Временная_метка'].iloc[-1]:.3f} с\n")
                    
                    f.write("\n" + "=" * 60 + "\n")
                
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
