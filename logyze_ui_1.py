# -*- coding: utf-8 -*-
"""
RF603HS Professional Oscillation Analyzer
Графический интерфейс на PyQt5 для анализа затухающих колебаний
"""

import sys
import os
import time
import csv
from datetime import datetime
import traceback

import serial
import serial.tools.list_ports
import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QTextEdit, QFileDialog, QMessageBox, QProgressBar,
    QSplitter, QTabWidget, QCheckBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# ============================================================================
# КЛАСС ДЛЯ СБОРА ДАННЫХ В ОТДЕЛЬНОМ ПОТОКЕ
# ============================================================================

class DataCollectionThread(QThread):
    """Улучшенный поток для сбора данных с датчика RF603HS"""
    data_received = pyqtSignal(dict)
    status_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    collection_finished = pyqtSignal(str)
    
    def __init__(self, port, baudrate, address=1):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.address = address
        self.is_running = False
        self.ser = None
        self.data_buffer = []
        self.start_time = None
        self.range_mm = 100
        self.last_counter = None  # Для отслеживания счётчика пакетов
        self.sync_errors = 0  # Счётчик ошибок синхронизации
        
    def send_request(self, code):
        """Отправка запроса датчику"""
        inc0 = self.address & 0x7F
        inc1 = 0x80 | (code & 0x0F)
        request = bytes([inc0, inc1])
        self.ser.write(request)
        self.ser.flush()
        time.sleep(0.01)  # Небольшая задержка для стабильности
    
    def clear_buffer(self):
        """Очистка входного буфера"""
        if self.ser and self.ser.is_open:
            self.ser.reset_input_buffer()
            time.sleep(0.05)
    
    def find_packet_start(self, timeout=2.0):
        """
        Поиск начала пакета данных
        Ищет два последовательных байта со старшим битом = 1
        """
        start_time = time.time()
        byte_buffer = []
        
        while (time.time() - start_time) < timeout:
            if self.ser.in_waiting > 0:
                byte = self.ser.read(1)
                if len(byte) > 0:
                    byte_buffer.append(byte[0])
                    
                    # Ищем последовательность из двух байтов с битом 0x80
                    if len(byte_buffer) >= 2:
                        if (byte_buffer[-2] & 0x80) and (byte_buffer[-1] & 0x80):
                            # Проверяем, что это похоже на начало пакета данных
                            # (не команду)
                            sb = (byte_buffer[-2] >> 6) & 0x01  # Status bit
                            cnt = (byte_buffer[-2] >> 4) & 0x03  # Counter
                            # Если это выглядит как данные, возвращаем эти два байта
                            return bytes(byte_buffer[-2:])
                        
                        # Сохраняем только последний байт для следующей проверки
                        byte_buffer = byte_buffer[-1:]
            else:
                time.sleep(0.001)
        
        return None
    
    def decode_measurement_packet(self, byte1, byte2):
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
    
    def validate_measurement(self, raw_value, distance_mm):
        """
        Валидация измеренного значения
        Возвращает True, если значение корректно
        """
        # Проверка 1: raw_value должен быть в разумных пределах
        if raw_value <= 0 or raw_value > 0x4000:
            return False
        
        # Проверка 2: distance не должна быть отрицательной или слишком большой
        if distance_mm < 0 or distance_mm > (self.range_mm * 1.2):
            return False
        
        # Проверка 3: если есть предыдущее значение, скачок не должен быть огромным
        if len(self.data_buffer) > 0:
            last_distance = self.data_buffer[-1]['distance']
            delta = abs(distance_mm - last_distance)
            # Максимальное изменение за один отсчёт (зависит от скорости)
            # При 70 кГц максимум ~1.4 мм за отсчёт при скорости 100 м/с
            max_delta = self.range_mm * 0.1  # 10% от диапазона
            if delta > max_delta:
                return False
        
        return True
    
    def run(self):
        """Основной цикл сбора данных"""
        try:
            # === ПОДКЛЮЧЕНИЕ ===
            self.status_update.emit("Подключение к датчику...")
            self.ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_EVEN,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.5  # Уменьшенный timeout для быстрого отклика
            )
            
            # Очистка буфера
            self.clear_buffer()
            time.sleep(0.1)
            
            # === ИДЕНТИФИКАЦИЯ ===
            self.status_update.emit("Идентификация датчика...")
            self.send_request(0x01)
            time.sleep(0.1)
            
            response = self.ser.read(16)
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
                    fw_version = decoded_bytes[1]
                    serial_num = decoded_bytes[2] | (decoded_bytes[3] << 8)
                    base_distance = decoded_bytes[4] | (decoded_bytes[5] << 8)
                    self.range_mm = decoded_bytes[6] | (decoded_bytes[7] << 8)
                    
                    self.status_update.emit(
                        f"✓ Датчик найден\n"
                        f"  S/N: {serial_num}\n"
                        f"  Диапазон: {self.range_mm} мм\n"
                        f"  Базовое расстояние: {base_distance} мм"
                    )
            else:
                self.error_occurred.emit("Ошибка идентификации датчика")
                return
            
            # Очистка буфера перед началом потока
            self.clear_buffer()
            
            # === ЗАПУСК ПОТОКА ДАННЫХ ===
            self.status_update.emit("Запуск потока данных...")
            self.send_request(0x07)
            time.sleep(0.1)
            
            # === СИНХРОНИЗАЦИЯ ===
            self.status_update.emit("Поиск начала пакета...")
            initial_packet = self.find_packet_start(timeout=3.0)
            
            if initial_packet is None:
                self.error_occurred.emit("Не удалось синхронизироваться с потоком данных")
                return
            
            self.status_update.emit("✓ Синхронизация успешна, начинаем сбор...")
            
            # === ОСНОВНОЙ ЦИКЛ СБОРА ===
            self.is_running = True
            self.start_time = time.time()
            self.data_buffer = []
            self.last_counter = None
            self.sync_errors = 0
            
            # Читаем первый пакет, который уже получили при синхронизации
            byte1, byte2 = initial_packet[0], initial_packet[1]
            
            consecutive_errors = 0
            max_consecutive_errors = 10
            
            while self.is_running:
                # Декодируем текущий пакет
                raw_value, status_bit, counter, valid = self.decode_measurement_packet(byte1, byte2)
                
                if valid:
                    # Проверка счётчика пакетов
                    if self.last_counter is not None:
                        expected_counter = (self.last_counter + 1) % 4
                        if counter != expected_counter:
                            self.sync_errors += 1
                            self.status_update.emit(
                                f"⚠ Пропущен пакет (ожидали {expected_counter}, получили {counter}). "
                                f"Всего ошибок: {self.sync_errors}"
                            )
                    
                    self.last_counter = counter
                    
                    # Преобразование в миллиметры
                    if raw_value > 0:
                        distance_mm = (raw_value * self.range_mm) / 0x4000
                        
                        # Валидация
                        if self.validate_measurement(raw_value, distance_mm):
                            current_time = time.time() - self.start_time
                            
                            data_point = {
                                'distance': distance_mm,
                                'point': len(self.data_buffer),
                                'time': current_time,
                                'raw_value': raw_value,
                                'status_bit': status_bit,
                                'counter': counter
                            }
                            
                            self.data_buffer.append(data_point)
                            self.data_received.emit(data_point)
                            consecutive_errors = 0
                        else:
                            consecutive_errors += 1
                            if consecutive_errors >= max_consecutive_errors:
                                self.error_occurred.emit(
                                    f"Слишком много невалидных значений подряд ({consecutive_errors}). "
                                    "Возможна проблема с датчиком или подключением."
                                )
                                break
                else:
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        self.error_occurred.emit(
                            "Потеряна синхронизация с потоком данных. "
                            "Попытка пересинхронизации..."
                        )
                        # Попытка пересинхронизации
                        self.clear_buffer()
                        time.sleep(0.05)
                        new_packet = self.find_packet_start(timeout=2.0)
                        if new_packet:
                            byte1, byte2 = new_packet[0], new_packet[1]
                            consecutive_errors = 0
                            self.status_update.emit("✓ Синхронизация восстановлена")
                            continue
                        else:
                            break
                
                # Читаем следующий пакет (2 байта)
                if self.ser.in_waiting >= 2:
                    next_bytes = self.ser.read(2)
                    if len(next_bytes) == 2:
                        byte1, byte2 = next_bytes[0], next_bytes[1]
                    else:
                        time.sleep(0.001)
                else:
                    time.sleep(0.001)
            
            # === ОСТАНОВКА ===
            self.status_update.emit("Остановка потока данных...")
            self.send_request(0x08)
            time.sleep(0.1)
            self.clear_buffer()
            
            # === СОХРАНЕНИЕ ===
            if self.data_buffer:
                self.status_update.emit(f"Сохранение {len(self.data_buffer)} точек...")
                filename = self.save_data()
                if filename:
                    self.status_update.emit(f"✓ Данные сохранены: {filename}")
                    self.collection_finished.emit(filename)
            else:
                self.status_update.emit("⚠ Нет данных для сохранения")
            
        except serial.SerialException as e:
            self.error_occurred.emit(f"Ошибка COM-порта: {str(e)}")
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            self.error_occurred.emit(f"Ошибка: {str(e)}\n\nДетали:\n{error_details}")
        finally:
            if self.ser and self.ser.is_open:
                try:
                    self.send_request(0x08)  # Остановка потока
                    time.sleep(0.05)
                except:
                    pass
                self.ser.close()
                self.status_update.emit("Датчик отключен")
    
    def save_data(self):
        """Сохранение данных в CSV с дополнительной информацией"""
        try:
            session_folder = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(session_folder, exist_ok=True)
            filename = os.path.join(session_folder, "raw_data.csv")
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=';')
                # Расширенный заголовок с дополнительными полями
                writer.writerow([
                    'Расстояние_мм', 
                    'Номер_точки', 
                    'Временная_метка',
                    'Raw_Value',
                    'Status_Bit',
                    'Counter'
                ])
                for point in self.data_buffer:
                    writer.writerow([
                        f"{point['distance']:.3f}",
                        point['point'],
                        f"{point['time']:.6f}",
                        point.get('raw_value', ''),
                        point.get('status_bit', ''),
                        point.get('counter', '')
                    ])
            
            # Дополнительно сохраняем информацию о сессии
            info_file = os.path.join(session_folder, "session_info.txt")
            with open(info_file, 'w', encoding='utf-8') as f:
                f.write(f"Сессия сбора данных RF603HS\n")
                f.write(f"{'='*50}\n\n")
                f.write(f"Дата и время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Порт: {self.port}\n")
                f.write(f"Скорость: {self.baudrate} бод\n")
                f.write(f"Адрес датчика: {self.address}\n")
                f.write(f"Диапазон: {self.range_mm} мм\n")
                f.write(f"Точек собрано: {len(self.data_buffer)}\n")
                if self.data_buffer:
                    f.write(f"Длительность: {self.data_buffer[-1]['time']:.3f} сек\n")
                f.write(f"Ошибок синхронизации: {self.sync_errors}\n")
            
            return filename
        except Exception as e:
            self.error_occurred.emit(f"Ошибка сохранения: {str(e)}")
            return None
    
    def stop(self):
        """Остановка сбора"""
        self.is_running = False


# ============================================================================
# КЛАСС ДЛЯ АНАЛИЗА КОЛЕБАНИЙ
# ============================================================================

class OscillationAnalyzer:
    """Класс для анализа затухающих колебаний"""
    
    def __init__(self):
        self.data = None
        self.processed_data = None
        self.current_period = None
        self.current_frequency = None
        self.log_decrement = None
        self.loss_factor = None
        self.damping_ratio = None
        self.peaks = None
        
    def load_and_analyze(self, filename, duration=1.0):
        """Загрузка и автоматический анализ"""
        try:
            # Загрузка
            self.data = pd.read_csv(filename, delimiter=';', encoding='utf-8')
            
            # Нормировка
            self.processed_data = self.data.copy()
            first_distance = self.processed_data.iloc[0]['Расстояние_мм']
            self.processed_data['Расстояние_норм'] = (
                self.processed_data['Расстояние_мм'] - first_distance
            )
            
            # Поиск начала колебаний
            start_idx = self.find_release_point()
            start_time = self.processed_data.iloc[start_idx]['Временная_метка']
            end_time = start_time + duration
            
            # Обрезка
            mask = ((self.processed_data['Временная_метка'] >= start_time) & 
                   (self.processed_data['Временная_метка'] <= end_time))
            self.processed_data = self.processed_data[mask].reset_index(drop=True)
            
            # Расчет параметров
            return self.calculate_parameters()
            
        except Exception as e:
            return False, f"Ошибка анализа: {str(e)}"
    
    def find_release_point(self, threshold=0.5):
        """Поиск начала колебаний"""
        distances = self.processed_data['Расстояние_норм'].values
        for i in range(1, len(distances) - 10):
            window = distances[i:i+10]
            if np.max(window) - np.min(window) > threshold:
                return max(0, i - 5)
        return 0
    
    def calculate_parameters(self):
        """Расчет всех параметров"""
        try:
            distances = self.processed_data['Расстояние_норм'].values
            timestamps = self.processed_data['Временная_метка'].values
            
            # Сглаживание
            if len(distances) > 11:
                distances_smooth = savgol_filter(distances, 11, 3)
            else:
                distances_smooth = distances
            
            # Поиск пиков
            amplitude = np.max(distances_smooth) - np.min(distances_smooth)
            self.peaks, _ = find_peaks(
                distances_smooth,
                height=amplitude * 0.1,
                distance=max(3, len(distances_smooth) // 20),
                prominence=amplitude * 0.05
            )
            
            if len(self.peaks) < 2:
                return False, "Недостаточно пиков"
            
            # Расчет периодов
            periods = []
            for i in range(len(self.peaks) - 1):
                period = timestamps[self.peaks[i + 1]] - timestamps[self.peaks[i]]
                periods.append(period)
            
            self.current_period = np.mean(periods)
            self.current_frequency = 1.0 / self.current_period
            
            # Логарифмический декремент
            amplitudes = distances[self.peaks]
            decrements = []
            for i in range(len(self.peaks) - 1):
                A_i = abs(amplitudes[i])
                A_i_plus_1 = abs(amplitudes[i + 1])
                if A_i_plus_1 > 0:
                    decrements.append(np.log(A_i / A_i_plus_1))
            
            if decrements:
                self.log_decrement = np.mean(decrements)
                self.damping_ratio = self.log_decrement / np.sqrt(4 * np.pi**2 + self.log_decrement**2)
                self.loss_factor = 2 * self.damping_ratio
            
            return True, "Анализ выполнен"
            
        except Exception as e:
            return False, f"Ошибка расчета: {str(e)}"
    
    def get_results(self):
        """Получение результатов"""
        return {
            'period': self.current_period,
            'frequency': self.current_frequency,
            'log_decrement': self.log_decrement,
            'damping_ratio': self.damping_ratio,
            'loss_factor': self.loss_factor,
            'num_peaks': len(self.peaks) if self.peaks is not None else 0
        }


# ============================================================================
# ВИДЖЕТ ДЛЯ MATPLOTLIB
# ============================================================================

class MatplotlibWidget(QWidget):
    """Виджет для отображения графиков"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
    
    def clear(self):
        """Очистка графика"""
        self.figure.clear()
        self.canvas.draw()


# ============================================================================
# ГЛАВНОЕ ОКНО ПРИЛОЖЕНИЯ
# ============================================================================

class RF603AnalyzerGUI(QMainWindow):
    """Главное окно приложения"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RF603HS Professional Oscillation Analyzer")
        self.setGeometry(100, 100, 1400, 900)
        
        # Данные
        self.collection_thread = None
        self.live_data = []
        self.current_session_folder = None
        self.analyzer = OscillationAnalyzer()
        
        # Логирование
        self.log_file = None
        self.init_logging()
        
        # UI
        self.init_ui()
        
        # Таймер для обновления графика
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_live_plot)
        
        self.log("Приложение запущено")
    
    def init_logging(self):
        """Инициализация логирования"""
        log_folder = "logs"
        os.makedirs(log_folder, exist_ok=True)
        log_filename = os.path.join(log_folder, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        self.log_file = open(log_filename, 'w', encoding='utf-8')
    
    def log(self, message):
        """Добавление записи в лог"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        if self.log_file:
            self.log_file.write(log_entry + '\n')
            self.log_file.flush()
        
        if hasattr(self, 'log_console'):
            self.log_console.append(log_entry)
            self.log_console.moveCursor(QTextCursor.End)
    
    def init_ui(self):
        """Инициализация интерфейса"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Horizontal)
        
        # Левая панель - управление
        left_panel = self.create_control_panel()
        
        # Правая панель - визуализация
        right_panel = self.create_visualization_panel()
        
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 1000])
        
        main_layout.addWidget(splitter)
    
    def create_control_panel(self):
        """Создание панели управления"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # === ПОДКЛЮЧЕНИЕ ===
        conn_group = QGroupBox("📡 Подключение к датчику")
        conn_layout = QVBoxLayout()
        
        # Порт
        port_layout = QHBoxLayout()
        port_layout.addWidget(QLabel("COM-порт:"))
        self.port_combo = QComboBox()
        self.refresh_ports()
        port_layout.addWidget(self.port_combo)
        
        refresh_btn = QPushButton("🔄")
        refresh_btn.setMaximumWidth(40)
        refresh_btn.clicked.connect(self.refresh_ports)
        port_layout.addWidget(refresh_btn)
        conn_layout.addLayout(port_layout)
        
        # Скорость
        baud_layout = QHBoxLayout()
        baud_layout.addWidget(QLabel("Скорость:"))
        self.baud_combo = QComboBox()
        self.baud_combo.addItems(['9600', '19200', '38400', '57600', '115200'])
        self.baud_combo.setCurrentText('9600')
        baud_layout.addWidget(self.baud_combo)
        conn_layout.addLayout(baud_layout)
        
        # Адрес
        addr_layout = QHBoxLayout()
        addr_layout.addWidget(QLabel("Адрес:"))
        self.address_spin = QSpinBox()
        self.address_spin.setRange(1, 127)
        self.address_spin.setValue(1)
        addr_layout.addWidget(self.address_spin)
        conn_layout.addLayout(addr_layout)
        
        conn_group.setLayout(conn_layout)
        layout.addWidget(conn_group)
        
        # === СБОР ДАННЫХ ===
        collect_group = QGroupBox("📊 Сбор данных")
        collect_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.start_btn = QPushButton("▶ Старт")
        self.start_btn.clicked.connect(self.start_collection)
        self.start_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        
        self.stop_btn = QPushButton("⏹ Стоп")
        self.stop_btn.clicked.connect(self.stop_collection)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet("background-color: #f44336; color: white; font-weight: bold; padding: 10px;")
        
        btn_layout.addWidget(self.start_btn)
        btn_layout.addWidget(self.stop_btn)
        collect_layout.addLayout(btn_layout)
        
        self.stats_label = QLabel("Готов к сбору данных")
        self.stats_label.setWordWrap(True)
        self.stats_label.setStyleSheet("background-color: #e3f2fd; padding: 10px; border-radius: 5px;")
        collect_layout.addWidget(self.stats_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setMaximum(0)
        self.progress_bar.hide()
        collect_layout.addWidget(self.progress_bar)
        
        collect_group.setLayout(collect_layout)
        layout.addWidget(collect_group)
        
        # === АНАЛИЗ ===
        analysis_group = QGroupBox("🔬 Параметры анализа")
        analysis_layout = QVBoxLayout()
        
        duration_layout = QHBoxLayout()
        duration_layout.addWidget(QLabel("Длительность (сек):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(0.1, 10.0)
        self.duration_spin.setValue(1.0)
        self.duration_spin.setSingleStep(0.1)
        self.duration_spin.setDecimals(1)
        duration_layout.addWidget(self.duration_spin)
        analysis_layout.addLayout(duration_layout)
        
        self.auto_analyze_check = QCheckBox("Автоматический анализ после сбора")
        self.auto_analyze_check.setChecked(True)
        analysis_layout.addWidget(self.auto_analyze_check)
        
        self.analyze_btn = QPushButton("🔍 Анализировать")
        self.analyze_btn.clicked.connect(self.run_analysis)
        self.analyze_btn.setEnabled(False)
        self.analyze_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 10px;")
        analysis_layout.addWidget(self.analyze_btn)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        # === ЭКСПОРТ ===
        export_group = QGroupBox("💾 Экспорт")
        export_layout = QVBoxLayout()
        
        self.export_btn = QPushButton("📦 Экспортировать всё")
        self.export_btn.clicked.connect(self.export_all)
        self.export_btn.setEnabled(False)
        self.export_btn.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 10px;")
        export_layout.addWidget(self.export_btn)
        
        self.open_folder_btn = QPushButton("📁 Открыть папку сессии")
        self.open_folder_btn.clicked.connect(self.open_session_folder)
        self.open_folder_btn.setEnabled(False)
        export_layout.addWidget(self.open_folder_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        # === ЗАГРУЗКА ===
        load_group = QGroupBox("📂 Загрузка данных")
        load_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("📄 Открыть CSV файл")
        self.load_btn.clicked.connect(self.load_file)
        load_layout.addWidget(self.load_btn)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        layout.addStretch()
        
        return panel
    
    def create_visualization_panel(self):
        """Создание панели визуализации"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        tabs = QTabWidget()
        
        # Вкладка 1: Живой график
        self.live_plot_widget = MatplotlibWidget()
        tabs.addTab(self.live_plot_widget, "📈 Живой график")
        
        # Вкладка 2: Анализ
        self.analysis_plot_widget = MatplotlibWidget()
        tabs.addTab(self.analysis_plot_widget, "🔬 Анализ")
        
        # Вкладка 3: Результаты
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(['Параметр', 'Значение'])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setAlternatingRowColors(True)
        results_layout.addWidget(self.results_table)
        
        tabs.addTab(results_tab, "📊 Результаты")
        
        # Вкладка 4: Лог
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet("background-color: #263238; color: #00ff00; font-family: 'Courier New';")
        log_layout.addWidget(self.log_console)
        
        clear_log_btn = QPushButton("🗑️ Очистить лог")
        clear_log_btn.clicked.connect(self.log_console.clear)
        log_layout.addWidget(clear_log_btn)
        
        tabs.addTab(log_tab, "📝 Лог")
        
        layout.addWidget(tabs)
        
        return panel
    
    def refresh_ports(self):
        """Обновление списка портов"""
        self.port_combo.clear()
        ports = serial.tools.list_ports.comports()
        for port in ports:
            self.port_combo.addItem(f"{port.device} - {port.description}", port.device)
        self.log(f"Найдено портов: {len(ports)}")
    
    def start_collection(self):
        """Начало сбора данных"""
        if self.port_combo.count() == 0:
            QMessageBox.warning(self, "Ошибка", "Нет доступных COM-портов")
            return
        
        port = self.port_combo.currentData()
        baudrate = int(self.baud_combo.currentText())
        address = self.address_spin.value()
        
        self.log(f"Начало сбора: порт={port}, скорость={baudrate}")
        
        self.collection_thread = DataCollectionThread(port, baudrate, address)
        self.collection_thread.data_received.connect(self.on_data_received)
        self.collection_thread.status_update.connect(self.on_status_update)
        self.collection_thread.error_occurred.connect(self.on_error)
        self.collection_thread.collection_finished.connect(self.on_collection_finished)
        
        self.live_data = []
        self.live_plot_widget.clear()
        
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.analyze_btn.setEnabled(False)
        self.export_btn.setEnabled(False)
        self.progress_bar.show()
        
        self.collection_thread.start()
        self.update_timer.start(100)
    
    def stop_collection(self):
        """Остановка сбора данных"""
        if self.collection_thread:
            self.log("Остановка сбора данных")
            self.collection_thread.stop()
            self.stop_btn.setEnabled(False)
    
    def on_data_received(self, data_point):
        """Обработка полученных данных"""
        self.live_data.append(data_point)
        stats = (f"📊 Точек: {len(self.live_data)}\n"
                f"⏱️ Время: {data_point['time']:.2f} сек\n"
                f"📏 Расстояние: {data_point['distance']:.3f} мм")
        self.stats_label.setText(stats)
    
    def update_live_plot(self):
        """Обновление живого графика"""
        if not self.live_data:
            return
        
        try:
            times = [p['time'] for p in self.live_data]
            distances = [p['distance'] for p in self.live_data]
            
            self.live_plot_widget.figure.clear()
            ax = self.live_plot_widget.figure.add_subplot(111)
            
            ax.plot(times, distances, 'b-', linewidth=1.5)
            ax.set_xlabel('Время (сек)')
            ax.set_ylabel('Расстояние (мм)')
            ax.set_title('Живой график данных', fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            self.live_plot_widget.canvas.draw()
        except:
            pass
    
    def on_status_update(self, message):
        """Обработка обновления статуса"""
        self.log(message)
    
    def on_error(self, error_message):
        """Обработка ошибок"""
        self.log(f"ОШИБКА: {error_message}")
        QMessageBox.critical(self, "Ошибка", error_message)
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()
        self.update_timer.stop()
    
    def on_collection_finished(self, filename):
        """Обработка завершения сбора"""
        self.log(f"Данные сохранены: {filename}")
        
        self.current_session_folder = os.path.dirname(filename)
        
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.hide()
        self.update_timer.stop()
        self.open_folder_btn.setEnabled(True)
        
        self.update_live_plot()
        
        if self.auto_analyze_check.isChecked():
            self.log("Запуск автоматического анализа")
            self.run_analysis_on_file(filename)
        else:
            self.analyze_btn.setEnabled(True)
            QMessageBox.information(self, "Успех", f"Данные собраны!\nТочек: {len(self.live_data)}")
    
    def run_analysis(self):
        """Запуск анализа"""
        if not self.current_session_folder:
            QMessageBox.warning(self, "Ошибка", "Нет данных для анализа")
            return
        
        filename = os.path.join(self.current_session_folder, "raw_data.csv")
        self.run_analysis_on_file(filename)
    
    def run_analysis_on_file(self, filename):
        """Анализ файла"""
        self.log(f"Начало анализа: {filename}")
        
        try:
            duration = self.duration_spin.value()
            success, message = self.analyzer.load_and_analyze(filename, duration)
            
            if not success:
                self.log(f"Ошибка анализа: {message}")
                QMessageBox.critical(self, "Ошибка", message)
                return
            
            self.log("Анализ выполнен успешно")
            
            results = self.analyzer.get_results()
            self.update_results_table(results)
            self.plot_analysis()
            self.save_analysis_results(results)
            
            self.export_btn.setEnabled(True)
            
            msg = "Анализ завершен!\n\n"
            if results['period']:
                msg += f"Период: {results['period']:.6f} сек\n"
                msg += f"Частота: {results['frequency']:.2f} Гц\n"
            if results['log_decrement']:
                msg += f"Лог. декремент: {results['log_decrement']:.6f}\n"
            QMessageBox.information(self, "Успех", msg)
            
        except Exception as e:
            error_msg = f"Ошибка анализа: {str(e)}"
            self.log(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def update_results_table(self, results):
        """Обновление таблицы результатов"""
        self.results_table.setRowCount(0)
        
        data = [
            ("Период (T)", f"{results['period']:.6f} сек" if results['period'] else "—"),
            ("Частота (f)", f"{results['frequency']:.2f} Гц" if results['frequency'] else "—"),
            ("Угловая частота (ω)", f"{2 * np.pi * results['frequency']:.2f} рад/с" if results['frequency'] else "—"),
            ("Лог. декремент (δ)", f"{results['log_decrement']:.6f}" if results['log_decrement'] else "—"),
            ("Коэфф. демпфирования (ζ)", f"{results['damping_ratio']:.6f}" if results['damping_ratio'] else "—"),
            ("Коэфф. потерь (η)", f"{results['loss_factor']:.6f}" if results['loss_factor'] else "—"),
            ("Количество пиков", str(results['num_peaks'])),
        ]
        
        for i, (param, value) in enumerate(data):
            self.results_table.insertRow(i)
            self.results_table.setItem(i, 0, QTableWidgetItem(param))
            value_item = QTableWidgetItem(value)
            value_item.setFont(QFont("Courier New", 10, QFont.Bold))
            self.results_table.setItem(i, 1, value_item)
        
        self.results_table.resizeColumnsToContents()
    
    def plot_analysis(self):
        """Построение графиков анализа"""
        try:
            self.analysis_plot_widget.figure.clear()
            
            time_data = self.analyzer.processed_data['Временная_метка'].values
            distance_data = self.analyzer.processed_data['Расстояние_норм'].values
            peaks = self.analyzer.peaks
            
            gs = self.analysis_plot_widget.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
            
            # График 1: Нормированные данные с пиками
            ax1 = self.analysis_plot_widget.figure.add_subplot(gs[0, :])
            ax1.plot(time_data, distance_data, 'g-', linewidth=1.5, label='Нормированное')
            
            if peaks is not None and len(peaks) > 0:
                ax1.plot(time_data[peaks], distance_data[peaks], 'ro', markersize=6, label=f'Пики ({len(peaks)})')
            
            results = self.analyzer.get_results()
            info_text = ''
            if results['period']:
                info_text = f"T = {results['period']:.6f} с\nf = {results['frequency']:.2f} Гц"
            if results['log_decrement']:
                info_text += f"\nδ = {results['log_decrement']:.6f}"
            
            if info_text:
                ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9), fontsize=9)
            
            ax1.set_xlabel('Время (сек)')
            ax1.set_ylabel('Расстояние (мм)')
            ax1.set_title('Затухающие колебания', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # График 2: Амплитуды
            if peaks is not None and len(peaks) >= 2:
                ax2 = self.analysis_plot_widget.figure.add_subplot(gs[1, 0])
                peak_times = time_data[peaks]
                peak_amplitudes = distance_data[peaks]
                
                ax2.plot(peak_times, peak_amplitudes, 'bo-', markersize=5)
                ax2.set_xlabel('Время (сек)')
                ax2.set_ylabel('Амплитуда (мм)')
                ax2.set_title('Амплитуды пиков', fontweight='bold')
                ax2.grid(True, alpha=0.3)
                
                # График 3: Логарифм
                ax3 = self.analysis_plot_widget.figure.add_subplot(gs[1, 1])
                log_amplitudes = np.log(np.abs(peak_amplitudes))
                ax3.plot(peak_times, log_amplitudes, 'ro-', markersize=5, label='ln(A)')
                
                if len(peak_times) >= 2:
                    coeffs = np.polyfit(peak_times, log_amplitudes, 1)
                    trend_line = np.polyval(coeffs, peak_times)
                    ax3.plot(peak_times, trend_line, 'r--', alpha=0.7, label=f'y = {coeffs[0]:.3f}x + {coeffs[1]:.3f}')
                
                ax3.set_xlabel('Время (сек)')
                ax3.set_ylabel('ln(Амплитуда)')
                ax3.set_title('Логарифм амплитуд', fontweight='bold')
                ax3.legend(fontsize=8)
                ax3.grid(True, alpha=0.3)
            
            self.analysis_plot_widget.canvas.draw()
            self.log("Графики построены")
            
        except Exception as e:
            self.log(f"Ошибка построения графиков: {str(e)}")
    
    def save_analysis_results(self, results):
        """Сохранение результатов"""
        if not self.current_session_folder:
            return
        
        try:
            results_file = os.path.join(self.current_session_folder, "analysis_results.txt")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("РЕЗУЛЬТАТЫ АНАЛИЗА ЗАТУХАЮЩИХ КОЛЕБАНИЙ RF603HS\n")
                f.write("=" * 70 + "\n")
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("ПАРАМЕТРЫ КОЛЕБАНИЙ:\n")
                f.write("-" * 70 + "\n")
                if results['period']:
                    f.write(f"Период (T):              {results['period']:.6f} сек\n")
                    f.write(f"Частота (f):             {results['frequency']:.2f} Гц\n")
                    f.write(f"Угловая частота (ω):     {2 * np.pi * results['frequency']:.2f} рад/с\n")
                
                f.write("\nПАРАМЕТРЫ ЗАТУХАНИЯ:\n")
                f.write("-" * 70 + "\n")
                if results['log_decrement']:
                    f.write(f"Лог. декремент (δ):      {results['log_decrement']:.6f}\n")
                if results['damping_ratio']:
                    f.write(f"Коэфф. демпфирования (ζ): {results['damping_ratio']:.6f}\n")
                if results['loss_factor']:
                    f.write(f"Коэфф. потерь (η):       {results['loss_factor']:.6f}\n")
                
                f.write("\nСТАТИСТИКА:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Количество пиков:        {results['num_peaks']}\n")
                f.write(f"Количество периодов:     {results['num_peaks'] - 1}\n")
                
                f.write("\n" + "=" * 70 + "\n")
            
            self.log(f"Результаты сохранены: {results_file}")
            
        except Exception as e:
            self.log(f"Ошибка сохранения результатов: {str(e)}")
    
    def export_all(self):
        """Экспорт всех данных"""
        if not self.current_session_folder:
            QMessageBox.warning(self, "Ошибка", "Нет данных для экспорта")
            return
        
        self.log("Начало экспорта")
        
        try:
            # Обработанные данные
            if self.analyzer.processed_data is not None:
                processed_file = os.path.join(self.current_session_folder, "processed_data.csv")
                self.analyzer.processed_data.to_csv(processed_file, sep=';', index=False, encoding='utf-8')
                self.log(f"✓ Обработанные данные: {processed_file}")
            
            # График живых данных
            if self.live_data:
                live_plot_file = os.path.join(self.current_session_folder, "live_plot.png")
                self.live_plot_widget.figure.savefig(live_plot_file, dpi=300, bbox_inches='tight')
                self.log(f"✓ График живых данных: {live_plot_file}")
            
            # Графики анализа
            analysis_plot_file = os.path.join(self.current_session_folder, "analysis_plot.png")
            self.analysis_plot_widget.figure.savefig(analysis_plot_file, dpi=300, bbox_inches='tight')
            self.log(f"✓ Графики анализа: {analysis_plot_file}")
            
            # Сводный отчет
            summary_file = os.path.join(self.current_session_folder, "summary_report.txt")
            self.create_summary_report(summary_file)
            
            self.log("✓ Экспорт завершен")
            
            QMessageBox.information(self, "Успех", 
                f"Все данные экспортированы!\n\nПапка: {self.current_session_folder}")
            
        except Exception as e:
            error_msg = f"Ошибка экспорта: {str(e)}"
            self.log(error_msg)
            QMessageBox.critical(self, "Ошибка", error_msg)
    
    def create_summary_report(self, filename):
        """Создание сводного отчета"""
        try:
            results = self.analyzer.get_results()
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("=" * 70 + "\n")
                f.write("           СВОДНЫЙ ОТЧЕТ ПО ИЗМЕРЕНИЯМ\n")
                f.write("                RF603HS Analyzer\n")
                f.write("=" * 70 + "\n\n")
                
                f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Папка: {os.path.basename(self.current_session_folder)}\n\n")
                
                f.write("-" * 70 + "\n")
                f.write("ПАРАМЕТРЫ ПОДКЛЮЧЕНИЯ:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Порт:     {self.port_combo.currentText()}\n")
                f.write(f"Скорость: {self.baud_combo.currentText()} бод\n")
                f.write(f"Адрес:    {self.address_spin.value()}\n\n")
                
                f.write("-" * 70 + "\n")
                f.write("СТАТИСТИКА СБОРА:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Точек собрано: {len(self.live_data)}\n")
                if self.live_data:
                    f.write(f"Время записи:  {self.live_data[-1]['time']:.3f} сек\n")
                f.write("\n")
                
                f.write("-" * 70 + "\n")
                f.write("РЕЗУЛЬТАТЫ АНАЛИЗА:\n")
                f.write("-" * 70 + "\n")
                if results['period']:
                    f.write(f"Период (T):              {results['period']:.6f} сек\n")
                    f.write(f"Частота (f):             {results['frequency']:.2f} Гц\n")
                if results['log_decrement']:
                    f.write(f"Лог. декремент (δ):      {results['log_decrement']:.6f}\n")
                if results['damping_ratio']:
                    f.write(f"Коэфф. демпфирования (ζ): {results['damping_ratio']:.6f}\n")
                if results['loss_factor']:
                    f.write(f"Коэфф. потерь (η):       {results['loss_factor']:.6f}\n")
                f.write(f"Количество пиков:        {results['num_peaks']}\n\n")
                
                f.write("-" * 70 + "\n")
                f.write("ФАЙЛЫ В СЕССИИ:\n")
                f.write("-" * 70 + "\n")
                for fname in os.listdir(self.current_session_folder):
                    fpath = os.path.join(self.current_session_folder, fname)
                    size = os.path.getsize(fpath) / 1024
                    f.write(f"• {fname:<40} {size:>8.1f} KB\n")
                
                f.write("\n" + "=" * 70 + "\n")
                f.write("Отчет создан автоматически | RF603HS Analyzer v1.0\n")
                f.write("=" * 70 + "\n")
            
            self.log(f"✓ Сводный отчет: {filename}")
            
        except Exception as e:
            self.log(f"Ошибка создания отчета: {str(e)}")
    
    def open_session_folder(self):
        """Открытие папки сессии"""
        if self.current_session_folder and os.path.exists(self.current_session_folder):
            import subprocess
            import platform
            
            try:
                if platform.system() == 'Windows':
                    os.startfile(self.current_session_folder)
                elif platform.system() == 'Darwin':
                    subprocess.Popen(['open', self.current_session_folder])
                else:
                    subprocess.Popen(['xdg-open', self.current_session_folder])
                
                self.log(f"Открыта папка: {self.current_session_folder}")
            except Exception as e:
                self.log(f"Ошибка открытия папки: {str(e)}")
    
    def load_file(self):
        """Загрузка файла"""
        filename, _ = QFileDialog.getOpenFileName(self, "Выберите CSV", "", "CSV Files (*.csv)")
        
        if filename:
            self.log(f"Загрузка: {filename}")
            self.current_session_folder = os.path.dirname(filename)
            self.open_folder_btn.setEnabled(True)
            self.run_analysis_on_file(filename)
    
    def closeEvent(self, event):
        """Закрытие приложения"""
        if self.collection_thread and self.collection_thread.isRunning():
            reply = QMessageBox.question(self, 'Подтверждение',
                "Сбор данных идет. Выйти?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.No:
                event.ignore()
                return
            
            self.collection_thread.stop()
            self.collection_thread.wait(3000)
        
        if self.log_file:
            self.log("Приложение закрыто")
            self.log_file.close()
        
        event.accept()


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    window = RF603AnalyzerGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()