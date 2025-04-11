import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Union
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTabWidget, QTableView, QHeaderView, QSplitter,
    QTreeView, QTextEdit, QFrame, QScrollArea, QMessageBox, QFileDialog, QDialog
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal
from PyQt5.QtGui import QStandardItemModel, QStandardItem, QColor, QFont, QBrush
import logging
import json
import os
from datetime import datetime

# 주관식 텍스트 미리보기 패널 임포트
from src.gui.text_preview_panel import TextPreviewPanel

# 로거 설정
logger = logging.getLogger(__name__)

class ResultTableModel(QAbstractTableModel):
    """
    분석 결과 테이블 모델
    """
    def __init__(self, data: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.load_data(data)
        
    def load_data(self, data: Optional[Dict[str, Any]]):
        """데이터 로드"""
        self.header_labels = []
        self.table_data = []
        
        # 로그 디렉토리 확인
        os.makedirs("logs", exist_ok=True)
        
        # 데이터 로딩 과정 로그 파일에 기록
        try:
            with open("logs/data_parsing.log", "a", encoding="utf-8") as log_file:
                log_file.write("\n\n===== 데이터 파싱 시작 =====\n")
                log_file.write(f"데이터 타입: {type(data)}\n")
                if data is None:
                    log_file.write("데이터 없음 (None)\n")
                elif isinstance(data, (int, float, str, bool)):
                    log_file.write(f"스칼라 값: {data}\n")
                elif isinstance(data, dict):
                    log_file.write(f"딕셔너리 키 목록: {list(data.keys())}\n")
                    # 각 키의 값 타입 기록
                    for key in data.keys():
                        value_type = type(data[key]).__name__
                        value_info = ""
                        if isinstance(data[key], dict):
                            sub_keys = list(data[key].keys())
                            value_info = f", 키 수: {len(sub_keys)}, 샘플 키: {sub_keys[:3]}"
                        elif isinstance(data[key], (list, tuple)):
                            value_info = f", 길이: {len(data[key])}"
                        log_file.write(f"  - 키: {key}, 값 타입: {value_type}{value_info}\n")
                elif isinstance(data, (list, tuple)):
                    log_file.write(f"리스트/튜플, 길이: {len(data)}\n")
                    if data and len(data) > 0:
                        log_file.write(f"첫 번째 항목 타입: {type(data[0]).__name__}\n")
                elif isinstance(data, pd.DataFrame):
                    log_file.write(f"DataFrame, 행: {len(data)}, 열: {len(data.columns)}\n")
                    log_file.write(f"열 목록: {list(data.columns)}\n")
                else:
                    log_file.write(f"기타 데이터 타입: {type(data).__name__}\n")
        except Exception as e:
            logger.error(f"로그 파일 기록 중 오류: {str(e)}")
        
        if data is None:
            return
            
        # 기본 타입 처리 (숫자, 문자열 등)
        if isinstance(data, (int, float, str, bool, np.number)):
            self.header_labels = ['Value']
            self.table_data = [[data]]
            return
            
        # 결과 유형에 따라 테이블 형식으로 변환
        if isinstance(data, dict):
            # 상관관계 매트릭스 특수 처리
            if "type" in data and data["type"] == "correlation_matrix":
                if "header" in data and "data" in data:
                    self.header_labels = data["header"]
                    self.table_data = data["data"]
                    return
            
            if 'mean' in data and isinstance(data['mean'], dict):
                # 기술 통계의 경우 열 이름과 통계량으로 구성
                stats = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
                available_stats = [s for s in stats if s in data]
                
                # 헤더 설정
                self.header_labels = ['Column'] + available_stats
                
                # 값 설정
                columns = list(data['mean'].keys()) if 'mean' in data else []
                
                for col in columns:
                    row_data = [col]
                    for stat in available_stats:
                        if col in data[stat]:
                            row_data.append(data[stat][col])
                        else:
                            row_data.append(None)
                    self.table_data.append(row_data)
            elif 'pearson' in data and isinstance(data['pearson'], dict):
                # 상관관계 분석
                columns = list(data['pearson'].keys())
                
                # 헤더 설정
                self.header_labels = ['Column'] + columns
                
                # 값 설정
                for col1 in columns:
                    row_data = [col1]
                    for col2 in columns:
                        if col2 in data['pearson'][col1]:
                            row_data.append(data['pearson'][col1][col2])
                        else:
                            row_data.append(None)
                    self.table_data.append(row_data)
            else:
                # 일반적인 키-값 쌍
                self.header_labels = ['Key', 'Value']
                for key, value in data.items():
                    if isinstance(value, dict):
                        # 중첩 딕셔너리는 딕셔너리인 경우에만 문자열로 표시, 작은 딕셔너리는 전체 표시
                        if len(value) > 10:  # 10개 이상의 항목이 있는 경우 요약
                            self.table_data.append([key, f"{type(value).__name__} ({len(value)} 항목)"])
                        else:
                            # 각 키-값 쌍을 별도의 행으로 추가
                            self.table_data.append([key, ""])
                            for sub_key, sub_value in value.items():
                                self.table_data.append([f"  {sub_key}", sub_value])
                    elif isinstance(value, (pd.DataFrame, np.ndarray)) and hasattr(value, 'shape'):
                        # DataFrame이나 배열 요약 정보
                        shape_str = f"({', '.join(map(str, value.shape))})" if hasattr(value, 'shape') else ""
                        self.table_data.append([key, f"{type(value).__name__} {shape_str}"])
                    elif isinstance(value, list):
                        # 리스트는 길이에 따라 처리
                        if len(value) > 10:  # 긴 리스트는 요약 표시
                            self.table_data.append([key, f"List ({len(value)} 항목)"])
                        else:
                            # 짧은 리스트는 모든 항목 표시
                            self.table_data.append([key, str(value)])
                    else:
                        # 그 외 데이터 타입은 그대로 표시
                        self.table_data.append([key, value])
        elif isinstance(data, list):
            # 리스트 형태의 데이터
            if not data:  # 빈 리스트인 경우
                self.header_labels = ["상태"]
                self.table_data = [["빈 리스트입니다"]]
            elif isinstance(data[0], dict):
                # 딕셔너리 리스트의 경우 (예: 레코드 목록)
                self.header_labels = ['Index'] + list(data[0].keys())
                for i, item in enumerate(data):
                    row_data = [i]
                    for key in data[0].keys():
                        row_data.append(item.get(key))
                    self.table_data.append(row_data)
            else:
                # 일반 리스트
                self.header_labels = ['Index', 'Value']
                for i, value in enumerate(data):
                    self.table_data.append([i, value])
        elif isinstance(data, pd.DataFrame):
            # DataFrame 처리
            self.header_labels = [''] + list(data.columns)
            for idx, row in data.iterrows():
                self.table_data.append([idx] + list(row.values))
        elif isinstance(data, np.ndarray):
            # NumPy 배열 처리
            if data.ndim == 1:
                self.header_labels = ['Index', 'Value']
                for i, value in enumerate(data):
                    self.table_data.append([i, value])
            elif data.ndim == 2:
                self.header_labels = [''] + [f'Col {i}' for i in range(data.shape[1])]
                for i in range(data.shape[0]):
                    self.table_data.append([f'Row {i}'] + list(data[i]))
            else:
                self.header_labels = ['Shape', 'Type']
                self.table_data.append([str(data.shape), str(data.dtype)])
        else:
            # 처리할 수 없는 데이터 유형인 경우
            self.header_labels = ['Type', 'Description']
            self.table_data = [[type(data).__name__, "지원되지 않는 데이터 유형입니다."]]
        
    def rowCount(self, parent=None):
        return len(self.table_data)
        
    def columnCount(self, parent=None):
        return len(self.header_labels)
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self.table_data) or index.column() >= len(self.header_labels):
            return QVariant()
            
        value = self.table_data[index.row()][index.column()]
        
        if role == Qt.DisplayRole:
            if value is None:
                return "N/A"
            elif isinstance(value, float):
                # 10점 척도와 같은 항목은 정수로 표시(예: 3.0 -> 3)
                if value.is_integer():
                    return str(int(value))
                # 소수점이 있는 경우 적절한 자리수로 표시
                return f"{value:.2f}"
            elif isinstance(value, (int, np.integer)):
                # 정수는 그대로 표시
                return str(value)
            else:
                return str(value)
                
        elif role == Qt.TextAlignmentRole:
            if index.column() == 0:  # 첫 번째 열(이름)은 왼쪽 정렬
                return Qt.AlignLeft | Qt.AlignVCenter
            elif isinstance(value, (int, float, np.number)):  # 숫자는 오른쪽 정렬
                return Qt.AlignRight | Qt.AlignVCenter
            else:
                return Qt.AlignLeft | Qt.AlignVCenter
                
        elif role == Qt.BackgroundRole:
            # 상관관계 행렬인 경우 히트맵 스타일 색상
            if "pearson" in self.header_labels[0].lower() or "correlation" in self.header_labels[0].lower():
                if index.row() > 0 and index.column() > 0:
                    if value is not None and isinstance(value, (int, float)):
                        # 상관계수에 따른 색상 결정
                        if value > 0.8:  # 강한 양의 상관관계
                            return QBrush(QColor(255, 100, 100, 150))
                        elif value > 0.5:  # 중간 양의 상관관계
                            return QBrush(QColor(255, 150, 150, 150))
                        elif value > 0.3:  # 약한 양의 상관관계
                            return QBrush(QColor(255, 200, 200, 150))
                        elif value < -0.8:  # 강한 음의 상관관계
                            return QBrush(QColor(100, 100, 255, 150))
                        elif value < -0.5:  # 중간 음의 상관관계
                            return QBrush(QColor(150, 150, 255, 150))
                        elif value < -0.3:  # 약한 음의 상관관계
                            return QBrush(QColor(200, 200, 255, 150))
            
            # 교대로 행 배경색 변경
            if index.row() % 2 == 0:
                return QBrush(QColor(245, 245, 245))
                
        return QVariant()
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
            
        if orientation == Qt.Horizontal and section < len(self.header_labels):
            return self.header_labels[section]
            
        if orientation == Qt.Vertical:
            return section + 1
            
        return QVariant()
        
    def setResultData(self, data):
        """테이블에 결과 데이터 설정 및 레이아웃 갱신"""
        self.load_data(data)
        self.layoutChanged.emit()

class AnalysisView(QWidget):
    """
    분석 결과 뷰 위젯
    """
    visualizeRequested = pyqtSignal(str, dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.analysis_engine = None
        self.current_results = {}
        self.current_selection = None
        self.initUI()
        
    def initUI(self):
        """UI 구성 요소 초기화"""
        # 메인 레이아웃
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 상단 도구 모음
        top_layout = QHBoxLayout()
        layout.addLayout(top_layout)
        
        # 분석 유형 선택
        self.analysis_combo = QComboBox()
        self.analysis_combo.addItems([
            "기술 통계", "상관관계 분석", "그룹별 통계", "결측치 분석"
        ])
        top_layout.addWidget(QLabel("분석 유형:"))
        top_layout.addWidget(self.analysis_combo)
        
        # 분석 실행 버튼
        self.run_button = QPushButton("분석 실행")
        self.run_button.clicked.connect(self.runAnalysis)
        top_layout.addWidget(self.run_button)
        
        top_layout.addStretch()
        
        # 결과 내보내기 버튼 추가
        self.export_button = QPushButton("결과 내보내기")
        self.export_button.clicked.connect(self.exportResults)
        self.export_button.setEnabled(False)  # 초기에는 비활성화
        top_layout.addWidget(self.export_button)
        
        # 결과 새로고침 버튼
        self.refresh_button = QPushButton("새로고침")
        self.refresh_button.clicked.connect(self.updateResults)
        top_layout.addWidget(self.refresh_button)
        
        # 메인 콘텐츠 영역
        self.splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(self.splitter)
        
        # 왼쪽 패널 - 결과 트리
        self.result_tree = QTreeView()
        self.result_tree.setHeaderHidden(True)
        self.tree_model = QStandardItemModel()
        self.result_tree.setModel(self.tree_model)
        self.result_tree.clicked.connect(self.onResultItemClicked)
        # 키보드 이벤트 연결 추가
        self.result_tree.selectionModel().selectionChanged.connect(self.onResultItemSelected)
        self.splitter.addWidget(self.result_tree)
        
        # 오른쪽 패널 - 결과 뷰
        self.right_panel = QWidget()
        right_layout = QVBoxLayout(self.right_panel)
        
        # 결과 제목
        self.result_title = QLabel("선택된 결과 없음")
        self.result_title.setStyleSheet("font-weight: bold; font-size: 14px;")
        right_layout.addWidget(self.result_title)
        
        # 결과 상세 정보 라벨 추가
        self.result_detail_label = QLabel("")
        self.result_detail_label.setStyleSheet("color: #555; font-size: 12px;")
        right_layout.addWidget(self.result_detail_label)
        
        # 결과 설명
        self.result_description = QLabel("")
        self.result_description.setWordWrap(True)
        right_layout.addWidget(self.result_description)
        
        # 구분선
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        right_layout.addWidget(line)
        
        # 결과 테이블
        self.result_table = QTableView()
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table_model = ResultTableModel()
        self.result_table.setModel(self.result_table_model)
        right_layout.addWidget(self.result_table)
        
        # 주관식 텍스트 미리보기 패널 (처음엔 숨김 상태)
        self.text_preview_panel = TextPreviewPanel()
        self.text_preview_panel.visualizationRequested.connect(self.visualizeTextColumn)
        self.text_preview_panel.hide()
        right_layout.addWidget(self.text_preview_panel)
        
        self.splitter.addWidget(self.right_panel)
        
        # 초기 스플리터 비율 설정
        self.splitter.setSizes([200, 600])
        
    def setAnalysisEngine(self, engine):
        """분석 엔진 설정"""
        self.analysis_engine = engine
        
    def initializeWithData(self, dataframe):
        """데이터프레임으로 초기화"""
        if self.analysis_engine is None:
            from src.core.analysis_engine import AnalysisEngine
            self.analysis_engine = AnalysisEngine()
        
        self.analysis_engine.set_dataframe(dataframe)
        self.updateResults()
        
    def updateResults(self):
        """분석 결과 업데이트"""
        if not self.analysis_engine:
            logger.error("분석 엔진이 설정되지 않았습니다")
            return
            
        # 현재 분석 결과 가져오기
        self.current_results = self.analysis_engine.get_all_results()
        
        # 로그 기록
        try:
            with open("logs/results_update.log", "a", encoding="utf-8") as log_file:
                log_file.write("\n===== 분석 결과 업데이트 =====\n")
                log_file.write(f"결과 타입: {type(self.current_results).__name__}\n")
                log_file.write(f"결과 키: {list(self.current_results.keys())}\n")
                
                # 각 최상위 결과에 대한 세부 정보 기록
                for key, value in self.current_results.items():
                    log_file.write(f"\n- 결과 키: {key}, 값 타입: {type(value).__name__}\n")
                    if isinstance(value, dict):
                        log_file.write(f"  서브키 목록: {list(value.keys())}\n")
                        
                        # 몇 개의 샘플 서브키 값 기록
                        sample_keys = list(value.keys())[:3]  # 최대 3개
                        for sub_key in sample_keys:
                            sub_value = value[sub_key]
                            log_file.write(f"  - 서브키: {sub_key}, 값 타입: {type(sub_value).__name__}\n")
        except Exception as e:
            logger.error(f"결과 업데이트 로깅 중 오류: {str(e)}")
        
        # 결과 트리 모델 업데이트
        self.tree_model.clear()
        root = self.tree_model.invisibleRootItem()
        
        # 분석 타입별로 처리
        analysis_type = self.analysis_combo.currentText()
        
        if not self.current_results:
            # 결과가 없는 경우
            no_result_item = QStandardItem("결과 없음")
            no_result_item.setEnabled(False)
            root.appendRow(no_result_item)
            logger.warning("분석 결과가 없습니다")
            return
            
        # 결과 타입 목록 정렬
        result_types = sorted(self.current_results.keys())
        
        for result_type in result_types:
            result_data = self.current_results[result_type]
            
            # 결과 타입에 맞는 아이템 생성
            type_item = QStandardItem(result_type)
            
            # 데이터 경로 설정 (트리 항목에 저장되어 나중에 결과를 찾는 데 사용)
            type_item.setData(result_type, Qt.UserRole)
            
            # 결과 데이터 유형에 따라 다른 처리
            if isinstance(result_data, dict) and result_data:
                # 딕셔너리 타입 결과는 하위 항목으로 표시
                # 결과 하위 항목 정렬 (일관된 표시를 위해)
                sorted_keys = sorted(result_data.keys())
                
                for key in sorted_keys:
                    sub_data = result_data[key]
                    sub_item = QStandardItem(str(key))
                    
                    # 하위 항목 데이터 경로 설정 (상위.하위 형식)
                    sub_item.setData(f"{result_type}.{key}", Qt.UserRole)
                    
                    # 하위 항목에 실제 데이터 설정 - 명시적 형변환 추가
                    if key == "numeric_columns" or key == "categorical_columns":
                        # 리스트 항목을 확실하게 데이터로 설정
                        logger.info(f"리스트 항목 추가: {key}, 항목 수: {len(sub_data) if isinstance(sub_data, list) else 'N/A'}")
                        sub_item.setData(sub_data, Qt.UserRole + 1)
                    else:
                        sub_item.setData(sub_data, Qt.UserRole + 1)
                    
                    # 하위 항목 추가
                    type_item.appendRow(sub_item)
                    
                    # 디버그 로깅
                    logger.debug(f"결과 하위 항목 추가: {key}, 데이터 경로: {result_type}.{key}")
            
            # 최상위 항목 추가
            root.appendRow(type_item)
            
            # 최상위 항목에 실제 데이터 설정
            type_item.setData(result_data, Qt.UserRole + 1)
            
            # 디버그 로깅
            logger.debug(f"결과 최상위 항목 추가: {result_type}")
            
        # 결과 트리 확장
        self.result_tree.expandAll()
        
        # 결과 테이블 초기화
        self.result_table_model.load_data(None)
        self.result_table_model.layoutChanged.emit()
        
        # 상태 메시지 업데이트
        result_count = len(self.current_results)
        logger.info(f"{result_count}개의 분석 결과가 로드되었습니다")
        
    def runAnalysis(self):
        """선택한 분석 유형 실행"""
        if not self.analysis_engine:
            return
            
        analysis_type = self.analysis_combo.currentText()
        
        try:
            # 모든 분석 유형에 대해 analyze 메서드 사용
            if self.analysis_engine.df is not None:
                self.analysis_engine.analyze(self.analysis_engine.df)
                # 결과 업데이트
                self.updateResults()
                
                # 첫 번째 결과 항목 자동 선택
                first_index = self.tree_model.index(0, 0)
                if first_index.isValid():
                    self.result_tree.setCurrentIndex(first_index)
                    self.onResultItemClicked(first_index)
            else:
                print("분석할 데이터가 없습니다.")
            
        except Exception as e:
            print(f"분석 실행 중 오류: {str(e)}")
            
    def onResultItemClicked(self, index):
        """결과 항목 클릭 시 호출되는 메서드"""
        try:
            # 로그 디렉토리 확인
            os.makedirs("logs", exist_ok=True)
            
            # 클릭된 아이템 가져오기
            item = self.tree_model.itemFromIndex(index)
            if not item:
                logger.warning("클릭된 항목을 찾을 수 없습니다")
                return
            
            # 로깅 강화: 클릭된 아이템 정보
            logger.info(f"클릭된 항목: {item.text()}, 행: {index.row()}, 열: {index.column()}")
            
            # 아이템 데이터 경로 가져오기 (Qt.UserRole)
            item_path = item.data(Qt.UserRole)
            
            # 아이템 실제 데이터 가져오기 (Qt.UserRole + 1)
            item_data = item.data(Qt.UserRole + 1)
            
            # 제목과 기본 라벨 업데이트
            self.result_title.setText(item.text())
            self.result_detail_label.setText(f"선택된 항목: {item.text()}")
            
            # 주관식 미리보기 패널 초기 숨김
            self.text_preview_panel.hide()
            self.result_table.show()
            
            # 주관식 텍스트 데이터 여부 확인
            if item_path == "text_analysis":
                # 클릭된 것이 text_analysis 노드인 경우
                logger.info("주관식 텍스트 분석 결과 그룹 선택됨")
                
                # 기본 설명 업데이트
                self.result_description.setText("주관식 응답에 대한 텍스트 분석 결과입니다. 분석된 열을 선택하여 상세 결과를 확인하세요.")
                
                # 자식 항목이 있는 경우 첫 번째 자식 자동 선택
                if item.hasChildren() and item.child(0):
                    first_child_index = self.tree_model.indexFromItem(item.child(0))
                    if first_child_index.isValid():
                        self.result_tree.setCurrentIndex(first_child_index)
                        self.onResultItemClicked(first_child_index)
                return
            
            # 주관식 데이터 항목 특수 처리
            if item_path and item_path.startswith("text_analysis."):
                try:
                    # text_analysis.열이름 형식에서 열 이름 추출
                    column_name = item_path.split('.')[1]
                    logger.info(f"주관식 열 '{column_name}' 선택됨")
                    
                    # 주관식 데이터 가져오기
                    text_data = None
                    if 'text_analysis' in self.current_results and column_name in self.current_results['text_analysis']:
                        text_data = self.current_results['text_analysis'][column_name]
                        # 디버깅을 위해 text_data 구조 로깅 추가
                        logger.info(f"주관식 데이터 구조: {type(text_data)}")
                        if isinstance(text_data, dict):
                            logger.info(f"주관식 데이터 키: {list(text_data.keys())}")
                            
                            # 응답 데이터 확인 (responses 또는 samples 키 중 하나 사용)
                            if 'responses' in text_data:
                                logger.info(f"'responses' 키에서 응답 데이터 확인: {len(text_data['responses'])}개")
                            elif 'samples' in text_data:
                                logger.info(f"'samples' 키에서 응답 데이터 확인: {len(text_data['samples'])}개 (대체 키)")
                            else:
                                logger.error(f"필수 키 'responses' 또는 'samples'가 주관식 데이터에 없습니다.")
                        else:
                            logger.error(f"주관식 데이터가 dict 형식이 아닙니다: {type(text_data)}")
                    
                    if text_data:
                        # 테이블 숨기고 미리보기 패널 표시
                        self.result_table.hide()
                        self.text_preview_panel.show()
                        
                        # 텍스트 데이터 설정
                        self.text_preview_panel.set_text_data(column_name, text_data)
                        
                        # 결과 설명 업데이트
                        self.result_description.setText(f"'{column_name}' 열의 주관식 응답 분석 결과입니다.")
                        
                        # 내보내기 버튼 활성화
                        self.export_button.setEnabled(True)
                        
                        return
                except Exception as e:
                    logger.error(f"주관식 데이터 처리 중 오류: {str(e)}")
            
            # DataFrame 데이터 특별 처리 (correlation 항목 등)
            if item_path == "correlation" or (item_data is not None and isinstance(item_data, pd.DataFrame)):
                try:
                    # 상관관계 매트릭스 처리
                    corr_data = self.current_results.get("correlation")
                    if corr_data is not None and isinstance(corr_data, pd.DataFrame):
                        logger.info(f"상관관계 매트릭스 처리: 행 {corr_data.shape[0]}, 열 {corr_data.shape[1]}")
                        
                        # 데이터프레임을 2차원 리스트로 변환 - 안전한 방법
                        data_rows = []
                        # 헤더 행 추가
                        header_row = ["변수"] + list(corr_data.columns)
                        data_rows.append(header_row)
                        
                        # 각 행 추가
                        for idx, row in corr_data.iterrows():
                            row_data = [idx]  # 행 이름
                            for col in corr_data.columns:
                                try:
                                    val = row[col]
                                    if isinstance(val, float):
                                        val = round(val, 3)  # 소수점 3자리로 반올림
                                    row_data.append(val)
                                except:
                                    row_data.append("N/A")
                            data_rows.append(row_data)
                            
                        # 데이터를 직접 테이블 모델에 설정 가능한 형태로 준비
                        item_data = {
                            "header": header_row,
                            "data": data_rows[1:],  # 헤더 행 제외
                            "type": "correlation_matrix"
                        }
                except Exception as e:
                    logger.error(f"상관관계 데이터 변환 중 오류: {str(e)}")
                    
            if not item_data and item_path:
                # 데이터 경로가 있지만 데이터가 없는 경우 경로를 사용하여 데이터 가져오기
                try:
                    path_parts = item_path.split('.')
                    current_data = self.current_results
                    
                    for part in path_parts:
                        if part in current_data:
                            current_data = current_data[part]
                        else:
                            logger.warning(f"데이터 경로 '{item_path}'에서 키 '{part}'를 찾을 수 없습니다")
                            return
                    
                    item_data = current_data
                except Exception as e:
                    logger.error(f"데이터 경로 '{item_path}' 처리 중 오류: {str(e)}")
                    return
            
            if not item_data:
                logger.warning("선택한 항목에 데이터가 없습니다")
                self.result_detail_label.setText("선택한 값이 없습니다")
                return
                
            # DataFrame 특별 처리 - 데이터프레임은 불리언 컨텍스트에서 오류 발생
            if isinstance(item_data, pd.DataFrame):
                if item_data.empty:
                    logger.warning("선택한 DataFrame이 비어 있습니다")
                    self.result_detail_label.setText("데이터가 없습니다")
                    return
                    
                # DataFrame을 딕셔너리로 변환하여 처리
                logger.info(f"DataFrame 감지: 행 {item_data.shape[0]}, 열 {item_data.shape[1]}")
                # to_dict 메서드로 변환
                dict_data = {
                    "행 수": item_data.shape[0],
                    "열 수": item_data.shape[1],
                    "열 이름": list(item_data.columns),
                    "데이터": item_data.to_dict(orient='records')
                }
                item_data = dict_data
                
            # 로깅 강화: 아이템 데이터 상세 정보
            logger.info(f"아이템 데이터 유형: {type(item_data)}")
            
            try:
                # 테이블 모델에 데이터 설정 및 로깅
                with open("logs/item_data.log", "a", encoding="utf-8") as log_file:
                    log_file.write(f"\n===== 클릭된 항목 데이터: {datetime.now()} =====\n")
                    log_file.write(f"항목 텍스트: {item.text()}\n")
                    log_file.write(f"항목 데이터 유형: {type(item_data)}\n")
                    # 리스트 형식인 경우 항목 개수 추가 로깅
                    if isinstance(item_data, list):
                        log_file.write(f"리스트 항목 개수: {len(item_data)}\n")
                        log_file.write(f"리스트 항목: {item_data}\n")
                    else:
                        log_file.write(f"항목 데이터: {str(item_data)}\n")
                
                # 테이블 모델에 데이터 설정
                self.result_table_model.setResultData(item_data)
                
                # 라벨 업데이트
                if isinstance(item_data, dict) and "name" in item_data:
                    self.result_detail_label.setText(f"선택된 항목: {item_data['name']}")
                elif isinstance(item_data, list):
                    # 리스트인 경우 항목 개수 표시
                    self.result_detail_label.setText(f"선택된 항목: {item.text()} ({len(item_data)}개 항목)")
                
                # 내보내기 버튼 활성화
                self.export_button.setEnabled(True)
                
            except Exception as e:
                logger.error(f"테이블 모델 데이터 설정 중 오류: {str(e)}")
                
                # 오류 시 빈 테이블로 설정
                self.result_table_model.header_labels = ["오류"]
                self.result_table_model.table_data = [["데이터 표시 중 오류 발생"]]
                self.result_table_model.layoutChanged.emit()
                
                # 오류 메시지 표시
                self.result_description.setText(f"데이터 표시 중 오류 발생: {str(e)}")
                
                # 내보내기 버튼 비활성화
                self.export_button.setEnabled(False)
                
        except Exception as e:
            # 예외처리
            logger.error(f"결과 항목 클릭 처리 중 오류: {str(e)}")
            
            # 오류 추적 정보 로깅
            with open("logs/error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n===== 결과 항목 클릭 오류: {datetime.now()} =====\n")
                log_file.write(f"오류 메시지: {str(e)}\n")
                
                import traceback
                traceback_info = traceback.format_exc()
                log_file.write(f"추적 정보:\n{traceback_info}\n")
            
            self.result_table_model.header_labels = ["오류"]
            self.result_table_model.table_data = [["데이터 접근 중 오류가 발생했습니다"]]
            self.result_table_model.layoutChanged.emit()
            
            # 제목과 설명 업데이트
            if item:
                self.result_title.setText(f"{item.text()} - 오류")
                self.result_description.setText(f"데이터 접근 중 오류: {str(e)}")
            else:
                self.result_title.setText("오류 발생")
                self.result_description.setText(f"데이터 접근 중 오류: {str(e)}")
            
            self.export_button.setEnabled(False)
            
            # GUI 메시지 박스는 불필요한 사용자 중단을 초래할 수 있으므로 로그만 기록
            logger.error(f"데이터 접근 중 오류: {str(e)}")

    def onResultItemSelected(self, selected, deselected):
        """키보드 선택 변경 시 호출되는 메서드"""
        try:
            # 로그 디렉토리 확인
            os.makedirs("logs", exist_ok=True)
            
            # 선택된 항목의 인덱스 가져오기
            selected_indexes = selected.indexes()
            if not selected_indexes:
                logger.warning("선택된 항목이 없습니다")
                return
                
            # 첫 번째 선택된 항목 처리
            index = selected_indexes[0]
            
            # 로그에 선택 이벤트 기록
            logger.info(f"키보드로 선택된 항목 인덱스: 행={index.row()}, 열={index.column()}")
            
            # 클릭 이벤트와 동일하게 처리
            self.onResultItemClicked(index)
                
        except Exception as e:
            # 오류 로깅
            logger.error(f"결과 항목 선택 처리 중 오류: {str(e)}")
            
            # 오류 추적 정보 로깅
            with open("logs/error.log", "a", encoding="utf-8") as log_file:
                log_file.write(f"\n===== 키보드 선택 오류: {datetime.now()} =====\n")
                log_file.write(f"오류 메시지: {str(e)}\n")
                
                import traceback
                traceback_info = traceback.format_exc()
                log_file.write(f"추적 정보:\n{traceback_info}\n")

    def exportResults(self):
        """현재 분석 결과를 파일로 내보내기"""
        try:
            import pandas as pd
            import json
            from PyQt5.QtWidgets import QFileDialog
            import os
            
            # 로그 기록
            logger.info("결과 내보내기 시작")
            
            # 현재 선택된 항목 확인
            selected_indexes = self.result_tree.selectedIndexes()
            if not selected_indexes:
                logger.warning("내보낼 항목이 선택되지 않았습니다")
                return
                
            index = selected_indexes[0]
            item = self.tree_model.itemFromIndex(index)
            
            if not item:
                logger.warning("선택된 항목을 찾을 수 없습니다")
                return
                
            # 항목 데이터 및 텍스트 가져오기
            item_data = item.data(Qt.UserRole + 1)  # 항목 실제 데이터
            item_path = item.data(Qt.UserRole)  # 항목 경로
            item_text = item.text()
            
            logger.info(f"내보내기 선택 항목: {item_text}, 경로: {item_path}")
            
            # 주관식 텍스트 데이터 내보내기 특별 처리
            is_text_data = False
            text_df = None
            
            if item_path and item_path.startswith("text_analysis."):
                # 텍스트 미리보기 패널에서 보여지는 데이터 내보내기
                logger.info("주관식 텍스트 데이터 내보내기 시도")
                if hasattr(self, 'text_preview_panel') and self.text_preview_panel.data is not None:
                    text_df = self.text_preview_panel.data
                    logger.info(f"텍스트 미리보기 데이터프레임 획득: {text_df.shape}")
                    is_text_data = True
                else:
                    logger.warning("텍스트 미리보기 패널의 데이터를 가져올 수 없습니다.")
            
            # 파일 저장 다이얼로그
            options = QFileDialog.Options()
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "결과 저장", 
                f"{item_text}.csv", 
                "CSV 파일 (*.csv);;Excel 파일 (*.xlsx);;JSON 파일 (*.json)",
                options=options
            )
            
            if not file_path:
                logger.info("파일 저장이 취소되었습니다")
                return
                
            # 저장 디렉토리 확인
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # 파일 확장자 추출
            file_ext = os.path.splitext(file_path)[1].lower()
            
            # 주관식 텍스트 데이터 내보내기
            if is_text_data and text_df is not None:
                logger.info("주관식 텍스트 데이터 내보내기 처리 중...")
                
                # 파일 형식별 저장
                if file_ext == '.csv':
                    text_df.to_csv(file_path, index=False, encoding='utf-8-sig')
                    logger.info(f"텍스트 데이터 CSV 파일로 저장됨: {file_path}")
                elif file_ext == '.xlsx':
                    text_df.to_excel(file_path, index=False, sheet_name=item_text[:31])  # 시트 이름 제한
                    logger.info(f"텍스트 데이터 Excel 파일로 저장됨: {file_path}")
                elif file_ext == '.json':
                    # JSON 파일로 저장
                    text_json = text_df.to_dict(orient='records')
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(text_json, f, ensure_ascii=False, indent=2)
                    logger.info(f"텍스트 데이터 JSON 파일로 저장됨: {file_path}")
                else:
                    logger.error(f"지원되지 않는 파일 형식: {file_ext}")
                
                # 성공적인 내보내기 후 반환
                return
            
            # 일반 테이블 데이터 내보내기 (기존 로직)
            # 테이블 데이터를 데이터프레임으로 변환
            header = self.result_table_model.header_labels
            data = self.result_table_model.table_data
            
            if not data:
                logger.warning("내보낼 데이터가 없습니다")
                return
                
            # 데이터프레임 생성
            df = pd.DataFrame(data, columns=header)
            
            # 파일 형식별 저장
            if file_ext == '.csv':
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                logger.info(f"CSV 파일로 저장됨: {file_path}")
            elif file_ext == '.xlsx':
                df.to_excel(file_path, index=False, sheet_name=item_text[:31])  # 시트 이름 제한
                logger.info(f"Excel 파일로 저장됨: {file_path}")
            elif file_ext == '.json':
                # JSON 파일로 저장
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f"JSON 파일로 저장됨: {file_path}")
            else:
                logger.error(f"지원되지 않는 파일 형식: {file_ext}")
                
        except Exception as e:
            logger.error(f"결과 내보내기 중 오류: {str(e)}")
            # 오류 추적 정보 로깅
            import traceback
            traceback_info = traceback.format_exc()
            logger.error(f"결과 내보내기 오류 추적: {traceback_info}")

    def visualizeTextColumn(self, data_frame):
        """주관식 열에 대한 시각화 요청 처리 (TextPreviewPanel의 새 API 호환)"""
        if data_frame is not None and isinstance(data_frame, pd.DataFrame):
            # DataFrame에서 첫 번째 열 이름을 추출하여 시각화 요청
            if not data_frame.empty and len(data_frame.columns) > 0:
                column_name = data_frame.columns[0]
                
                # 텍스트 항목 추출 (비어있지 않은 문자열만)
                text_items = data_frame[column_name].dropna().astype(str).tolist()
                text_items = [item for item in text_items if item and item.strip()]
                
                if not text_items:
                    logger.warning(f"'{column_name}' 열에 시각화할 텍스트 항목이 없습니다.")
                    return
                
                # 데이터 수집 (시각화용)
                data = {
                    'dataframe': data_frame,
                    'column': column_name,
                    'text_items': text_items
                }
                
                # 시각화 요청 이벤트 발생
                logger.info(f"'{column_name}' 열에 대한 시각화 요청 - 항목 {len(text_items)}개")
                self.visualizeRequested.emit(column_name, data)
            else:
                logger.warning("빈 DataFrame으로 시각화 요청이 취소되었습니다")
        else:
            logger.warning("시각화에 필요한 데이터가 전달되지 않았습니다")