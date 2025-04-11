from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                            QCheckBox, QPushButton, QTableWidget, 
                            QTableWidgetItem, QHeaderView, QWidget)
from PyQt5.QtCore import Qt
import pandas as pd
import logging

class ColumnSettingsDialog(QDialog):
    """
    사용자가 주관식 응답을 포함하는 열을 선택할 수 있는 다이얼로그
    """
    
    def __init__(self, df: pd.DataFrame, parent=None):
        """
        ColumnSettingsDialog 초기화
        
        Args:
            df (pd.DataFrame): 설정할 데이터프레임
            parent: 부모 위젯
        """
        super().__init__(parent)
        self.df = df
        self.subjective_columns = []  # 주관식 열 저장
        self.logger = logging.getLogger(__name__)
        
        self.setWindowTitle("열 설정")
        self.setMinimumWidth(700)
        self.setMinimumHeight(500)
        
        self.init_ui()
        
    def init_ui(self):
        """UI 초기화"""
        layout = QVBoxLayout()
        
        # 안내 메시지
        info_label = QLabel("주관식 응답이 포함된 열을 선택하세요. 주관식 응답은 텍스트 분석 및 요약 보고서로 생성됩니다.")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # 테이블 설정
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["열 이름", "주관식 응답", "데이터 예시"])
        
        # 열 데이터 추가
        self._populate_table()
        
        # 테이블 너비 조정
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.horizontalHeader().setStretchLastSection(True)
        
        layout.addWidget(self.table)
        
        # 버튼 추가
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("모두 선택")
        select_none_btn = QPushButton("모두 해제")
        ok_btn = QPushButton("확인")
        cancel_btn = QPushButton("취소")
        
        select_all_btn.clicked.connect(self._select_all)
        select_none_btn.clicked.connect(self._select_none)
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        
        btn_layout.addWidget(select_all_btn)
        btn_layout.addWidget(select_none_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        
        layout.addLayout(btn_layout)
        self.setLayout(layout)
        
    def _populate_table(self):
        """테이블에 데이터프레임 열 정보 채우기"""
        try:
            if self.df is None or self.df.empty:
                self.logger.warning("테이블을 채울 데이터가 없습니다.")
                return
                
            self.table.setRowCount(len(self.df.columns))
            
            for i, col in enumerate(self.df.columns):
                # 열 이름
                col_item = QTableWidgetItem(str(col))
                col_item.setFlags(col_item.flags() & ~Qt.ItemIsEditable)  # 편집 불가능하게 설정
                self.table.setItem(i, 0, col_item)
                
                # 주관식 체크박스
                checkbox = QCheckBox()
                checkbox.setChecked(False)  # 기본값은 체크 해제
                checkbox_cell = QHBoxLayout()
                checkbox_cell.addWidget(checkbox)
                checkbox_cell.setAlignment(Qt.AlignCenter)
                checkbox_cell.setContentsMargins(0, 0, 0, 0)
                
                checkbox_widget = QWidget()
                checkbox_widget.setLayout(checkbox_cell)
                self.table.setCellWidget(i, 1, checkbox_widget)
                
                # 데이터 샘플
                sample_data = ""
                if not self.df[col].dropna().empty:
                    sample_value = self.df[col].dropna().iloc[0]
                    sample_data = str(sample_value)
                    if len(sample_data) > 50:  # 긴 텍스트 자르기
                        sample_data = sample_data[:50] + "..."
                
                sample_item = QTableWidgetItem(sample_data)
                sample_item.setFlags(sample_item.flags() & ~Qt.ItemIsEditable)  # 편집 불가능하게 설정
                self.table.setItem(i, 2, sample_item)
                
                # 자동 감지 - 텍스트가 긴 열은 주관식으로 가정
                if self.df[col].dtype == 'object':
                    # 평균 텍스트 길이 계산
                    text_lengths = self.df[col].astype(str).str.len()
                    avg_len = text_lengths.mean()
                    max_len = text_lengths.max()
                    
                    # 평균 길이가 25자 이상이거나 최대 길이가 100자 이상이면 주관식 추정
                    if avg_len > 25 or max_len > 100:
                        checkbox.setChecked(True)
                
        except Exception as e:
            self.logger.error(f"테이블 채우기 오류: {e}")
    
    def _select_all(self):
        """모든 열을 주관식으로 선택"""
        for i in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(i, 1)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(True)
    
    def _select_none(self):
        """모든 주관식 선택 해제"""
        for i in range(self.table.rowCount()):
            checkbox_widget = self.table.cellWidget(i, 1)
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox:
                    checkbox.setChecked(False)
        
    def get_subjective_columns(self):
        """주관식으로 체크된 열 이름 목록 반환"""
        subjective_cols = []
        for i in range(self.table.rowCount()):
            col_name = self.table.item(i, 0).text()
            checkbox_widget = self.table.cellWidget(i, 1)
            
            if checkbox_widget:
                checkbox = checkbox_widget.findChild(QCheckBox)
                if checkbox and checkbox.isChecked():
                    subjective_cols.append(col_name)
        
        return subjective_cols 