import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableView, QHeaderView, QSplitter, QComboBox, QLineEdit,
    QFormLayout, QGroupBox, QCheckBox, QSpinBox, QDialog,
    QDialogButtonBox, QMessageBox
)
from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant, pyqtSignal
from PyQt5.QtGui import QColor, QFont

class DataTableModel(QAbstractTableModel):
    """
    판다스 데이터프레임을 위한 테이블 모델
    """
    def __init__(self, dataframe: Optional[pd.DataFrame] = None):
        super().__init__()
        self.dataframe = dataframe
        
    def rowCount(self, parent=None):
        if self.dataframe is None:
            return 0
        return len(self.dataframe)
        
    def columnCount(self, parent=None):
        if self.dataframe is None:
            return 0
        return len(self.dataframe.columns)
        
    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.dataframe is None:
            return QVariant()
            
        row, col = index.row(), index.column()
        
        if role == Qt.DisplayRole:
            value = self.dataframe.iloc[row, col]
            if pd.isna(value):
                return "NaN"
            elif isinstance(value, float):
                return f"{value:.4f}"
            elif isinstance(value, np.integer):
                return str(int(value))
            else:
                return str(value)
                
        elif role == Qt.TextAlignmentRole:
            value = self.dataframe.iloc[row, col]
            if isinstance(value, (int, float, np.number)):
                # 숫자 데이터는 오른쪽 정렬
                return Qt.AlignRight | Qt.AlignVCenter
            else:
                # 문자열은 왼쪽 정렬
                return Qt.AlignLeft | Qt.AlignVCenter
                
        elif role == Qt.BackgroundRole:
            value = self.dataframe.iloc[row, col]
            if pd.isna(value):
                # 결측치는 연한 빨간색 배경
                return QColor(255, 200, 200)
            return QVariant()
            
        return QVariant()
        
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return QVariant()
            
        if orientation == Qt.Horizontal:
            if self.dataframe is None:
                return QVariant()
            return str(self.dataframe.columns[section])
            
        if orientation == Qt.Vertical:
            if self.dataframe is None:
                return QVariant()
            return str(self.dataframe.index[section])
            
        return QVariant()
        
    def setData(self, dataframe):
        self.beginResetModel()
        self.dataframe = dataframe
        self.endResetModel()
        return True
        
    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable

class DataTableView(QWidget):
    """
    데이터 테이블 뷰 위젯
    """
    # 시그널 정의
    dataChanged = pyqtSignal(pd.DataFrame)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
    def initUI(self):
        """UI 구성 요소 초기화"""
        # 메인 레이아웃
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 상단 도구 모음
        top_layout = QHBoxLayout()
        layout.addLayout(top_layout)
        
        # 데이터 정보 레이블
        self.info_label = QLabel("데이터 없음")
        top_layout.addWidget(self.info_label)
        
        top_layout.addStretch()
        
        # 필터 버튼
        self.filter_button = QPushButton("필터")
        self.filter_button.clicked.connect(self.showFilterDialog)
        top_layout.addWidget(self.filter_button)
        
        # 열 관리 버튼
        self.columns_button = QPushButton("열 관리")
        self.columns_button.clicked.connect(self.showColumnDialog)
        top_layout.addWidget(self.columns_button)
        
        # 결측치 관리 버튼
        self.missing_button = QPushButton("결측치 처리")
        self.missing_button.clicked.connect(self.showMissingDialog)
        top_layout.addWidget(self.missing_button)
        
        # 테이블 뷰
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table_view)
        
        # 테이블 모델 설정
        self.model = DataTableModel()
        self.table_view.setModel(self.model)
        
    def setData(self, dataframe: pd.DataFrame):
        """데이터프레임 설정"""
        if dataframe is None or dataframe.empty:
            return
            
        # 모델 업데이트
        self.model.setData(dataframe)
        
        # 열 너비 자동 조정
        for col in range(self.model.columnCount()):
            self.table_view.resizeColumnToContents(col)
            
        # 정보 업데이트
        self.updateInfo()
        
        # 시그널 발생
        self.dataChanged.emit(dataframe)
        
    def updateInfo(self):
        """데이터 정보 레이블 업데이트"""
        if self.model.dataframe is None:
            self.info_label.setText("데이터 없음")
            return
            
        df = self.model.dataframe
        info_text = f"행: {len(df)}, 열: {len(df.columns)}"
        
        # 결측치 정보 추가
        missing_count = df.isna().sum().sum()
        if missing_count > 0:
            info_text += f", 결측치: {missing_count}"
            
        # 데이터 유형 정보 추가
        num_cols = len(df.select_dtypes(include=['number']).columns)
        cat_cols = len(df.select_dtypes(include=['object', 'category']).columns)
        date_cols = len(df.select_dtypes(include=['datetime']).columns)
        
        info_text += f" | 수치형: {num_cols}, 범주형: {cat_cols}, 날짜: {date_cols}"
        
        self.info_label.setText(info_text)
        
    def getData(self) -> Optional[pd.DataFrame]:
        """현재 모델에서 데이터프레임 가져오기"""
        return self.model.dataframe
        
    def showFilterDialog(self):
        """데이터 필터링 다이얼로그 표시"""
        if self.model.dataframe is None:
            QMessageBox.warning(self, "필터 오류", "필터링할 데이터가 없습니다.")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("데이터 필터링")
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # 필터 폼
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # 열 선택 콤보박스
        column_combo = QComboBox()
        column_combo.addItems(self.model.dataframe.columns)
        form_layout.addRow("열 선택:", column_combo)
        
        # 조건 선택 콤보박스
        condition_combo = QComboBox()
        condition_combo.addItems(["같음", "포함", "보다 큼", "보다 작음", "사이"])
        form_layout.addRow("조건:", condition_combo)
        
        # 값 입력
        value_input = QLineEdit()
        form_layout.addRow("값:", value_input)
        
        # 값2 입력 (사이 조건용)
        value2_input = QLineEdit()
        value2_input.setVisible(False)
        form_layout.addRow("값2:", value2_input)
        
        # 사이 조건에 대한 값2 표시 설정
        def onConditionChanged(index):
            value2_input.setVisible(condition_combo.currentText() == "사이")
        
        condition_combo.currentIndexChanged.connect(onConditionChanged)
        
        # 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Apply).setText("적용")
        button_box.button(QDialogButtonBox.Cancel).setText("취소")
        layout.addWidget(button_box)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # 다이얼로그 실행
        if dialog.exec() == QDialog.Accepted:
            column = column_combo.currentText()
            condition = condition_combo.currentText()
            value = value_input.text()
            
            try:
                # 기존 데이터프레임 복사
                df = self.model.dataframe.copy()
                
                # 필터링 조건 적용
                if condition == "같음":
                    try:
                        # 숫자로 변환 시도
                        numeric_value = float(value)
                        filtered_df = df[df[column] == numeric_value]
                    except ValueError:
                        # 문자열로 처리
                        filtered_df = df[df[column] == value]
                        
                elif condition == "포함":
                    filtered_df = df[df[column].astype(str).str.contains(value, na=False)]
                    
                elif condition == "보다 큼":
                    try:
                        numeric_value = float(value)
                        filtered_df = df[df[column] > numeric_value]
                    except (ValueError, TypeError):
                        QMessageBox.warning(self, "필터 오류", "숫자 형식의 값을 입력하세요.")
                        return
                        
                elif condition == "보다 작음":
                    try:
                        numeric_value = float(value)
                        filtered_df = df[df[column] < numeric_value]
                    except (ValueError, TypeError):
                        QMessageBox.warning(self, "필터 오류", "숫자 형식의 값을 입력하세요.")
                        return
                        
                elif condition == "사이":
                    value2 = value2_input.text()
                    try:
                        min_value = float(value)
                        max_value = float(value2)
                        filtered_df = df[(df[column] >= min_value) & (df[column] <= max_value)]
                    except (ValueError, TypeError):
                        QMessageBox.warning(self, "필터 오류", "숫자 형식의 값을 입력하세요.")
                        return
                        
                # 필터링 결과가 비어있는지 확인
                if filtered_df.empty:
                    QMessageBox.warning(self, "필터 결과", "필터링 결과가 없습니다.")
                    return
                    
                # 모델 업데이트
                self.setData(filtered_df)
                
            except Exception as e:
                QMessageBox.critical(self, "필터 오류", f"필터링 중 오류 발생: {str(e)}")
                
    def showColumnDialog(self):
        """열 관리 다이얼로그 표시"""
        if self.model.dataframe is None:
            QMessageBox.warning(self, "열 관리 오류", "관리할 데이터가 없습니다.")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("열 관리")
        dialog.resize(400, 400)
        
        layout = QVBoxLayout(dialog)
        
        # 열 선택 그룹박스
        columns_group = QGroupBox("표시할 열 선택")
        columns_layout = QVBoxLayout(columns_group)
        
        # 체크박스 생성
        checkboxes = []
        for column in self.model.dataframe.columns:
            checkbox = QCheckBox(column)
            checkbox.setChecked(True)
            columns_layout.addWidget(checkbox)
            checkboxes.append(checkbox)
            
        layout.addWidget(columns_group)
        
        # 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Apply).setText("적용")
        button_box.button(QDialogButtonBox.Cancel).setText("취소")
        layout.addWidget(button_box)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # 다이얼로그 실행
        if dialog.exec() == QDialog.Accepted:
            # 선택된 열만 포함하는 데이터프레임 생성
            selected_columns = [cb.text() for cb in checkboxes if cb.isChecked()]
            
            if not selected_columns:
                QMessageBox.warning(self, "열 관리 오류", "최소한 하나의 열을 선택해야 합니다.")
                return
                
            # 열 선택 적용
            filtered_df = self.model.dataframe[selected_columns]
            self.setData(filtered_df)
            
    def showMissingDialog(self):
        """결측치 처리 다이얼로그 표시"""
        if self.model.dataframe is None:
            QMessageBox.warning(self, "결측치 처리 오류", "처리할 데이터가 없습니다.")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("결측치 처리")
        dialog.resize(450, 300)
        
        layout = QVBoxLayout(dialog)
        
        # 결측치 정보 표시
        df = self.model.dataframe
        missing_info = df.isna().sum()
        missing_columns = missing_info[missing_info > 0]
        
        if len(missing_columns) == 0:
            layout.addWidget(QLabel("결측치가 없습니다."))
        else:
            layout.addWidget(QLabel("결측치가 있는 열:"))
            for col, count in missing_columns.items():
                layout.addWidget(QLabel(f"- {col}: {count}개 ({count/len(df)*100:.1f}%)"))
                
            layout.addWidget(QLabel("\n처리 방법 선택:"))
            
            # 처리 방법 폼
            form_layout = QFormLayout()
            layout.addLayout(form_layout)
            
            # 열 선택 콤보박스
            column_combo = QComboBox()
            column_combo.addItems(missing_columns.index)
            form_layout.addRow("처리할 열:", column_combo)
            
            # 방법 선택 콤보박스
            method_combo = QComboBox()
            method_combo.addItems(["제거", "평균으로 대체", "중앙값으로 대체", "최빈값으로 대체", "특정 값으로 대체"])
            form_layout.addRow("처리 방법:", method_combo)
            
            # 특정 값 입력
            value_input = QLineEdit()
            value_input.setEnabled(False)
            form_layout.addRow("대체 값:", value_input)
            
            # 특정 값 대체 선택 시 입력창 활성화
            def onMethodChanged(index):
                value_input.setEnabled(method_combo.currentText() == "특정 값으로 대체")
            
            method_combo.currentIndexChanged.connect(onMethodChanged)
            
            # 버튼
            button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
            button_box.button(QDialogButtonBox.Apply).setText("적용")
            button_box.button(QDialogButtonBox.Cancel).setText("취소")
            layout.addWidget(button_box)
            
            button_box.accepted.connect(dialog.accept)
            button_box.rejected.connect(dialog.reject)
            
            # 다이얼로그 실행
            if dialog.exec() == QDialog.Accepted:
                column = column_combo.currentText()
                method = method_combo.currentText()
                
                try:
                    # 기존 데이터프레임 복사
                    df_copy = df.copy()
                    
                    # 처리 방법 적용
                    if method == "제거":
                        df_copy = df_copy.dropna(subset=[column])
                        
                    elif method == "평균으로 대체":
                        if pd.api.types.is_numeric_dtype(df_copy[column]):
                            df_copy[column] = df_copy[column].fillna(df_copy[column].mean())
                        else:
                            QMessageBox.warning(self, "처리 오류", "평균값 대체는 숫자형 데이터에만 적용 가능합니다.")
                            return
                            
                    elif method == "중앙값으로 대체":
                        if pd.api.types.is_numeric_dtype(df_copy[column]):
                            df_copy[column] = df_copy[column].fillna(df_copy[column].median())
                        else:
                            QMessageBox.warning(self, "처리 오류", "중앙값 대체는 숫자형 데이터에만 적용 가능합니다.")
                            return
                            
                    elif method == "최빈값으로 대체":
                        mode_value = df_copy[column].mode().iloc[0]
                        df_copy[column] = df_copy[column].fillna(mode_value)
                        
                    elif method == "특정 값으로 대체":
                        value = value_input.text()
                        if pd.api.types.is_numeric_dtype(df_copy[column]):
                            try:
                                numeric_value = float(value)
                                df_copy[column] = df_copy[column].fillna(numeric_value)
                            except ValueError:
                                QMessageBox.warning(self, "처리 오류", "숫자형 데이터에는 숫자 값만 입력 가능합니다.")
                                return
                        else:
                            df_copy[column] = df_copy[column].fillna(value)
                            
                    # 모델 업데이트
                    self.setData(df_copy)
                    
                except Exception as e:
                    QMessageBox.critical(self, "처리 오류", f"결측치 처리 중 오류 발생: {str(e)}") 