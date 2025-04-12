import os
import pandas as pd
from typing import Optional, List, Dict, Any
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QComboBox, QTabWidget, QScrollArea, QSplitter, QGroupBox,
    QFormLayout, QSpinBox, QCheckBox, QLineEdit, QFileDialog,
    QDialog, QDialogButtonBox, QMessageBox, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QSize
from PyQt5.QtGui import QPixmap, QImage
import logging

# VisualizationEngine 임포트 추가
from src.visualization.visualization_engine import VisualizationEngine

class ImagePanel(QWidget):
    """
    이미지 표시 패널
    """
    def __init__(self, image_path: str, title: str = "", description: str = "", parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.title = title
        self.description = description
        self.initUI()
        
    def initUI(self):
        """UI 구성 요소 초기화"""
        layout = QVBoxLayout(self)
        
        # 이미지 파일이 존재하는지 확인
        if not os.path.exists(self.image_path):
            layout.addWidget(QLabel(f"이미지를 찾을 수 없음: {self.image_path}"))
            return
            
        # 제목 표시
        if self.title:
            title_label = QLabel(self.title)
            title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
            layout.addWidget(title_label)
            
        # 이미지 표시
        try:
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                max_width = 600
                max_height = 400
                if pixmap.width() > max_width or pixmap.height() > max_height:
                    pixmap = pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                image_label.setAlignment(Qt.AlignCenter)
                layout.addWidget(image_label)
            else:
                layout.addWidget(QLabel("이미지를 로드할 수 없습니다."))
        except Exception as e:
            layout.addWidget(QLabel(f"이미지 로드 오류: {str(e)}"))
            
        # 설명 표시
        if self.description:
            desc_label = QLabel(self.description)
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)
            
        # 파일 정보 표시
        file_info = QLabel(f"파일: {os.path.basename(self.image_path)}")
        file_info.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(file_info)
        
        # 버튼 레이아웃
        button_layout = QHBoxLayout()
        
        # 원본 크기로 보기 버튼
        original_button = QPushButton("원본 크기로 보기")
        original_button.clicked.connect(self.showOriginalSize)
        button_layout.addWidget(original_button)
        
        # 파일 저장 버튼
        save_button = QPushButton("저장")
        save_button.clicked.connect(self.saveImage)
        button_layout.addWidget(save_button)
        
        layout.addLayout(button_layout)
        
    def showOriginalSize(self):
        """원본 크기로 이미지 표시"""
        try:
            dialog = QDialog(self)
            dialog.setWindowTitle(self.title or "이미지 보기")
            
            layout = QVBoxLayout(dialog)
            
            # 스크롤 영역 생성
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            
            # 이미지 표시
            pixmap = QPixmap(self.image_path)
            if not pixmap.isNull():
                image_label = QLabel()
                image_label.setPixmap(pixmap)
                scroll.setWidget(image_label)
                
                # 적절한 크기로 다이얼로그 조정
                screen_size = dialog.screen().size()
                max_width = int(screen_size.width() * 0.8)
                max_height = int(screen_size.height() * 0.8)
                
                dialog_width = min(pixmap.width() + 50, max_width)
                dialog_height = min(pixmap.height() + 100, max_height)
                
                dialog.resize(dialog_width, dialog_height)
                
                layout.addWidget(scroll)
                
                # 닫기 버튼
                button_box = QDialogButtonBox(QDialogButtonBox.Close)
                button_box.rejected.connect(dialog.reject)
                layout.addWidget(button_box)
                
                dialog.exec_()
            else:
                QMessageBox.warning(self, "이미지 오류", "이미지를 로드할 수 없습니다.")
        except Exception as e:
            QMessageBox.critical(self, "오류", f"이미지 표시 중 오류 발생: {str(e)}")
            
    def saveImage(self):
        """이미지 파일 저장"""
        try:
            # 파일 저장 다이얼로그
            options = QFileDialog.Options()
            default_name = os.path.basename(self.image_path)
            file_path, _ = QFileDialog.getSaveFileName(
                self, "이미지 저장", default_name,
                "이미지 파일 (*.png *.jpg *.jpeg *.bmp);;모든 파일 (*)", options=options
            )
            
            if file_path:
                # 이미지 복사
                pixmap = QPixmap(self.image_path)
                if not pixmap.isNull():
                    pixmap.save(file_path)
                    QMessageBox.information(self, "저장 완료", f"이미지가 저장되었습니다: {file_path}")
                else:
                    QMessageBox.warning(self, "저장 오류", "이미지를 로드할 수 없습니다.")
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"이미지 저장 중 오류 발생: {str(e)}")

class VisualizationView(QWidget):
    """
    시각화 뷰 위젯
    """
    # 시그널 정의
    visualizationUpdated = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualization_engine = None
        self.visualization_files = []
        self.logger = logging.getLogger(__name__)
        self.initUI()
        
    def initUI(self):
        """UI 구성 요소 초기화"""
        # 메인 레이아웃
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # 상단 도구 모음
        top_layout = QHBoxLayout()
        layout.addLayout(top_layout)
        
        # 시각화 생성 버튼
        self.generate_button = QPushButton("시각화 생성")
        self.generate_button.clicked.connect(self.generateVisualizations)
        top_layout.addWidget(self.generate_button)
        
        # 시각화 유형 선택
        self.type_combo = QComboBox()
        self.type_combo.addItems(["모든 시각화", "분포", "관계", "시계열", "범주형", "주관식"])
        top_layout.addWidget(QLabel("유형:"))
        top_layout.addWidget(self.type_combo)
        
        # 시각화 설정 버튼
        self.settings_button = QPushButton("설정")
        self.settings_button.clicked.connect(self.showSettings)
        top_layout.addWidget(self.settings_button)
        
        top_layout.addStretch()
        
        # 새로고침 버튼
        self.refresh_button = QPushButton("새로고침")
        self.refresh_button.clicked.connect(self.updateVisualizations)
        top_layout.addWidget(self.refresh_button)
        
        # 시각화 탭 위젯
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # 기본 탭 생성
        self.all_tab = QWidget()
        self.tab_widget.addTab(self.all_tab, "모든 시각화")
        
        # 기본 탭 레이아웃
        self.all_scroll = QScrollArea()
        self.all_scroll.setWidgetResizable(True)
        self.all_scroll_content = QWidget()
        self.all_layout = QVBoxLayout(self.all_scroll_content)
        self.all_scroll.setWidget(self.all_scroll_content)
        
        all_tab_layout = QVBoxLayout(self.all_tab)
        all_tab_layout.addWidget(self.all_scroll)
        
        # 카테고리별 탭 생성
        self.tabs = {}
        
        for category in ["분포", "관계", "시계열", "범주형", "주관식"]:
            tab_widget, tab_layout = self.createTab(category)
            self.tabs[category] = (tab_widget, tab_layout)
            self.tab_widget.addTab(tab_widget, category)
            
        # 시각화 엔진 초기화 (MainWindow와 같은 출력 디렉토리 사용)
        # output_dir = os.path.join(os.getcwd(), "output", "visualizations")
        output_dir = os.path.join(os.getcwd(), "output", "viz")
        self.visualization_engine = VisualizationEngine(output_dir)
        
        # 초기화 로깅
        self.logger.info("시각화 뷰 초기화 완료")
        
    def createTab(self, name: str):
        """새 탭 생성"""
        tab = QWidget()
        self.tab_widget.addTab(tab, name)
        
        # 스크롤 영역 생성
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        # 스크롤 영역의 내용 위젯
        content = QWidget()
        layout = QVBoxLayout(content)
        
        scroll.setWidget(content)
        
        # 탭 레이아웃에 스크롤 영역 추가
        tab_layout = QVBoxLayout(tab)
        tab_layout.addWidget(scroll)
        
        return (tab, layout)
        
    def setVisualizationEngine(self, engine):
        """시각화 엔진 설정"""
        self.visualization_engine = engine
        
    def initializeWithData(self, dataframe):
        """데이터로 초기화"""
        # 시각화 엔진이 없으면 생성
        if self.visualization_engine is None:
            # 출력 디렉토리 설정 - MainWindow와 동일한 경로 사용
            output_dir = os.path.join(os.getcwd(), "output", "viz")
            self.visualization_engine = VisualizationEngine(output_dir)
            self.logger.info(f"시각화 엔진 초기화 (출력 경로: {output_dir})")
        
        self.visualization_engine.set_dataframe(dataframe)
        self.updateVisualizations()
        
    def updateVisualizations(self):
        """시각화 목록 업데이트"""
        if not self.visualization_engine:
            return
            
        # 이전 시각화 위젯 제거
        for _, layout in self.tabs.values():
            # 모든 위젯 제거
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        
        # 모든 시각화 탭 위젯 제거
        while self.all_layout.count():
            item = self.all_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                    
        # 시각화 파일 목록 초기화
        self.visualization_files = []
        
        try:
            # 시각화 디렉토리 확인
            viz_dir = self.visualization_engine.output_dir
            
            if os.path.exists(viz_dir):
                # 시각화 파일 목록 생성
                for filename in os.listdir(viz_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                        file_path = os.path.join(viz_dir, filename)
                        
                        # 파일 기본 정보
                        title = os.path.splitext(filename)[0].replace('_', ' ').title()
                        
                        # 카테고리 추정
                        category = "기타"
                        if 'keywords_' in filename or 'summary_' in filename:
                            category = "주관식"
                        elif 'hist' in filename or 'distribution' in filename:
                            category = "분포"
                        elif 'scatter' in filename or 'correlation' in filename or 'pair' in filename:
                            category = "관계"
                        elif 'time' in filename or 'trend' in filename:
                            category = "시계열"
                        elif 'bar' in filename or 'pie' in filename:
                            category = "범주형"
                            
                        # 시각화 파일 정보 추가
                        self.visualization_files.append({
                            'file_path': file_path,
                            'title': title,
                            'category': category,
                            'description': ""
                        })
                        
                # 시각화 패널 생성
                for viz_info in self.visualization_files:
                    panel = ImagePanel(
                        viz_info['file_path'], 
                        viz_info['title'], 
                        viz_info['description']
                    )
                    
                    # 모든 시각화 탭에 추가
                    self.all_layout.addWidget(panel)
                    
                    # 해당 카테고리 탭에 추가
                    category = viz_info['category']
                    if category in self.tabs:
                        _, tab_layout = self.tabs[category]
                        tab_layout.addWidget(panel)
                        
                # 각 카테고리별 스트레치 추가
                for _, layout in self.tabs.values():
                    layout.addStretch()
                    
                # 시그널 발생
                self.visualizationUpdated.emit()
                
        except Exception as e:
            QMessageBox.critical(self, "오류", f"시각화 업데이트 중 오류 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())
            
    def generateVisualizations(self):
        """시각화 생성"""
        if not self.visualization_engine:
            QMessageBox.warning(self, "시각화 오류", "시각화 엔진이 초기화되지 않았습니다.")
            return
            
        if not self.visualization_engine.get_dataframe() is not None:
            QMessageBox.warning(self, "시각화 오류", "시각화할 데이터가 없습니다.")
            return
            
        try:
            # 기본 시각화 수행
            viz_result = self.visualization_engine.export_all_visualizations()
            
            # 시각화 뷰 업데이트
            self.updateVisualizations()
            
            # 성공 메시지
            QMessageBox.information(self, "시각화 완료", "시각화가 생성되었습니다.")
            
        except Exception as e:
            QMessageBox.critical(self, "시각화 오류", f"시각화 생성 중 오류 발생: {str(e)}")
            
    def changeVisualizationType(self, category: str):
        """시각화 유형 변경"""
        # 해당 카테고리의 탭 보여주기
        for i in range(self.tab_widget.count()):
            if self.tab_widget.tabText(i) == category:
                self.tab_widget.setCurrentIndex(i)
                break
                
    def showSettings(self):
        """시각화 설정 다이얼로그 표시"""
        if not self.visualization_engine:
            QMessageBox.warning(self, "설정 오류", "시각화 엔진이 초기화되지 않았습니다.")
            return
            
        dialog = QDialog(self)
        dialog.setWindowTitle("시각화 설정")
        dialog.resize(400, 300)
        
        layout = QVBoxLayout(dialog)
        
        # 설정 폼
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        # 그림 크기 설정
        width_spin = QSpinBox()
        width_spin.setRange(1, 30)
        width_spin.setValue(12)
        form_layout.addRow("그림 너비(인치):", width_spin)
        
        height_spin = QSpinBox()
        height_spin.setRange(1, 30)
        height_spin.setValue(8)
        form_layout.addRow("그림 높이(인치):", height_spin)
        
        dpi_spin = QSpinBox()
        dpi_spin.setRange(50, 300)
        dpi_spin.setValue(100)
        form_layout.addRow("해상도(DPI):", dpi_spin)
        
        # 테마 설정
        theme_combo = QComboBox()
        theme_combo.addItems(["default", "darkgrid", "whitegrid", "dark", "white", "ticks"])
        form_layout.addRow("테마:", theme_combo)
        
        # 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Apply).setText("적용")
        button_box.button(QDialogButtonBox.Cancel).setText("취소")
        layout.addWidget(button_box)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # 다이얼로그 실행
        if dialog.exec() == QDialog.Accepted:
            try:
                # 설정 적용
                width = width_spin.value()
                height = height_spin.value()
                dpi = dpi_spin.value()
                theme = theme_combo.currentText()
                
                # 시각화 엔진 설정
                self.visualization_engine.set_figure_size(width, height, dpi)
                self.visualization_engine.set_theme(theme)
                
                QMessageBox.information(self, "설정 적용", "시각화 설정이 적용되었습니다.")
                
            except Exception as e:
                QMessageBox.critical(self, "설정 오류", f"설정 적용 중 오류 발생: {str(e)}")

    def updateVisualizationOptions(self):
        """시각화 유형에 따라 UI 업데이트"""
        try:
            if self.visualization_engine is None or self.visualization_engine.dataframe is None:
                print("시각화 옵션 업데이트 중 오류: 데이터가 로드되지 않았습니다.")
                return
            
            # 현재 선택된 시각화 유형
            viz_type = self.type_combo.currentText()
            
            # 컬럼 콤보박스 초기화
            self.x_column_combo.clear()
            self.y_column_combo.clear()
            
            # 데이터프레임 컬럼 추가
            columns = self.visualization_engine.dataframe.columns.tolist()
            self.x_column_combo.addItems(columns)
            
            # 시각화 유형에 따른 Y축 컬럼 옵션 설정
            if viz_type in ["관계", "막대 그래프"]:
                self.y_column_combo.setEnabled(True)
                self.y_column_combo.addItems(columns)
                self.y_column_label.setVisible(True)
                self.y_column_combo.setVisible(True)
            else:
                self.y_column_combo.setEnabled(False)
                self.y_column_label.setVisible(viz_type != "히트맵")
                self.y_column_combo.setVisible(viz_type != "히트맵")
            
            # X축 컬럼 라벨 숨김 (히트맵만)
            self.x_column_label.setVisible(viz_type != "히트맵")
            self.x_column_combo.setVisible(viz_type != "히트맵")
            
            print(f"시각화 옵션 업데이트 완료: {viz_type}")
            
        except Exception as e:
            import traceback
            print(f"시각화 옵션 업데이트 오류: {e}")
            print(traceback.format_exc())

    def addVisualization(self, image_path: str, title: str = "", description: str = "", category: str = "기타"):
        """
        시각화 패널 추가
        
        Args:
            image_path (str): 이미지 파일 경로
            title (str): 시각화 제목
            description (str): 시각화 설명
            category (str): 시각화 카테고리 (분포, 관계, 시계열, 범주형, 주관식)
        """
        try:
            if not os.path.exists(image_path):
                self.logger.warning(f"시각화 이미지 파일이 존재하지 않음: {image_path}")
                return
                
            # 시각화 파일 정보 추가
            self.visualization_files.append({
                'file_path': image_path,
                'title': title,
                'category': category,
                'description': description
            })
            
            # 시각화 패널 생성
            panel = ImagePanel(image_path, title, description)
            
            # 모든 시각화 탭에 추가
            self.all_layout.addWidget(panel)
            
            # 해당 카테고리 탭에 추가 (없으면 기타 카테고리로)
            if category in self.tabs:
                _, tab_layout = self.tabs[category]
                tab_layout.addWidget(panel)
            
            # 시그널 발생
            self.visualizationUpdated.emit()
            
            return True
        except Exception as e:
            self.logger.error(f"시각화 추가 중 오류: {str(e)}")
            return False 