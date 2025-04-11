import os
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QListWidget, QListWidgetItem, QTextEdit, QScrollArea,
    QFrame, QSplitter, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class TextItemWidget(QWidget):
    """텍스트 항목을 표시하는 위젯 (인용 스타일)"""
    
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self.text = text
        self.initUI()
        self.setText(text)
    
    def initUI(self):
        """UI 초기화"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 상단 인용 부호 레이아웃
        top_layout = QHBoxLayout()
        
        # 인용 부호 아이콘
        quote_label = QLabel(''')
        quote_label.setStyleSheet("""
                QLabel {
                    font-size: 28px;
                    color: #5c85d6;
                    font-family: 'Malgun Gothic';
                    font-weight: bold;
                    padding-bottom: 5px;
                }
        """)
        top_layout.addWidget(quote_label)
        top_layout.addStretch()
        
        main_layout.addLayout(top_layout)
        
        # 텍스트 표시 영역
        self.text_display = QTextEdit()
        self.text_display.setReadOnly(True)
        self.text_display.setStyleSheet("""
                QTextEdit {
                    background-color: #f8f9fa;
                    border: none;
                    border-radius: 5px;
                    padding: 10px;
                    color: #333;
                    font-family: 'Malgun Gothic';
                    font-size: 13px;
                    line-height: 1.5;
                }
        """)
        main_layout.addWidget(self.text_display)
        
        # 하단 레이아웃
        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        
        # 닫는 인용 부호
        close_quote_label = QLabel(''')
        close_quote_label.setStyleSheet("""
                QLabel {
                    font-size: 28px;
                    color: #5c85d6;
                    font-family: 'Malgun Gothic';
                    font-weight: bold;
                    padding-top: 5px;
                }
        """)
        bottom_layout.addWidget(close_quote_label)
        
        main_layout.addLayout(bottom_layout)
        
        # 구분선 추가
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("""
                QFrame {
                    background-color: #e6e6e6;
                    max-height: 1px;
                    margin-top: 5px;
                }
        """)
        main_layout.addWidget(separator)
        
        self.setStyleSheet("""
            TextItemWidget {
                background-color: white;
                margin-bottom: 5px;
            }
        """)
    
    def setText(self, text):
        """텍스트 설정 및 크기 조정"""
        if not text:
            text = "(내용 없음)"
        
        self.text = text
        self.text_display.setPlainText(text)
        
        # 줄 수에 따라 높이 조절
        line_count = text.count('\n') + 1
        # 최소 2줄, 최대 12줄
        adjusted_line_count = max(2, min(12, line_count))
        
        # 줄당 20픽셀 + 패딩 20픽셀
        text_height = adjusted_line_count * 20 + 20
        self.text_display.setMinimumHeight(text_height)
        self.text_display.setMaximumHeight(text_height)
        
        # 스크롤바 필요 여부 설정
        self.text_display.setVerticalScrollBarPolicy(
            Qt.ScrollBarAsNeeded if line_count > 12 else Qt.ScrollBarAlwaysOff
        )

class TextPreviewPanel(QWidget):
    """텍스트 응답을 표시하는 패널"""
    
    visualizationRequested = pyqtSignal(pd.DataFrame)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # 데이터 초기화
        self.text_items = []
        self.data = None
        # UI 초기화
        self.initUI()
        # 초기화 완료 로그
        logger.info("텍스트 미리보기 패널 초기화됨")
        
    def initUI(self):
        """UI 초기화"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        
        # 제목과 시각화 버튼
        top_layout = QHBoxLayout()
        
        title_label = QLabel("주관식 응답 미리보기")
        title_label.setFont(QFont("Malgun Gothic", 12, QFont.Bold))
        title_label.setStyleSheet("color: #333;")
        top_layout.addWidget(title_label)
        
        top_layout.addStretch()
        
        # 시각화 버튼 추가
        self.visualize_btn = QPushButton("워드클라우드 생성")
        self.visualize_btn.setIcon(QIcon("resources/icons/chart.png"))
        self.visualize_btn.setMaximumWidth(150)
        self.visualize_btn.setStyleSheet("""
            QPushButton {
                background-color: #5c85d6;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #4a6db3;
            }
            QPushButton:pressed {
                background-color: #3a5a99;
            }
        """)
        self.visualize_btn.clicked.connect(self.requestVisualization)
        top_layout.addWidget(self.visualize_btn)
        
        main_layout.addLayout(top_layout)
        
        # 스크롤 영역 설정
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea {
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
        """)
        
        # 스크롤 영역에 들어갈 컨테이너 위젯
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(15, 15, 15, 15)
        self.scroll_layout.setSpacing(10)
        
        # 안내 메시지 추가
        self.empty_label = QLabel("선택된 항목이 없습니다.\n분석 결과에서 주관식 응답을 선택하세요.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet("""
                QLabel {
                    color: #888;
                    font-size: 13px;
                    padding: 20px;
                }
        """)
        self.scroll_layout.addWidget(self.empty_label)
        
        # 스트레치 추가 (내용이 적을 때 빈 공간 처리)
        self.scroll_layout.addStretch()
        
        scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(scroll_area, 1)  # 스크롤 영역이 확장되도록 설정
    
    def displayTextItems(self, items: List[str], data: Optional[pd.DataFrame] = None):
        """텍스트 항목 목록 표시"""
        # 이전 항목 정리
        self.clearItems()
        
        # 데이터 저장
        self.data = data
        
        if data is not None:
            logger.info(f"데이터프레임 설정됨: {data.shape[0]}행, {data.shape[1]}열")
        else:
            logger.warning("displayTextItems에 전달된 데이터프레임이 None입니다.")
        
        if not items or len(items) == 0:
            logger.warning("표시할 텍스트 항목이 없습니다.")
            self.empty_label.setVisible(True)
            self.visualize_btn.setEnabled(False)
            return
        
        self.empty_label.setVisible(False)
        self.visualize_btn.setEnabled(True if data is not None else False)
        
        # 빈 텍스트 필터링
        valid_items = [item for item in items if item and isinstance(item, str) and item.strip()]
        if len(valid_items) != len(items):
            logger.warning(f"빈 항목 또는 문자열이 아닌 항목 {len(items) - len(valid_items)}개가 필터링되었습니다.")
        
        # 표시할 항목 수 정보 추가
        count_info = QLabel(f"표시된 응답: {len(valid_items)} / 전체: {len(items)}개")
        count_info.setStyleSheet("color: #666; font-size: 12px; margin-bottom: 10px;")
        count_info.setAlignment(Qt.AlignRight)
        self.scroll_layout.insertWidget(0, count_info)
        
        # 각 텍스트 항목에 대해 위젯 생성 및 추가
        # 텍스트 항목 최대 50개까지만 표시
        display_items = valid_items[:50]
        for text in display_items:
            text_widget = TextItemWidget(text)
            self.text_items.append(text_widget)
            self.scroll_layout.insertWidget(self.scroll_layout.count() - 1, text_widget)
        
        logger.info(f"텍스트 미리보기 패널에 {len(display_items)}개 항목 표시됨")
    
    def clearItems(self):
        """기존 항목 모두 제거"""
        # 스트레치 항목을 제외한 모든 위젯 제거
        while self.scroll_layout.count() > 1:  # 마지막 스트레치 항목 유지
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # 데이터 및 상태 초기화
        self.text_items = []
        self.data = None
        self.empty_label.setVisible(True)
        self.scroll_layout.insertWidget(0, self.empty_label)
        self.visualize_btn.setEnabled(False)
        
        logger.info("텍스트 미리보기 패널 항목이 모두 제거되었습니다.")
    
    def set_text_data(self, column_name, text_data):
        """텍스트 데이터 설정 및 UI 업데이트 (기존 API 호환성 유지)
        
        Args:
            column_name (str): 표시할 열 이름
            text_data (dict): 텍스트 데이터 {'summary': {...}, 'responses': [...]} 또는 {'samples': [...]}
        """
        logger.info(f"{column_name} 열의 텍스트 데이터 로드 시작")
        
        # 텍스트 데이터 구조 검증
        if not isinstance(text_data, dict):
            logger.error(f"text_data가 dict 형식이 아닙니다: {type(text_data)}")
            self.clearItems()
            return
            
        # 응답 데이터 확인
        responses = None
        if 'responses' in text_data:
            responses = text_data['responses']
            logger.info(f"'responses' 키에서 {len(responses)}개 항목 발견")
        elif 'samples' in text_data:
            responses = text_data['samples']
            logger.info(f"'samples' 키에서 {len(responses)}개 항목 발견 (대체 키)")
        else:
            logger.error(f"'responses' 또는 'samples' 키가 text_data에 없습니다. 가용한 키: {list(text_data.keys())}")
            self.clearItems()
            return
            
        # 데이터 프레임 생성 (시각화용)
        if responses and len(responses) > 0:
            # 유효한 데이터인지 확인
            if not all(isinstance(r, str) for r in responses):
                logger.warning("일부 응답이 문자열이 아닙니다. 문자열로 변환합니다.")
                responses = [str(r) if r is not None else "" for r in responses]
                
            # 데이터프레임 생성
            df = pd.DataFrame({column_name: responses})
            self.data = df  # 시각화용 데이터 설정
            logger.info(f"데이터프레임 생성 완료: {len(responses)}행, 1열")
            
            # 응답 항목 표시
            self.displayTextItems(responses, df)
            
            # 상단 메시지 추가 (응답 수, 평균 길이 등)
            response_count = len(responses)
            
            if 'summary' in text_data:
                summary = text_data['summary']
                avg_length = summary.get('avg_length', 0)
                
                # 정보 레이블 추가
                if self.scroll_layout.count() > 1:
                    info_label = QLabel(f"'{column_name}' 열 분석: 총 {response_count}개 응답 / 평균 길이: {avg_length:.1f}자")
                    info_label.setStyleSheet("""
                        background-color: #f0f7ff; 
                        color: #333; 
                        padding: 8px; 
                        border-radius: 4px; 
                        border-left: 3px solid #5c85d6;
                        font-weight: bold;
                    """)
                    self.scroll_layout.insertWidget(0, info_label)
            elif 'avg_length' in text_data:
                # summary가 없는 경우 직접 avg_length 사용
                avg_length = text_data.get('avg_length', 0)
                response_count = text_data.get('response_count', len(responses))
                
                # 정보 레이블 추가
                if self.scroll_layout.count() > 1:
                    info_label = QLabel(f"'{column_name}' 열 분석: 총 {response_count}개 응답 / 평균 길이: {avg_length:.1f}자")
                    info_label.setStyleSheet("""
                        background-color: #f0f7ff; 
                        color: #333; 
                        padding: 8px; 
                        border-radius: 4px; 
                        border-left: 3px solid #5c85d6;
                        font-weight: bold;
                    """)
                    self.scroll_layout.insertWidget(0, info_label)
            
            logger.info(f"{column_name} 열의 텍스트 데이터 로드 완료: {response_count}개 응답")
        else:
            # 응답 데이터가 없는 경우
            self.clearItems()
            self.data = None
            logger.warning(f"{column_name} 열에 표시할 텍스트 응답이 없습니다")
    
    def requestVisualization(self):
        """시각화 생성 요청"""
        if self.data is not None:
            logger.info("텍스트 데이터 시각화 요청됨")
            self.visualizationRequested.emit(self.data)
        else:
            logger.error("시각화할 데이터가 없습니다. self.data가 None입니다.")
            # 현재 상태 로깅
            logger.error(f"text_items 개수: {len(self.text_items) if self.text_items else 0}")
            logger.error("텍스트 항목이 로드되었으나 데이터프레임이 설정되지 않았습니다.")
            QMessageBox.warning(self, "시각화 오류", "시각화할 데이터가 없습니다.")