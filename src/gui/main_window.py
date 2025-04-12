import os
import sys
import logging
from typing import Optional, Dict, List, Any
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QTabWidget, QSplitter,
    QTableView, QMessageBox, QAction, QToolBar, QStatusBar, 
    QDialog, QLineEdit, QFormLayout, QComboBox, QCheckBox,
    QDialogButtonBox, QGroupBox, QMenu, QFrame, QTreeView, QRadioButton,
    QProgressBar
)
from PyQt5.QtCore import Qt, QSize, pyqtSignal, QModelIndex, QSettings, QThread
from PyQt5.QtGui import QIcon, QFont, QStandardItemModel, QStandardItem

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
root_dir = os.path.dirname(parent_dir)
sys.path.insert(0, root_dir)

# 이제 상대 경로 임포트 사용
try:
    # 각종 경로에서 모듈 임포트 시도
    try:
        from src.reporting.report_integrator import ReportIntegrator
        from src.utils.data_loader import DataLoader
        from src.core.data_processor import DataProcessor
        from src.core.analysis_engine import AnalysisEngine
        from src.visualization.visualization_engine import VisualizationEngine
    except ImportError:
        from reporting.report_integrator import ReportIntegrator
        from utils.data_loader import DataLoader
        from core.data_processor import DataProcessor
        from core.analysis_engine import AnalysisEngine
        from visualization.visualization_engine import VisualizationEngine
except ImportError as e:
    print(f"모듈 임포트 오류: {e}")
    sys.exit(1)

from .data_view import DataTableModel, DataTableView
from .visualization_view import VisualizationView
from .analysis_view import AnalysisView
from .report_view import ReportView
from .column_settings_dialog import ColumnSettingsDialog

# 파일 로드를 위한 작업자 스레드
class DataLoaderThread(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        
    def run(self):
        try:
            # 진행 상황 업데이트
            self.progress.emit(10)
            
            # 데이터 로더 생성 및 파일 로드
            loader = DataLoader()
            
            # 진행 상황 업데이트
            self.progress.emit(40)
            
            dataframe = loader.load_file(self.file_path)
            
            # 진행 상황 업데이트
            self.progress.emit(80)
            
            # 결과 전송
            self.finished.emit(dataframe)
            
        except Exception as e:
            self.error.emit(str(e))

# 파일 저장을 위한 작업자 스레드
class SaveFileThread(QThread):
    finished = pyqtSignal(bool)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, dataframe, file_path):
        super().__init__()
        self.dataframe = dataframe
        self.file_path = file_path
        
    def run(self):
        try:
            # 진행 상황 업데이트
            self.progress.emit(10)
            
            # 파일 확장자에 따라 저장 방식 결정
            if self.file_path.endswith('.csv'):
                # CSV 파일 저장
                self.progress.emit(30)
                self.dataframe.to_csv(self.file_path, index=False, encoding='utf-8-sig')
                self.progress.emit(80)
            else:
                # Excel 파일 저장
                self.progress.emit(30)
                self.dataframe.to_excel(self.file_path, index=False)
                self.progress.emit(80)
            
            # 완료 신호 전송
            self.progress.emit(100)
            self.finished.emit(True)
            
        except Exception as e:
            self.error.emit(str(e))

# 데이터 처리를 위한 작업자 스레드
class DataProcessThread(QThread):
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, dataframe, processor):
        super().__init__()
        self.dataframe = dataframe
        self.processor = processor
        
    def run(self):
        try:
            # 진행 상황 업데이트
            self.progress.emit(10)
            
            # 데이터프레임 설정
            self.processor.set_dataframe(self.dataframe)
            
            # 진행 상황 업데이트
            self.progress.emit(40)
            
            # 데이터 처리 실행
            processed_data = self.processor.process(self.dataframe)
            
            # 진행 상황 업데이트
            self.progress.emit(90)
            
            # 완료 신호 전송
            self.finished.emit(processed_data)
            
        except Exception as e:
            self.error.emit(str(e))

# 분석을 위한 작업자 스레드
class AnalysisThread(QThread):
    finished = pyqtSignal()
    error = pyqtSignal(str)
    progress = pyqtSignal(int)
    
    def __init__(self, dataframe, analysis_engine):
        super().__init__()
        self.dataframe = dataframe
        self.analysis_engine = analysis_engine
        
    def run(self):
        try:
            # 진행 상황 업데이트
            self.progress.emit(10)
            
            # 데이터프레임 설정
            self.analysis_engine.set_dataframe(self.dataframe)
            
            # 진행 상황 업데이트
            self.progress.emit(40)
            
            # 분석 실행
            self.analysis_engine.analyze(self.dataframe)
            
            # 진행 상황 업데이트
            self.progress.emit(90)
            
            # 완료 신호 전송
            self.finished.emit()
            
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    """
    엑셀 분석 시스템의 메인 윈도우
    
    사용자 인터페이스의 주요 구성 요소와 기능을 담당합니다.
    """
    
    def __init__(self):
        super().__init__()
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 통합 엔진 초기화
        reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../reports")
        templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../templates")
        
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        if not os.path.exists(templates_dir):
            os.makedirs(templates_dir)
            
        self.integrator = ReportIntegrator(
            output_dir=reports_dir,
            template_dir=templates_dir
        )
        
        # UI 설정
        self.setWindowTitle("엑셀 데이터 분석 시스템")
        self.setGeometry(100, 100, 1200, 800)
        
        # 애플리케이션 설정 로드
        self.settings = QSettings("ExcelAnalysis", "ExcelAnalysisSystem")
        self.loadSettings()
        
        # 현재 로드된 파일 경로
        self.current_file_path = None
        
        # 상태 표시줄 설정
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("준비")
        
        # 프로그레스 바 생성
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setVisible(False)  # 초기에는 숨김
        self.status_bar.addPermanentWidget(self.progress_bar)
        
        # UI 초기화
        self.initUI()
        
    def initUI(self):
        """UI 구성 요소 초기화"""
        # 중앙 위젯 생성
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 메인 레이아웃 설정
        main_layout = QVBoxLayout(central_widget)
        
        # 메뉴바 및 툴바 설정
        self.createActions()
        self.createMenus()
        self.createToolBar()
        
        # 탭 위젯 생성
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)
        
        # 데이터 뷰 탭
        self.data_view = DataTableView()
        self.tab_widget.addTab(self.data_view, "데이터")
        
        # 분석 뷰 탭
        self.analysis_view = AnalysisView()
        self.tab_widget.addTab(self.analysis_view, "분석")
        
        # 시각화 뷰 탭
        self.visualization_view = VisualizationView()
        self.tab_widget.addTab(self.visualization_view, "시각화")
        
        # 보고서 뷰 탭
        self.report_view = ReportView()
        self.tab_widget.addTab(self.report_view, "보고서")
        
        # 탭 변경 시그널 연결
        self.tab_widget.currentChanged.connect(self.onTabChanged)
        
        # 하단 상태 레이아웃
        status_layout = QHBoxLayout()
        main_layout.addLayout(status_layout)
        
        # 현재 파일 표시 레이블
        self.file_label = QLabel("파일 없음")
        status_layout.addWidget(self.file_label)
        
        # 스페이서
        status_layout.addStretch()
        
        # 데이터 정보 레이블
        self.data_info_label = QLabel("데이터 없음")
        status_layout.addWidget(self.data_info_label)
        
    def createActions(self):
        """액션 생성"""
        # 파일 메뉴 액션
        self.action_open = QAction("열기(&O)...", self)
        self.action_open.setShortcut("Ctrl+O")
        self.action_open.setStatusTip("엑셀 파일 열기")
        self.action_open.triggered.connect(self.openFile)
        
        self.action_save = QAction("저장(&S)...", self)
        self.action_save.setShortcut("Ctrl+S")
        self.action_save.setStatusTip("파일 저장")
        self.action_save.triggered.connect(self.saveFile)
        
        self.action_exit = QAction("종료(&X)", self)
        self.action_exit.setShortcut("Alt+F4")
        self.action_exit.setStatusTip("프로그램 종료")
        self.action_exit.triggered.connect(self.close)
        
        # 데이터 메뉴 액션
        self.action_process = QAction("데이터 처리(&P)...", self)
        self.action_process.setStatusTip("데이터 전처리 수행")
        self.action_process.triggered.connect(self.processData)
        
        # 분석 메뉴 액션
        self.action_analyze = QAction("분석 실행(&A)", self)
        self.action_analyze.setStatusTip("데이터 분석 실행")
        self.action_analyze.triggered.connect(self.runAnalysis)
        
        # 시각화 메뉴 액션
        self.action_visualize = QAction("시각화 생성(&V)", self)
        self.action_visualize.setStatusTip("시각화 생성")
        self.action_visualize.triggered.connect(self.createVisualizations)
        
        # 보고서 메뉴 액션
        self.action_report = QAction("리포트 생성(&G)", self)
        self.action_report.setStatusTip("분석 리포트 생성")
        self.action_report.triggered.connect(self.generateReport)
        
        # 도움말 메뉴 액션
        self.action_about = QAction("정보(&A)...", self)
        self.action_about.setStatusTip("프로그램 정보")
        self.action_about.triggered.connect(self.showAbout)
        
        # 보기 메뉴
        self.action_refresh = QAction("새로고침(&R)", self)
        self.action_refresh.setShortcut("F5")
        self.action_refresh.setStatusTip("데이터 뷰 새로고침")
        
    def createMenus(self):
        """메뉴바 설정"""
        # 파일 메뉴
        file_menu = self.menuBar().addMenu("파일(&F)")
        file_menu.addAction(self.action_open)
        file_menu.addAction(self.action_save)
        file_menu.addSeparator()
        file_menu.addAction(self.action_exit)
        
        # 편집 메뉴
        edit_menu = self.menuBar().addMenu("편집(&E)")
        edit_menu.addAction(self.action_process)
        
        # 보기 메뉴
        view_menu = self.menuBar().addMenu("보기(&V)")
        view_menu.addAction(self.action_refresh)
        
        # 설정 메뉴 추가
        settings_menu = self.menuBar().addMenu("설정(&S)")
        
        # 한글 폰트 적용 메뉴 항목 추가
        self.action_apply_korean_font = QAction("한글 폰트 적용(&K)", self)
        self.action_apply_korean_font.setStatusTip("한글 폰트를 적용하여 차트와 그래프에 한글이 올바르게 표시되도록 합니다")
        self.action_apply_korean_font.triggered.connect(self.applyKoreanFont)
        settings_menu.addAction(self.action_apply_korean_font)
        
        # 도구 메뉴
        tools_menu = self.menuBar().addMenu("도구(&T)")
        tools_menu.addAction(self.action_analyze)
        tools_menu.addAction(self.action_visualize)
        tools_menu.addAction(self.action_report)
        
        # 도움말 메뉴
        help_menu = self.menuBar().addMenu("도움말(&H)")
        help_menu.addAction(self.action_about)
        
    def createToolBar(self):
        """툴바 설정"""
        # 메인 툴바
        main_toolbar = self.addToolBar("Main")
        main_toolbar.setMovable(False)
        main_toolbar.addAction(self.action_open)
        main_toolbar.addAction(self.action_save)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.action_process)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.action_analyze)
        main_toolbar.addAction(self.action_visualize)
        main_toolbar.addAction(self.action_report)
        
        # 한글 폰트 적용 버튼 추가
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.action_apply_korean_font)
        
    def loadSettings(self):
        """설정 로드"""
        # 윈도우 크기 및 위치 복원
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        
        # 툴바 및 도킹 위젯 상태 복원
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
        
        # 최근 파일 경로
        self.last_directory = self.settings.value("lastDirectory", "")
        
    def saveSettings(self):
        """애플리케이션 설정 저장"""
        # 윈도우 위치 및 크기 저장
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        
        # 최근 파일 경로
        if self.last_directory:
            self.settings.setValue("lastDirectory", self.last_directory)
        
    def closeEvent(self, event):
        """애플리케이션 종료 시 설정 저장"""
        self.saveSettings()
        event.accept()
        
    def openFile(self):
        """파일 열기 다이얼로그"""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "엑셀 파일 열기", self.last_directory,
            "Excel 파일 (*.xlsx *.xls *.csv);;모든 파일 (*)", options=options
        )
        
        if file_path:
            self.last_directory = os.path.dirname(file_path)
            self.loadFile(file_path)
            
    def loadFile(self, file_path=None):
        """파일 로드 함수"""
        if not file_path:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "엑셀 파일 열기", "", "Excel Files (*.xlsx *.xls *.csv);;All Files (*)"
            )
            
        if file_path:
            # 상태 표시줄 업데이트 및 프로그레스 바 표시
            self.status_bar.showMessage(f"파일 로드 중: {file_path}")
            self.showProgressBar(True, 0)
            
            # 작업자 스레드 생성 및 시작
            self.loader_thread = DataLoaderThread(file_path)
            
            # 시그널 연결
            self.loader_thread.progress.connect(self.updateProgress)
            self.loader_thread.finished.connect(lambda df: self.onFileLoaded(df, file_path))
            self.loader_thread.error.connect(self.onLoadError)
            
            # 스레드 시작
            self.loader_thread.start()
    
    def onFileLoaded(self, dataframe, file_path):
        """파일 로드 완료 시 호출되는 함수"""
        try:
            # 데이터프레임 설정
            self.data_view.setData(dataframe)
            
            # 주관식 열 설정 다이얼로그 표시
            try:
                dialog = ColumnSettingsDialog(dataframe, self)
                if dialog.exec_() == QDialog.Accepted:
                    subjective_columns = dialog.get_subjective_columns()
                    
                    # 분석 엔진이 없으면 생성
                    if self.analysis_view.analysis_engine is None:
                        from src.core.analysis_engine import AnalysisEngine
                        self.analysis_view.analysis_engine = AnalysisEngine()
                        
                    # 주관식 열 설정
                    self.analysis_view.analysis_engine.set_subjective_columns(subjective_columns)
                    self.logger.info(f"주관식 열 {len(subjective_columns)}개 설정됨")
            except Exception as e:
                self.logger.error(f"주관식 열 설정 중 오류: {str(e)}")
            
            # 분석 및 시각화 뷰 초기화
            self.analysis_view.initializeWithData(dataframe)
            self.visualization_view.initializeWithData(dataframe)
            
            # 현재 파일 경로 설정
            self.current_file_path = file_path
            
            # 상태 표시줄 업데이트
            self.status_bar.showMessage(f"파일 로드 완료: {file_path}", 5000)
            
            # 창 제목 업데이트
            file_name = os.path.basename(file_path)
            self.setWindowTitle(f"엑셀 분석기 - {file_name}")
            
            # 탭 활성화
            for i in range(self.tab_widget.count()):
                self.tab_widget.setTabEnabled(i, True)
                
            # 프로그레스 바 완료 및 숨김
            self.showProgressBar(True, 100)
            self.showProgressBar(False)
                
        except Exception as e:
            QMessageBox.critical(self, "파일 로드 오류", f"파일을 처리하는 중 오류가 발생했습니다: {str(e)}")
            self.status_bar.showMessage("파일 로드 실패", 5000)
            self.showProgressBar(False)  # 프로그레스 바 숨김
            
    def onLoadError(self, error_message):
        """파일 로드 오류 시 호출되는 함수"""
        QMessageBox.critical(self, "파일 로드 오류", f"파일을 로드하는 중 오류가 발생했습니다: {error_message}")
        self.status_bar.showMessage("파일 로드 실패", 5000)
        self.showProgressBar(False)  # 프로그레스 바 숨김
    
    def saveFile(self):
        """파일 저장 함수"""
        if not self.current_file_path or not self.data_view.model or self.data_view.model.dataframe is None:
            QMessageBox.warning(self, "저장 오류", "저장할 데이터가 없습니다.")
            return
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, "파일 저장", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;All Files (*)"
        )
        
        if save_path:
            # 상태 표시줄 업데이트 및 프로그레스 바 표시
            self.status_bar.showMessage(f"파일 저장 중: {save_path}")
            self.showProgressBar(True, 0)
            
            # 작업자 스레드 생성 및 시작
            df = self.data_view.model.dataframe
            self.save_thread = SaveFileThread(df, save_path)
            
            # 시그널 연결
            self.save_thread.progress.connect(self.updateProgress)
            self.save_thread.finished.connect(lambda success: self.onSaveCompleted(success, save_path))
            self.save_thread.error.connect(self.onSaveError)
            
            # 스레드 시작
            self.save_thread.start()
    
    def onSaveCompleted(self, success, save_path):
        """파일 저장 완료 시 호출되는 함수"""
        if success:
            self.status_bar.showMessage(f"파일 저장 완료: {save_path}", 5000)
        else:
            self.status_bar.showMessage("파일 저장 실패", 5000)
            
        # 프로그레스 바 완료 및 숨김
        self.showProgressBar(True, 100)
        self.showProgressBar(False)
    
    def onSaveError(self, error_message):
        """파일 저장 오류 시 호출되는 함수"""
        QMessageBox.critical(self, "파일 저장 오류", f"파일을 저장하는 중 오류가 발생했습니다: {error_message}")
        self.status_bar.showMessage("파일 저장 실패", 5000)
        self.showProgressBar(False)  # 프로그레스 바 숨김
                
    def processData(self):
        """데이터 처리"""
        try:
            # 파일이 로드되었는지 확인
            if not self.current_file_path:
                QMessageBox.warning(self, "경고", "먼저 엑셀 파일을 로드하세요.")
                return
                
            # 원본 데이터 가져오기
            original_data = self.integrator.data_loader.df
            
            if original_data is None or original_data.empty:
                QMessageBox.warning(self, "경고", "유효한 데이터가 없습니다.")
                return
                
            # 처리 대화상자 표시 (미구현)
            # process_dialog = DataProcessDialog(self, original_data)
            # if process_dialog.exec_() != QDialog.Accepted:
            #     return
            
            # 상태 표시줄 업데이트 및 프로그레스 바 표시
            self.status_bar.showMessage("데이터 처리 중...")
            self.showProgressBar(True, 0)
                
            # 데이터 프로세서 초기화
            if not self.integrator.data_processor:
                self.integrator.data_processor = DataProcessor()
                
            # 작업자 스레드 생성 및 시작
            self.process_thread = DataProcessThread(original_data, self.integrator.data_processor)
            
            # 시그널 연결
            self.process_thread.progress.connect(self.updateProgress)
            self.process_thread.finished.connect(self.onProcessCompleted)
            self.process_thread.error.connect(self.onProcessError)
            
            # 스레드 시작
            self.process_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "처리 오류", f"데이터 처리 준비 중 오류 발생: {str(e)}")
            self.status_bar.showMessage("데이터 처리 실패")
            self.showProgressBar(False)
            
    def onProcessCompleted(self, processed_data):
        """데이터 처리 완료 시 호출되는 함수"""
        try:
            # 결과 보여주기
            self.data_view.setData(processed_data)
            
            # 분석 및 시각화 뷰 업데이트
            self.analysis_view.initializeWithData(processed_data)
            self.visualization_view.initializeWithData(processed_data)
            
            # 상태 업데이트
            self.status_bar.showMessage("데이터 처리 완료", 5000)
            
            # 프로그레스 바 완료 및 숨김
            self.showProgressBar(True, 100)
            self.showProgressBar(False)
            
        except Exception as e:
            QMessageBox.critical(self, "처리 오류", f"처리 결과 표시 중 오류 발생: {str(e)}")
            self.status_bar.showMessage("데이터 처리 결과 표시 실패", 5000)
            self.showProgressBar(False)
            
    def onProcessError(self, error_message):
        """데이터 처리 오류 시 호출되는 함수"""
        QMessageBox.critical(self, "데이터 처리 오류", f"데이터 처리 중 오류가 발생했습니다: {error_message}")
        self.status_bar.showMessage("데이터 처리 실패", 5000)
        self.showProgressBar(False)  # 프로그레스 바 숨김
    
    def updateProgress(self, value):
        """진행 상황 업데이트"""
        self.showProgressBar(True, value)
    
    def createVisualizations(self):
        """시각화 생성 함수"""
        if not self.integrator.visualization_engine:
            QMessageBox.warning(self, "시각화 오류", "시각화할 데이터가 없습니다.")
            return
            
        try:
            # 상태 표시줄 업데이트 및 프로그레스 바 표시
            self.status_bar.showMessage("시각화 생성 중...")
            self.showProgressBar(True, 0)
            
            # 분석 결과 가져오기
            analysis_results = {}
            if self.analysis_view.analysis_engine:
                analysis_results = self.analysis_view.analysis_engine.get_all_results()
            
            # 기본 시각화 수행
            df = self.data_view.model.dataframe
            viz_result = self.integrator.visualization_engine.generate_visualizations(df, analysis_results)
            
            # 주관식 텍스트 시각화 추가 (분석 결과에 텍스트 분석 결과가 있는 경우)
            if analysis_results and 'text_analysis' in analysis_results and analysis_results['text_analysis']:
                self.createTextVisualizations(analysis_results['text_analysis'])
            
            # 시각화 뷰 업데이트
            self.visualization_view.updateVisualizations()
            
            # 사용자에게 시각화 탭으로 이동 알림
            self.tab_widget.setCurrentWidget(self.visualization_view)
            
            # 프로그레스 바 완료 및 숨김
            self.showProgressBar(True, 100)
            self.showProgressBar(False)
            
            # 성공 메시지
            self.status_bar.showMessage("시각화가 완료되었습니다.", 5000)
        except Exception as e:
            QMessageBox.critical(self, "시각화 오류", f"시각화 중 오류 발생: {str(e)}")
            self.logger.error(f"시각화 중 오류: {str(e)}")
            self.showProgressBar(False)

    def createTextVisualizations(self, text_analysis):
        """주관식 텍스트 시각화 생성"""
        if not text_analysis:
            return
        
        try:
            # 시각화 생성 시도
            from src.analysis.text_analyzer import TextAnalyzer
            text_analyzer = TextAnalyzer()
            
            # 시각화 결과 저장 디렉토리
            output_dir = os.path.join(os.getcwd(), 'output', 'viz')
            os.makedirs(output_dir, exist_ok=True)
            
            # 각 주관식 열에 대한 시각화 생성
            for col_name, text_data in text_analysis.items():
                visualizations = text_analyzer.create_text_visualizations(
                    text_data, col_name, output_dir
                )
                
                # 시각화 뷰에 추가
                for viz in visualizations:
                    # 시각화 이미지 경로와 메타데이터 설정
                    self.visualization_view.addVisualization(
                        viz['path'], 
                        viz['title'], 
                        viz['description'], 
                        '주관식'  # 범주를 '주관식'으로 설정
                    )
        except Exception as e:
            self.logger.error(f"주관식 시각화 생성 중 오류: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            
    def generateReport(self):
        """보고서 생성 함수"""
        if not self.current_file_path or not self.data_view.model or self.data_view.model.dataframe is None:
            QMessageBox.warning(self, "보고서 생성 오류", "보고서를 생성할 데이터가 없습니다.")
            return
        
        # 보고서 대화상자 생성
        dialog = QDialog(self)
        dialog.setWindowTitle("보고서 생성")
        dialog.resize(500, 400)
        
        # 대화상자 레이아웃
        layout = QVBoxLayout(dialog)
        
        # 보고서 제목
        form_layout = QFormLayout()
        layout.addLayout(form_layout)
        
        title_edit = QLineEdit("데이터 분석 보고서")
        form_layout.addRow("보고서 제목:", title_edit)
        
        from datetime import datetime
        subtitle_edit = QLineEdit(f"생성일: {datetime.now().strftime('%Y-%m-%d')}")
        form_layout.addRow("부제목:", subtitle_edit)
        
        # 포함할 내용 체크박스
        group_box = QGroupBox("포함할 내용")
        group_layout = QVBoxLayout(group_box)
        
        include_summary = QCheckBox("데이터 요약")
        include_summary.setChecked(True)
        group_layout.addWidget(include_summary)
        
        include_stats = QCheckBox("기술 통계량")
        include_stats.setChecked(True)
        group_layout.addWidget(include_stats)
        
        include_viz = QCheckBox("데이터 시각화")
        include_viz.setChecked(True)
        group_layout.addWidget(include_viz)
        
        layout.addWidget(group_box)
        
        # 형식 선택
        format_group = QGroupBox("보고서 형식")
        format_layout = QVBoxLayout(format_group)
        
        pdf_radio = QRadioButton("PDF")
        pdf_radio.setChecked(True)
        format_layout.addWidget(pdf_radio)
        
        html_radio = QRadioButton("HTML")
        format_layout.addWidget(html_radio)
        
        layout.addWidget(format_group)
        
        # 버튼
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Ok).setText("생성")
        button_box.button(QDialogButtonBox.Cancel).setText("취소")
        layout.addWidget(button_box)
        
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # 대화상자 실행
        if dialog.exec() == QDialog.Accepted:
            # 저장 경로 선택
            file_format = "PDF Files (*.pdf)" if pdf_radio.isChecked() else "HTML Files (*.html)"
            default_ext = ".pdf" if pdf_radio.isChecked() else ".html"
            
            save_path, _ = QFileDialog.getSaveFileName(
                self, "보고서 저장", f"보고서_{datetime.now().strftime('%Y%m%d')}{default_ext}", file_format
            )
            
            if save_path:
                try:
                    # 상태 표시줄 업데이트
                    self.status_bar.showMessage(f"보고서 생성 중: {save_path}")
                    QApplication.processEvents()
                    
                    # 보고서 생성
                    from src.utils.report_generator import ReportGenerator
                    
                    report_gen = ReportGenerator()
                    
                    # 보고서 설정
                    report_gen.set_title(title_edit.text())
                    report_gen.set_subtitle(subtitle_edit.text())
                    
                    # 데이터 설정
                    report_gen.set_dataframe(self.data_view.model.dataframe)
                    
                    # 분석 결과 설정
                    if include_stats.isChecked() and self.analysis_view.analysis_engine:
                        report_gen.set_statistics(self.analysis_view.analysis_engine.get_results())
                    
                    # 시각화 결과 설정
                    if include_viz.isChecked() and self.visualization_view.visualization_engine:
                        report_gen.set_visualizations(self.visualization_view.visualization_engine.get_plots())
                    
                    # 보고서 생성 및 저장
                    if pdf_radio.isChecked():
                        report_gen.generate_pdf(save_path)
                    else:
                        report_gen.generate_html(save_path)
                    
                    # 상태 표시줄 업데이트
                    self.status_bar.showMessage(f"보고서 생성 완료: {save_path}", 5000)
                    
                    # 보고서 열기
                    import webbrowser
                    webbrowser.open(save_path)
                    
                except Exception as e:
                    QMessageBox.critical(self, "보고서 생성 오류", f"보고서 생성 중 오류가 발생했습니다: {str(e)}")
                    self.status_bar.showMessage("보고서 생성 실패", 5000)
            
    def onTabChanged(self, index):
        """탭 변경 이벤트 핸들러"""
        # 각 탭 뷰 업데이트
        tab_widget = self.tab_widget.widget(index)
        
        if tab_widget == self.data_view:
            # 데이터 뷰 탭으로 변경
            pass
        elif tab_widget == self.analysis_view:
            # 분석 뷰 탭으로 변경
            self.analysis_view.updateResults()
        elif tab_widget == self.visualization_view:
            # 시각화 뷰 탭으로 변경
            self.visualization_view.updateVisualizations()
        elif tab_widget == self.report_view:
            # 보고서 뷰 탭으로 변경
            pass
            
    def showAbout(self):
        """프로그램 정보 표시"""
        about_msg = """<b>엑셀 데이터 분석 시스템</b><br>
        버전: 1.0<br><br>
        엑셀 파일 데이터를 로드하고 분석, 시각화하는 도구입니다.<br>
        Copyright © 2023"""
        
        QMessageBox.about(self, "프로그램 정보", about_msg)

    def set_components(self, data_loader=None, data_processor=None, 
                      analysis_engine=None, visualization_engine=None, 
                      report_engine=None, report_integrator=None):
        """시스템 구성 요소 설정"""
        # 통합기 구성 요소 설정
        if report_integrator:
            self.integrator = report_integrator
        
        # 각 뷰에 해당 엔진 설정
        if data_loader:
            self.data_loader = data_loader
            
        if data_processor:
            self.data_processor = data_processor
            
        if analysis_engine:
            self.analysis_engine = analysis_engine
            self.analysis_view.setAnalysisEngine(analysis_engine)
            
        if visualization_engine:
            self.visualization_engine = visualization_engine
            self.visualization_view.setVisualizationEngine(visualization_engine)
            
        if report_engine:
            self.report_engine = report_engine
            
        # 보고서 뷰에 모든 구성 요소 설정
        self.report_view.setComponents(
            report_engine=report_engine,
            report_integrator=report_integrator,
            data_loader=data_loader,
            processor=data_processor,
            analysis_engine=analysis_engine,
            visualization_engine=visualization_engine
        )
        
        # 상태 바 업데이트
        self.status_bar.showMessage("모든 구성 요소 초기화 완료")

    def showProgressBar(self, visible=True, value=0):
        """프로그레스 바 표시 상태 설정"""
        self.progress_bar.setVisible(visible)
        self.progress_bar.setValue(value)
        QApplication.processEvents()  # UI 업데이트

    def runAnalysis(self):
        """분석 실행 함수"""
        if not self.current_file_path:
            QMessageBox.warning(self, "분석 오류", "분석할 데이터가 로드되지 않았습니다.")
            return

        if not self.data_view.model or self.data_view.model.dataframe is None:
            QMessageBox.warning(self, "분석 오류", "분석할 데이터가 없습니다.")
            return
        
        # 상태 표시줄 업데이트 및 프로그레스 바 표시
        self.status_bar.showMessage("데이터 분석 중...")
        self.showProgressBar(True, 0)
        
        # 분석할 데이터프레임 가져오기
        df = self.data_view.model.dataframe
        
        # 작업자 스레드 생성 및 시작
        if self.analysis_view.analysis_engine is None:
            from src.core.analysis_engine import AnalysisEngine
            self.analysis_view.analysis_engine = AnalysisEngine()
            
        self.analysis_thread = AnalysisThread(df, self.analysis_view.analysis_engine)
        
        # 시그널 연결
        self.analysis_thread.progress.connect(self.updateProgress)
        self.analysis_thread.finished.connect(self.onAnalysisFinished)
        self.analysis_thread.error.connect(self.onAnalysisError)
        
        # 스레드 시작
        self.analysis_thread.start()
    
    def onAnalysisFinished(self):
        """분석 완료 시 호출되는 함수"""
        # 결과 업데이트
        self.analysis_view.updateResults()
        
        # 상태 표시줄 업데이트
        self.status_bar.showMessage("데이터 분석 완료", 5000)
        
        # 프로그레스 바 완료 및 숨김
        self.showProgressBar(True, 100)
        self.showProgressBar(False)
        
    def onAnalysisError(self, error_message):
        """분석 오류 시 호출되는 함수"""
        QMessageBox.critical(self, "분석 오류", f"분석 중 오류가 발생했습니다: {error_message}")
        self.status_bar.showMessage("데이터 분석 실패", 5000)
        self.showProgressBar(False)  # 프로그레스 바 숨김

    def applyKoreanFont(self):
        """한글 폰트 적용 메서드"""
        try:
            from src.visualization.visualization_engine import configure_korean_font, reset_font_cache
            
            # 진행 상태 표시
            self.status_bar.showMessage("한글 폰트 설정 중...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(10)
            
            # 폰트 캐시 초기화
            self.progress_bar.setValue(30)
            reset_font_cache()
            
            # 한글 폰트 설정
            self.progress_bar.setValue(60)
            result = configure_korean_font()
            
            self.progress_bar.setValue(100)
            
            # 결과 메시지 표시
            if result:
                QMessageBox.information(self, "폰트 설정 완료", 
                    "한글 폰트가 성공적으로 적용되었습니다.\n시각화 생성 시 한글이 올바르게 표시됩니다.")
                self.status_bar.showMessage("한글 폰트 설정 완료", 5000)
            else:
                QMessageBox.warning(self, "폰트 설정 실패", 
                    "한글 폰트를 적용하는 데 문제가 발생했습니다.\n한글 폰트가 시스템에 설치되어 있는지 확인하세요.")
                self.status_bar.showMessage("한글 폰트 설정 실패", 5000)
            
            # 프로그레스 바 숨김
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_bar.showMessage("한글 폰트 설정 오류", 5000)
            QMessageBox.critical(self, "오류", f"한글 폰트 설정 중 오류가 발생했습니다:\n{str(e)}")
            self.logger.error(f"한글 폰트 설정 오류: {str(e)}")

# 추가: 애플리케이션 실행 함수
def run():
    """
    GUI 애플리케이션 시작 함수
    """
    import sys
    from PyQt5.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()

if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 애플리케이션 실행
    run() 