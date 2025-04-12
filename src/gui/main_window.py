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
        from src.data.data_loader import DataLoader
        from src.core.data_processor import DataProcessor
        from src.core.analysis_engine import AnalysisEngine
        from src.visualization.visualization_engine import VisualizationEngine
    except ImportError:
        from reporting.report_integrator import ReportIntegrator
        from data.data_loader import DataLoader
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
        self.open_action = QAction(QIcon.fromTheme("document-open"), "열기", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self.openFile)
        
        self.save_action = QAction(QIcon.fromTheme("document-save"), "저장", self)
        self.save_action.setShortcut("Ctrl+S")
        self.save_action.triggered.connect(self.saveFile)
        
        self.exit_action = QAction(QIcon.fromTheme("application-exit"), "종료", self)
        self.exit_action.setShortcut("Ctrl+Q")
        self.exit_action.triggered.connect(self.close)
        
        # 데이터 메뉴 액션
        self.process_data_action = QAction("데이터 처리", self)
        self.process_data_action.triggered.connect(self.processData)
        
        # 분석 메뉴 액션
        self.analyze_action = QAction("분석 실행", self)
        self.analyze_action.triggered.connect(self.runAnalysis)
        
        # 시각화 메뉴 액션
        self.visualize_action = QAction("시각화 생성", self)
        self.visualize_action.triggered.connect(self.createVisualizations)
        
        # 보고서 메뉴 액션
        self.generate_report_action = QAction("보고서 생성", self)
        self.generate_report_action.triggered.connect(self.generateReport)
        
        # 도움말 메뉴 액션
        self.about_action = QAction("정보", self)
        self.about_action.triggered.connect(self.showAbout)
        
    def createMenus(self):
        """메뉴 생성"""
        # 메뉴바 생성
        menubar = self.menuBar()
        
        # 파일 메뉴
        file_menu = menubar.addMenu("파일")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)
        
        # 데이터 메뉴
        data_menu = menubar.addMenu("데이터")
        data_menu.addAction(self.process_data_action)
        
        # 분석 메뉴
        analysis_menu = menubar.addMenu("분석")
        analysis_menu.addAction(self.analyze_action)
        
        # 시각화 메뉴
        visualization_menu = menubar.addMenu("시각화")
        visualization_menu.addAction(self.visualize_action)
        
        # 보고서 메뉴
        report_menu = menubar.addMenu("보고서")
        report_menu.addAction(self.generate_report_action)
        
        # 도움말 메뉴
        help_menu = menubar.addMenu("도움말")
        help_menu.addAction(self.about_action)
        
    def createToolBar(self):
        """툴바 생성"""
        # 메인 툴바
        self.toolbar = QToolBar("메인 툴바")
        self.toolbar.setObjectName("mainToolBar")  # 객체 이름 설정
        self.toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(self.toolbar)
        
        # 툴바에 액션 추가
        self.toolbar.addAction(self.open_action)
        self.toolbar.addAction(self.save_action)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.analyze_action)
        self.toolbar.addAction(self.visualize_action)
        self.toolbar.addAction(self.generate_report_action)
        
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
            try:
                # 상태 표시줄 업데이트
                self.status_bar.showMessage(f"파일 저장 중: {save_path}")
                QApplication.processEvents()
                
                # 파일 확장자에 따라 저장 방식 결정
                df = self.data_view.model.dataframe
                
                if save_path.endswith('.csv'):
                    df.to_csv(save_path, index=False, encoding='utf-8-sig')
                else:
                    df.to_excel(save_path, index=False)
                
                # 상태 표시줄 업데이트
                self.status_bar.showMessage(f"파일 저장 완료: {save_path}", 5000)
                
            except Exception as e:
                QMessageBox.critical(self, "파일 저장 오류", f"파일을 저장하는 중 오류가 발생했습니다: {str(e)}")
                self.status_bar.showMessage("파일 저장 실패", 5000)
                
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
                
            # 데이터 프로세서 초기화
            if not self.integrator.data_processor:
                self.integrator.data_processor = DataProcessor()
                
            # 데이터 설정
            self.integrator.data_processor.set_dataframe(original_data)
            
            # 기본 처리 수행
            processed_data = self.integrator.data_processor.process(original_data)
            
            # 결과 보여주기
            self.data_view.setData(processed_data)
            
            # 분석 및 시각화 뷰 업데이트
            self.analysis_view.initializeWithData(processed_data)
            self.visualization_view.initializeWithData(processed_data)
            
            # 상태 업데이트
            self.status_bar.showMessage("데이터 처리 완료")
            
        except Exception as e:
            QMessageBox.critical(self, "처리 오류", f"데이터 처리 중 오류 발생: {str(e)}")
            self.status_bar.showMessage("데이터 처리 실패")
            
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