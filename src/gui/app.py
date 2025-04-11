import sys
import os
import logging
from typing import Optional

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QFileDialog, QSplashScreen
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap

# 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 모듈 임포트
from src.gui.main_window import MainWindow

# 모듈이 존재하는지 확인 후 임포트
try:
    from src.core.data_loader import ExcelLoader
except ImportError:
    try:
        from core.data_loader import ExcelLoader
    except ImportError:
        ExcelLoader = None
        print("ExcelLoader를 찾을 수 없습니다.")

try:
    from src.core.data_processor import DataProcessor
except ImportError:
    try:
        from core.data_processor import DataProcessor
    except ImportError:
        DataProcessor = None
        print("DataProcessor를 찾을 수 없습니다.")

try:
    from src.core.analysis_engine import AnalysisEngine
except ImportError:
    try:
        from core.analysis_engine import AnalysisEngine
    except ImportError:
        AnalysisEngine = None
        print("AnalysisEngine을 찾을 수 없습니다.")

try:
    from src.visualization.visualization_engine import VisualizationEngine
except ImportError:
    try:
        from visualization.visualization_engine import VisualizationEngine
    except ImportError:
        VisualizationEngine = None
        print("VisualizationEngine을 찾을 수 없습니다.")

try:
    from src.reporting.report_engine import ReportEngine
except ImportError:
    try:
        from reporting.report_engine import ReportEngine
    except ImportError:
        ReportEngine = None
        print("ReportEngine을 찾을 수 없습니다.")

try:
    from src.reporting.report_integrator import ReportIntegrator
except ImportError:
    try:
        from reporting.report_integrator import ReportIntegrator
    except ImportError:
        ReportIntegrator = None
        print("ReportIntegrator를 찾을 수 없습니다.")

def setup_logging():
    """로깅 설정"""
    # 로그 디렉토리 생성
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # 로깅 설정
    log_file = os.path.join(log_dir, "application.log")
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 파일 핸들러
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # 핸들러 추가
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "data",           # 데이터 파일
        "data/processed",  # 처리된 데이터
        "output",         # 출력 결과
        "output/viz",     # 시각화 결과
        "reports",        # 보고서
        "logs"            # 로그
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)

def create_dummy_classes():
    """모듈이 없는 경우 더미 클래스 생성"""
    global ExcelLoader, DataProcessor, AnalysisEngine, VisualizationEngine, ReportEngine, ReportIntegrator
    
    class DummyExcelLoader:
        def __init__(self):
            print("더미 ExcelLoader 생성됨")
    
    class DummyDataProcessor:
        def __init__(self):
            print("더미 DataProcessor 생성됨")
    
    class DummyAnalysisEngine:
        def __init__(self):
            print("더미 AnalysisEngine 생성됨")
        
        def get_all_results(self):
            return {}
    
    class DummyVisualizationEngine:
        def __init__(self, output_dir=None):
            self.output_dir = output_dir
            print(f"더미 VisualizationEngine 생성됨 (output_dir: {output_dir})")
    
    class DummyReportEngine:
        def __init__(self):
            print("더미 ReportEngine 생성됨")
    
    class DummyReportIntegrator:
        def __init__(self):
            print("더미 ReportIntegrator 생성됨")
    
    if ExcelLoader is None:
        ExcelLoader = DummyExcelLoader
    
    if DataProcessor is None:
        DataProcessor = DummyDataProcessor
    
    if AnalysisEngine is None:
        AnalysisEngine = DummyAnalysisEngine
    
    if VisualizationEngine is None:
        VisualizationEngine = DummyVisualizationEngine
    
    if ReportEngine is None:
        ReportEngine = DummyReportEngine
    
    if ReportIntegrator is None:
        ReportIntegrator = DummyReportIntegrator

def main():
    """메인 애플리케이션 진입점"""
    # 애플리케이션 초기화
    app = QApplication(sys.argv)
    app.setApplicationName("엑셀 데이터 분석 시스템")
    app.setStyle("Fusion")  # 모든 플랫폼에서 일관된 모양을 위해
    
    # 스플래시 화면 표시 (옵션)
    # splash_pixmap = QPixmap("resources/splash.png")
    # splash = QSplashScreen(splash_pixmap)
    # splash.show()
    # app.processEvents()
    
    # 로깅 설정
    logger = setup_logging()
    logger.info("애플리케이션 시작")
    
    # 필요한 디렉토리 생성
    create_directories()
    
    # 더미 클래스 생성
    create_dummy_classes()
    
    # 시스템 구성 요소 초기화
    try:
        # 데이터 로더
        data_loader = ExcelLoader()
        logger.info("데이터 로더 초기화 완료")
        
        # 데이터 프로세서
        data_processor = DataProcessor()
        logger.info("데이터 프로세서 초기화 완료")
        
        # 분석 엔진
        analysis_engine = AnalysisEngine()
        logger.info("분석 엔진 초기화 완료")
        
        # 시각화 엔진
        visualization_engine = VisualizationEngine(output_dir="output/viz")
        logger.info("시각화 엔진 초기화 완료")
        
        # 보고서 엔진
        report_engine = ReportEngine()
        logger.info("보고서 엔진 초기화 완료")
        
        # 보고서 통합기
        report_integrator = ReportIntegrator()
        logger.info("보고서 통합기 초기화 완료")
        
        # 메인 윈도우 생성
        main_window = MainWindow()
        main_window.set_components(
            data_loader=data_loader,
            data_processor=data_processor,
            analysis_engine=analysis_engine,
            visualization_engine=visualization_engine,
            report_engine=report_engine,
            report_integrator=report_integrator
        )
        logger.info("메인 윈도우 초기화 완료")
        
        # 스플래시 종료 타이머 (옵션)
        # QTimer.singleShot(1500, splash.close)
        
        # 메인 윈도우 표시
        main_window.show()
        logger.info("메인 윈도우 표시")
        
        # 애플리케이션 실행
        sys.exit(app.exec_())
        
    except Exception as e:
        # 초기화 중 오류 발생
        logger.error(f"초기화 중 오류 발생: {str(e)}", exc_info=True)
        QMessageBox.critical(None, "오류", f"애플리케이션 초기화 중 오류가 발생했습니다:\n{str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 