#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import pandas as pd
import logging
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

# 로그 설정
logging.basicConfig(level=logging.DEBUG,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 로그 디렉토리 확인
os.makedirs("logs", exist_ok=True)

# 소스 폴더를 모듈 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.gui.analysis_view import AnalysisView
    from src.core.analysis_engine import AnalysisEngine
except ImportError as e:
    logger.error(f"모듈 임포트 오류: {e}")
    sys.exit(1)

class TestWindow(QMainWindow):
    """테스트용 메인 윈도우"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("분석 뷰 테스트")
        self.resize(1000, 800)
        
        # 분석 뷰 생성
        self.analysis_view = AnalysisView()
        self.setCentralWidget(self.analysis_view)
        
        # 샘플 데이터 생성
        self.create_sample_data()
        
        # 분석 엔진 설정
        self.analysis_engine = AnalysisEngine()
        self.analysis_engine.set_dataframe(self.sample_data)
        
        # 분석 뷰에 엔진 설정
        self.analysis_view.setAnalysisEngine(self.analysis_engine)
        self.analysis_view.updateResults()
    
    def create_sample_data(self):
        """테스트용 샘플 데이터 생성"""
        import numpy as np
        
        # 데이터 생성
        np.random.seed(42)
        n = 100
        
        # 데이터프레임 생성
        self.sample_data = pd.DataFrame({
            '나이': np.random.randint(20, 65, n),
            '소득': np.random.normal(5000, 1500, n),
            '교육연수': np.random.randint(9, 22, n),
            '경력연수': np.random.randint(0, 40, n),
            '만족도': np.random.randint(1, 11, n)
        })
        
        # 로그 출력
        logger.info(f"샘플 데이터 생성됨: {self.sample_data.shape}")
        logger.info(f"데이터 열: {list(self.sample_data.columns)}")

def main():
    """테스트 실행 함수"""
    app = QApplication(sys.argv)
    
    # 메인 창 생성 및 표시
    window = TestWindow()
    window.show()
    
    # 자동 분석 실행
    window.analysis_view.runAnalysis()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main() 