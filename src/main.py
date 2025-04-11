#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Excel Analyzer 애플리케이션의 메인 진입점
"""

import sys
import os
import logging
from PyQt5.QtWidgets import QApplication

# 프로젝트 루트 디렉토리를 Python 경로에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/application.log"),
        logging.StreamHandler()
    ]
)

# 필요한 디렉토리 생성
os.makedirs("logs", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# GUI 모드 실행
def run_gui():
    """GUI 모드로 애플리케이션 실행"""
    print("GUI 환경을 초기화하는 중...")
    
    # GUI 모듈은 경로가 설정된 후에 임포트
    from src.gui.main_window import MainWindow
    
    # QApplication 인스턴스 생성
    app = QApplication(sys.argv)
    
    # 스타일시트 적용 (일반 스타일로 변경)
    app.setStyle('Fusion')  # 'Fusion', 'Windows', 'WindowsVista' 등 사용 가능
    
    # 메인 윈도우 생성 및 표시
    main_window = MainWindow()
    main_window.show()
    
    # 애플리케이션 실행
    sys.exit(app.exec_())

# CLI 모드 실행
def run_cli(args):
    """CLI 모드로 애플리케이션 실행"""
    print("CLI 모드로 실행 중...")
    
    try:
        from src.cli.cli_controller import CLIController
        
        controller = CLIController()
        return controller.process_args(args)
    except ImportError as e:
        print(f"CLI 컨트롤러 로드 실패: {str(e)}")
        print("기본 CLI 모드로 실행합니다...")
        
        # 대체 방안: 직접 cli.py 모듈 사용
        from src.cli import main as cli_main
        return cli_main(args)

if __name__ == "__main__":
    print("엑셀 분석기를 시작합니다...")
    
    if len(sys.argv) > 1:
        # CLI 모드
        run_cli(sys.argv[1:])
    else:
        # GUI 모드
        run_gui() 