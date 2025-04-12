import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import List, Dict, Tuple, Union, Optional, Any
from datetime import datetime
from pathlib import Path

# 모듈 임포트
from src.visualization.font_manager import setup_korean_fonts
from src.visualization.chart_generators import ChartGenerators
from src.visualization.data_validators import (
    validate_dataframe, validate_column, validate_column_data,
    validate_numeric_column, validate_categorical_column
)

# 로깅 설정
logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    데이터 시각화를 위한 엔진 클래스
    """
    
    def __init__(self, output_dir: str = 'output/viz'):
        """
        VisualizationEngine 초기화
        
        Args:
            output_dir (str): 시각화 결과물 저장 디렉토리
        """
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 설정
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터프레임 초기화
        self.df = None
        self.df_info = {
            'rows': 0,
            'columns': 0,
            'numeric_columns': [],
            'categorical_columns': []
        }
        
        # 폰트 설정
        self._setup_fonts()
        
        # 시각화 스타일 설정
        self._setup_visualization_style()
        
        # 기본 그림 크기 및 해상도 설정
        self.figure_size = (10, 6)
        self.dpi = 100
        
        # 차트 생성기 초기화
        self.chart_generator = None
    
    def _setup_fonts(self):
        """
        시각화에 사용할 한글 폰트 설정
        """
        try:
            setup_korean_fonts()
            self.logger.info("한글 폰트 설정 완료")
        except Exception as e:
            self.logger.warning(f"한글 폰트 설정 중 오류 발생: {str(e)}")
    
    def _setup_visualization_style(self):
        """
        시각화 스타일 설정
        """
        try:
            # Seaborn 스타일 설정
            sns.set_style('whitegrid')
            
            # Matplotlib 기본 설정
            plt.rcParams['figure.figsize'] = (10, 6)
            plt.rcParams['figure.dpi'] = 100
            
            self.logger.info("시각화 스타일 설정 완료")
        except Exception as e:
            self.logger.warning(f"시각화 스타일 설정 중 오류 발생: {str(e)}")
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        시각화할 데이터프레임 설정
        
        Args:
            df (pd.DataFrame): 시각화 대상 데이터프레임
        """
        self.df = df
        
        # 차트 생성기 초기화 또는 업데이트
        self.chart_generator = ChartGenerators(df, self.output_dir, 
                                             self.figure_size, self.dpi)
        
        self.logger.info(f"데이터프레임 설정 완료 (행: {df.shape[0]}, 열: {df.shape[1]})")
        
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """현재 데이터프레임 반환"""
        return self.df
        
    def set_theme(self, theme: str) -> None:
        """
        시각화 테마 설정
        
        Args:
            theme (str): 테마 이름 ('default', 'dark', 'light', 'colorblind', 'pastel', 'deep', 등)
        """
        available_themes = ['default', 'dark_background', 'whitegrid', 'darkgrid', 'white', 'dark', 
                          'colorblind', 'pastel', 'deep', 'muted', 'ticks']
                          
        if theme not in available_themes:
            self.logger.warning(f"지원되지 않는 테마: {theme}. 기본 테마를 사용합니다.")
            theme = 'default'
            
        if theme == 'default':
            plt.style.use('default')
        elif theme == 'dark_background':
            plt.style.use('dark_background')
        else:
            # seaborn 테마 적용
            if theme in ['whitegrid', 'darkgrid', 'white', 'dark', 'ticks']:
                sns.set_style(theme)
            else:
                sns.set_palette(theme)
                
        self.theme = theme
        self.logger.info(f"테마 설정 완료: {theme}")
        
    def set_figure_size(self, width: int, height: int, dpi: int = 100) -> None:
        """
        그림 크기 및 해상도 설정
        
        Args:
            width (int): 그림 너비
            height (int): 그림 높이
            dpi (int): 해상도 (dots per inch)
        """
        self.figure_size = (width, height)
        self.dpi = dpi
        
        # 차트 생성기가 초기화된 경우 설정 업데이트
        if self.chart_generator:
            self.chart_generator.figure_size = self.figure_size
            self.chart_generator.dpi = self.dpi
            
        self.logger.info(f"그림 크기 설정 완료: {width}x{height}, DPI: {dpi}")
        
    # 차트 생성 메소드들을 ChartGenerators로 위임
    def plot_histogram(self, column: str, bins: int = 20, 
                      kde: bool = True, title: Optional[str] = None, 
                      filename: Optional[str] = None) -> str:
        """
        히스토그램을 생성합니다.
        
        Args:
            column (str): 데이터 열 이름
            bins (int): 구간 수
            kde (bool): 밀도 커널 표시 여부
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.chart_generator is None:
            self.logger.error("차트 생성기가 초기화되지 않았습니다. set_dataframe()을 먼저 호출하세요.")
            return ""
            
        return self.chart_generator.plot_histogram(column, bins, kde, title, filename)
        
    def plot_bar(self, column: str, horizontal: bool = True, 
                top_n: int = 10, show_others: bool = True,
                title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        막대 그래프를 생성합니다.
        
        Args:
            column (str): 데이터 열 이름
            horizontal (bool): 수평 막대 그래프 여부
            top_n (int): 표시할 상위 항목 수
            show_others (bool): 상위 N개 이외 범주를 '기타'로 표시할지 여부
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.chart_generator is None:
            self.logger.error("차트 생성기가 초기화되지 않았습니다. set_dataframe()을 먼저 호출하세요.")
            return ""
            
        return self.chart_generator.plot_bar(column, horizontal, top_n, show_others, title, filename)
        
    def plot_scatter(self, x_column: str, y_column: str, hue_column: Optional[str] = None,
                     title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        두 열 간의 산점도를 생성합니다.
        
        Args:
            x_column (str): x축 데이터 열 이름
            y_column (str): y축 데이터 열 이름
            hue_column (str, optional): 색상 구분을 위한 열 이름
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.chart_generator is None:
            self.logger.error("차트 생성기가 초기화되지 않았습니다. set_dataframe()을 먼저 호출하세요.")
            return ""
            
        return self.chart_generator.plot_scatter(x_column, y_column, hue_column, title, filename)
        
    def create_dashboard(self, output_dir: str = "dashboard", 
                         max_categorical_cols: int = 10, max_numeric_cols: int = 10,
                         include_corr: bool = True) -> Dict[str, List[str]]:
        """
        주어진 데이터프레임에 대한 간단한 대시보드 생성
        
        Args:
            output_dir (str): 대시보드 출력 디렉토리 (사용되지 않음, 기존 호환성 유지를 위해 남겨둠)
            max_categorical_cols (int): 처리할 최대 범주형 변수 수
            max_numeric_cols (int): 처리할 최대 수치형 변수 수
            include_corr (bool): 상관관계 히트맵 포함 여부
            
        Returns:
            Dict[str, List[str]]: 생성된 시각화 파일 경로 목록
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return {}
            
        try:
            # output_dir 파라미터는 무시하고 기존 output_dir을 그대로 사용
            # 결과를 원래 output_dir에 직접 저장 (별도 하위 폴더 없이)
            self.logger.info(f"대시보드 이미지 저장 경로: {self.output_dir}")
            
            result_files = {
                'histograms': [],
                'bar_charts': [],
                'correlation': [],
                'boxplots': [],
                'pair_plots': [],
                'pie_charts': []
            }
            
            # 데이터 타입에 따라 변수 분류
            numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = self.df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
            
            # 최대 열 수 제한
            numeric_cols = numeric_cols[:max_numeric_cols]
            categorical_cols = categorical_cols[:max_categorical_cols]
            
            # 수치형 변수에 대한 히스토그램 생성
            for col in numeric_cols:
                try:
                    hist_file = self.plot_histogram(col, filename=f"histogram_{col}.png")
                    result_files['histograms'].append(hist_file)
                except Exception as e:
                    self.logger.error(f"히스토그램 생성 중 오류 발생 ({col}): {str(e)}")
            
            # 범주형 변수에 대한 막대 그래프 생성
            for col in categorical_cols:
                try:
                    # 막대 그래프 생성 (세로 막대 그래프 사용)
                    bar_file = self.plot_bar(col, horizontal=False, filename=f"bar_{col}.png")
                    result_files['bar_charts'].append(bar_file)
                except Exception as e:
                    self.logger.error(f"막대 그래프 생성 중 오류 발생 ({col}): {str(e)}")
            
            return result_files
        except Exception as e:
            self.logger.error(f"대시보드 생성 중 오류 발생: {str(e)}")
            return {}
    
    def export_all_visualizations(self, prefix: str = "analysis", format: str = "png") -> str:
        """
        모든 주요 시각화 유형을 생성하고 내보내기
        
        Args:
            prefix (str): 파일명 접두사
            format (str): 저장 형식 ('png', 'pdf', 'svg', 'jpg')
            
        Returns:
            str: 저장된 파일 경로 목록을 담은 문자열
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""

        # 출력 디렉토리가 존재하는지 확인하고 없으면 생성
        output_dir = str(self.output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 테마 설정
        self.set_theme('whitegrid')
        
        # 그림 크기 설정
        self.set_figure_size(12, 8, 100)
        
        # 기본 테스트 시각화 생성 (최소 1개의 파일 보장)
        test_file = ""
        try:
            if self.df is not None and len(self.df.columns) > 0:
                numeric_cols = self.df.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    test_file = self.plot_histogram(numeric_cols[0], 
                                    filename=f"{prefix}_test_histogram.{format}")
                    self.logger.info(f"테스트 시각화 생성 완료: {test_file}")
        except Exception as e:
            self.logger.error(f"테스트 시각화 생성 중 오류: {str(e)}")
        
        # 대시보드 생성
        result_files = {}
        try:
            # 대시보드 생성 (기본 viz 폴더에 직접 저장)
            result_files = self.create_dashboard(
                include_corr=True
            )
            
            self.logger.info(f"대시보드 생성 결과: {len(result_files)} 카테고리의 시각화 생성됨")
        except Exception as e:
            self.logger.error(f"대시보드 생성 중 오류: {str(e)}")
            
            # 테스트 파일만 결과로 반환
            if test_file:
                result_files = {"test_visualization": [test_file]}
                
        return str(result_files)

    def get_plots(self) -> List[str]:
        """
        생성된 모든 플롯 파일 경로 목록 반환
        
        Returns:
            List[str]: 생성된 플롯 파일 경로 목록
        """
        if not os.path.exists(self.output_dir):
            self.logger.warning(f"출력 디렉토리 '{self.output_dir}'가 존재하지 않습니다.")
            return []
        
        plot_files = [os.path.join(self.output_dir, f) for f in os.listdir(self.output_dir) 
                     if f.endswith(('.png', '.jpg', '.jpeg', '.svg', '.pdf'))]
                     
        return plot_files 