import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import List, Dict, Tuple, Union, Optional, Any
from datetime import datetime
from src.visualization.data_validators import (
    validate_dataframe, validate_column_data, 
    validate_numeric_column, validate_categorical_column
)

# 로깅 설정
logger = logging.getLogger(__name__)

class ChartGenerators:
    """
    다양한 차트를 생성하는 클래스
    
    VisualizationEngine에서 사용할 차트 생성 메서드들을 모아둔 클래스입니다.
    """
    
    def __init__(self, df: pd.DataFrame, output_dir: str, 
                 figure_size: Tuple[int, int] = (10, 6), dpi: int = 100):
        """
        초기화
        
        Args:
            df (pd.DataFrame): 시각화할 데이터프레임
            output_dir (str): 출력 디렉토리
            figure_size (Tuple[int, int]): 기본 그림 크기
            dpi (int): 해상도(DPI)
        """
        self.df = df
        self.output_dir = output_dir
        self.figure_size = figure_size
        self.dpi = dpi
        self.logger = logger
    
    def _save_figure(self, filename: str) -> str:
        """
        현재 그림을 파일로 저장
        
        Args:
            filename (str): 저장할 파일 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 출력 디렉토리가 존재하는지 확인하고 없으면 생성
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 절대 경로로 변환
            file_path = os.path.join(self.output_dir, filename)
            file_path = os.path.abspath(file_path)
            
            plt.tight_layout()
            plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"그림 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"그림 저장 중 오류 발생: {str(e)}")
            plt.close()
            return ""
    
    def _create_error_image(self, error_message: str, filename: str) -> str:
        """
        오류 메시지를 포함한 기본 이미지를 생성합니다.
        
        Args:
            error_message (str): 표시할 오류 메시지
            filename (str): 저장할 파일 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 그림 생성
            plt.figure(figsize=self.figure_size)
            
            # 배경 색상 설정
            ax = plt.gca()
            ax.set_facecolor('#ffeeee')  # 연한 빨간색 배경
            
            # 테두리 설정
            for spine in ax.spines.values():
                spine.set_edgecolor('red')
                spine.set_linewidth(2)
                
            # 오류 텍스트 표시
            plt.text(0.5, 0.5, f"오류 발생:\n{error_message}",
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=12,
                   color='darkred',
                   transform=ax.transAxes,
                   wrap=True)
                   
            # 추가 안내
            plt.text(0.5, 0.3, "이 이미지는 시각화 중 오류가 발생할 때 생성됩니다.\n데이터를 확인하고 다시 시도해주세요.",
                   horizontalalignment='center',
                   verticalalignment='center',
                   fontsize=10,
                   color='gray',
                   transform=ax.transAxes)
                   
            # 축 숨기기
            plt.axis('off')
            
            # 그림 저장
            return self._save_figure(filename)
        except Exception as e:
            self.logger.error(f"오류 이미지 생성 중에도 문제 발생: {str(e)}")
            plt.close()
            return ""
            
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
        try:
            # 데이터 검증
            series = validate_numeric_column(self.df, column)
            
            # 결측치 제거
            series = series.dropna()
            
            if len(series) == 0:
                raise ValueError(f"열 '{column}'에 유효한 데이터가 없습니다.")
            
            # 데이터 분포에 따른 bins 최적화
            if bins == 'auto':
                # Freedman-Diaconis 규칙으로 최적의 bin 수 추정
                q75, q25 = np.percentile(series, [75, 25])
                iqr = q75 - q25
                bin_width = 2 * iqr / (len(series) ** (1/3))
                if bin_width > 0:
                    data_range = series.max() - series.min()
                    bins = max(10, min(100, int(data_range / bin_width)))
                else:
                    bins = min(50, len(series.unique()))
            
            # 그래프 생성
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # 한글 폰트 설정 확인 및 명시적 설정
            self.logger.info(f"현재 폰트 설정: {plt.rcParams['font.family']}")
            self.logger.info(f"현재 sans-serif 폰트: {plt.rcParams['font.sans-serif']}")
            
            # 한글 폰트 명시적 설정
            plt.rcParams['font.family'] = 'sans-serif'
            if 'Malgun Gothic' not in plt.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = ['Malgun Gothic'] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 히스토그램 그리기
            sns.histplot(series, bins=bins, kde=kde)
            
            # 그래프 제목 및 레이블 설정
            plt.title(title or f"{column} 분포", fontproperties=self._get_korean_font())
            plt.xlabel(column, fontproperties=self._get_korean_font())
            plt.ylabel("빈도", fontproperties=self._get_korean_font())
            plt.grid(True, alpha=0.3)
            
            # 파일명 생성
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"histogram_{column}_{timestamp}.png"
                
            # 파일 저장
            return self._save_figure(filename)
            
        except Exception as e:
            self.logger.error(f"히스토그램 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 대체 이미지 반환
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"error_histogram_{column}_{timestamp}.png"
                
            return self._create_error_image(f"히스토그램 생성 실패: {str(e)}", filename)
            
    def plot_bar(self, column: str, horizontal: bool = False, 
                 top_n: int = 10, show_others: bool = True,
                 title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        막대 그래프를 생성합니다.
        
        Args:
            column (str): 데이터 열 이름
            horizontal (bool): 수평 막대 그래프 여부 (기본값: False)
            top_n (int): 표시할 상위 항목 수
            show_others (bool): 상위 N개 이외 범주를 '기타'로 표시할지 여부
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 범주형 데이터 검증
            categories, counts = validate_categorical_column(
                self.df, column, top_n if show_others else None
            )
            
            # 데이터 정렬 (내림차순)
            categories_sorted = []
            counts_sorted = []
            for cat, count in sorted(zip(categories, counts), key=lambda x: x[1], reverse=True):
                categories_sorted.append(cat)
                counts_sorted.append(count)
            
            categories = categories_sorted
            counts = counts_sorted
            
            # 그래프 생성
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # 그래프 스타일 설정
            sns.set_style("whitegrid")
            plt.grid(axis='x' if horizontal else 'y', linestyle='-', alpha=0.2)
            
            # 한글 폰트 설정 확인 및 명시적 설정
            self.logger.info(f"현재 폰트 설정: {plt.rcParams['font.family']}")
            self.logger.info(f"현재 sans-serif 폰트: {plt.rcParams['font.sans-serif']}")
            
            # 한글 폰트 명시적 설정
            plt.rcParams['font.family'] = 'sans-serif'
            if 'Malgun Gothic' not in plt.rcParams['font.sans-serif']:
                plt.rcParams['font.sans-serif'] = ['Malgun Gothic'] + plt.rcParams['font.sans-serif']
            plt.rcParams['axes.unicode_minus'] = False
            
            korean_font = self._get_korean_font()
            
            # 데이터가 많으면 그림 크기 조정
            if len(categories) > 8:
                if horizontal:
                    plt.figure(figsize=(self.figure_size[0], min(20, self.figure_size[1] + len(categories) * 0.3)))
                else:
                    plt.figure(figsize=(min(20, self.figure_size[0] + len(categories) * 0.2), self.figure_size[1]))
            
            # 막대 그래프 그리기
            ax = plt.gca()
            
            # 테두리 제거
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # 값에 따른 색상 설정 (값이 클수록 더 짙은 색)
            max_count = max(counts) if counts else 1
            # 색상 농도 계산 (값이 클수록 짙은 색)
            norm = plt.Normalize(0, max_count)
            colors = plt.cm.Blues(norm(counts))
            
            if horizontal:
                # 수평 막대 그래프
                bars = plt.barh(categories, counts, color=colors, alpha=0.9)
                
                # 값 레이블 추가
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    plt.text(width + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                             f"{width:,}", va='center', fontproperties=korean_font)
                
                plt.xlabel("빈도", fontproperties=korean_font, fontsize=11)
                plt.ylabel("", fontsize=0)  # Y축 레이블 제거
                
                # Y축 레이블 텍스트 설정
                max_len = max([len(str(cat)) for cat in categories])
                if max_len > 15:
                    import textwrap
                    wrapped_labels = [textwrap.fill(str(cat), 15) for cat in categories]
                    plt.yticks(range(len(categories)), wrapped_labels, fontproperties=korean_font)
            else:
                # 수직 막대 그래프
                bars = plt.bar(categories, counts, color=colors, alpha=0.9)
                
                # 값 레이블 추가
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, height + max(counts) * 0.01,
                             f"{height:,}", ha='center', va='bottom', fontproperties=korean_font)
                
                plt.ylabel("빈도", fontproperties=korean_font, fontsize=11)
                plt.xlabel("", fontsize=0)  # X축 레이블 제거
                
                # X축 레이블 회전 설정 대신 줄바꿈 적용
                if max([len(str(cat)) for cat in categories]) > 5 or len(categories) > 5:
                    import textwrap
                    wrapped_labels = [textwrap.fill(str(cat), width=10) for cat in categories]
                    plt.xticks(range(len(categories)), wrapped_labels, fontproperties=korean_font)
            
            # 그래프 제목 설정
            plt.title(title or f"{column} 분포", fontproperties=korean_font, fontsize=14)
            
            # 레이아웃 조정
            plt.tight_layout(pad=1.2)
            
            # 파일명 생성
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"bar_{column}_{timestamp}.png"
                
            # 파일 저장
            return self._save_figure(filename)
            
        except Exception as e:
            self.logger.error(f"막대 그래프 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 대체 이미지 반환
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"error_bar_{column}_{timestamp}.png"
                
            return self._create_error_image(f"막대 그래프 생성 실패: {str(e)}", filename)
            
    def plot_scatter(self, x_column: str, y_column: str, hue_column: Optional[str] = None,
                     title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        산점도를 생성합니다.
        
        Args:
            x_column (str): x축 데이터 열 이름
            y_column (str): y축 데이터 열 이름
            hue_column (str, optional): 색상 구분을 위한 열 이름
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        try:
            # 데이터 검증
            x_data = validate_numeric_column(self.df, x_column)
            y_data = validate_numeric_column(self.df, y_column)
            
            # hue 열 검증
            hue_data = None
            if hue_column is not None:
                try:
                    hue_data = validate_column_data(self.df, hue_column)
                except ValueError as e:
                    self.logger.warning(f"hue 열 '{hue_column}' 검증 실패: {str(e)}. hue 없이 진행합니다.")
                    hue_column = None
            
            # 임시 데이터프레임 생성
            data = pd.DataFrame({x_column: x_data, y_column: y_data})
            if hue_column:
                data[hue_column] = hue_data
            
            # 결측치 제거
            data = data.dropna()
            
            if len(data) == 0:
                raise ValueError("유효한 데이터가 없습니다.")
            
            # 그래프 생성
            plt.figure(figsize=self.figure_size, dpi=self.dpi)
            
            # 산점도 그리기
            if hue_column:
                scatter = sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue_column)
                plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(data=data, x=x_column, y=y_column)
            
            # 그래프 제목 및 레이블 설정
            plt.title(title or f"{y_column} vs {x_column}")
            plt.xlabel(x_column)
            plt.ylabel(y_column)
            plt.grid(True, alpha=0.3)
            
            # 파일명 생성
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"scatter_{x_column}_{y_column}_{timestamp}.png"
            
            # 파일 저장
            return self._save_figure(filename)
            
        except Exception as e:
            self.logger.error(f"산점도 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 대체 이미지 반환
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"error_scatter_{x_column}_{y_column}_{timestamp}.png"
                
            return self._create_error_image(f"산점도 생성 실패: {str(e)}", filename)
            
    # 한글 폰트 객체 가져오기 메서드 추가
    def _get_korean_font(self):
        try:
            import matplotlib.font_manager as fm
            # 맑은 고딕 폰트 경로 (Windows)
            font_path = "C:/Windows/Fonts/malgun.ttf"
            if not os.path.exists(font_path):
                # macOS 기본 한글 폰트
                font_path = '/Library/Fonts/AppleGothic.ttf'
                if not os.path.exists(font_path):
                    # Linux 나눔고딕 폰트
                    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
                    
            if os.path.exists(font_path):
                return fm.FontProperties(fname=font_path)
            else:
                # 폰트를 찾을 수 없는 경우 기본 폰트 속성 반환
                return fm.FontProperties(family='sans-serif')
        except Exception as e:
            self.logger.error(f"한글 폰트 가져오기 실패: {str(e)}")
            return fm.FontProperties(family='sans-serif') 