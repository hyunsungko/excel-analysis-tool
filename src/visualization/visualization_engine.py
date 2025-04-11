import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Union, Tuple, Any
import os
import logging
from datetime import datetime
import matplotlib
from pathlib import Path
import matplotlib.font_manager as fm

# 폰트 매니저 캐시 초기화 코드 추가
def reset_font_cache():
    """폰트 캐시를 재구성합니다."""
    try:
        # 최신 버전의 matplotlib에서는 fontManager.json_dump() 호출 후 재생성
        fm.fontManager.json_dump()
        logging.info("폰트 캐시를 재구성했습니다. (json_dump 메서드 사용)")
        return True
    except Exception as e1:
        try:
            # 구버전 matplotlib에서는 _rebuild() 메서드 사용
            if hasattr(fm, '_rebuild'):
                fm._rebuild()
                logging.info("폰트 캐시를 재구성했습니다. (_rebuild 메서드 사용)")
                return True
            else:
                # 캐시 재구성이 불가능한 경우
                logging.warning(f"폰트 캐시 재구성 방법을 찾을 수 없습니다: {str(e1)}")
                
                # 대안으로 fontManager 인스턴스 재생성 시도
                try:
                    fm.fontManager = fm.FontManager()
                    logging.info("폰트 매니저 인스턴스를 재생성했습니다.")
                    return True
                except Exception as e2:
                    logging.warning(f"폰트 매니저 재생성 실패: {str(e2)}")
        except Exception as e3:
            logging.warning(f"폰트 캐시 재구성 중 오류 발생: {str(e3)}")
    
    # 캐시 재구성을 건너뛰고 계속 진행
    return False

# 한글 폰트 설정 추가
def configure_korean_font():
    """한글 폰트 설정을 구성합니다."""
    import platform
    
    # 폰트 캐시 초기화 (실패하더라도 계속 진행)
    try:
        reset_font_cache()
    except Exception as e:
        logging.warning(f"폰트 캐시 재구성 중 오류 발생, 계속 진행합니다: {str(e)}")
    
    # Windows 환경인 경우
    if platform.system() == 'Windows':
        try:
            # 폰트 직접 등록 방식
            # 모든 사용 가능한 폰트 확인 (디버깅용)
            try:
                font_list = [f.name for f in fm.fontManager.ttflist]
                logging.info(f"사용 가능한 폰트 목록 (일부): {font_list[:10]}")
            except Exception as e:
                logging.warning(f"폰트 목록 조회 중 오류: {str(e)}")
            
            # 여러 한글 폰트 경로 시도
            font_paths = [
                r'C:\Users\ffgtt\AppData\Local\Microsoft\Windows\Fonts\현대하모니+B.ttf',  # 현대하모니 B 폰트
                r'C:\Users\ffgtt\AppData\Local\Microsoft\Windows\Fonts\현대하모니+M.ttf',  # 현대하모니 M 폰트
                r'C:\Users\ffgtt\AppData\Local\Microsoft\Windows\Fonts\현대하모니+L.ttf',  # 현대하모니 L 폰트
                r'C:\Windows\Fonts\NotoSansKR-VF.ttf',  # Noto Sans KR 가변 폰트
                r'C:\Windows\Fonts\malgun.ttf',         # 맑은 고딕
                r'C:\Windows\Fonts\HANBatang.ttf',      # 한 바탕
                r'C:\Windows\Fonts\HANDotum.ttf'        # 한 돋움
            ]
            
            font_found = False
            for font_path in font_paths:
                if os.path.exists(font_path):
                    logging.info(f"폰트 파일 확인됨: {font_path}")
                    
                    # 폰트 속성 가져오기
                    font_prop = fm.FontProperties(fname=font_path)
                    
                    # 폰트 이름 가져오기
                    font_name = font_prop.get_name()
                    
                    # 폰트 패밀리에 한글 폰트를 추가하는 방식으로 변경
                    plt.rcParams['font.family'] = 'sans-serif'
                    if 'font.sans-serif' in plt.rcParams:
                        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    else:
                        plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                    
                    logging.info(f"한글 폰트 '{font_name}' 등록 완료")
                    font_found = True
                    break
            
            if not font_found:
                raise Exception("유효한 한글 폰트 파일을 찾을 수 없습니다.")
                
            return font_found
        except Exception as e:
            # 기본 폰트 방식으로 시도
            font_names = [
                '현대하모니 B',   # 현대하모니 B
                '현대하모니 M',   # 현대하모니 M
                '현대하모니 L',   # 현대하모니 L
                'Noto Sans KR',  # 노토 산스 KR
                'Malgun Gothic', # 맑은 고딕
                'Gulim',         # 굴림
                'Batang',        # 바탕
                'Gungsuh'        # 궁서
            ]
            
            # 사용 가능한 폰트 찾기
            font_found = False
            for font_name in font_names:
                try:
                    plt.rcParams['font.family'] = 'sans-serif'
                    if 'font.sans-serif' in plt.rcParams:
                        plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                    else:
                        plt.rcParams['font.sans-serif'] = [font_name]
                    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
                    font_found = True
                    logging.info(f"한글 폰트 '{font_name}'을(를) 사용합니다.")
                    break
                except:
                    continue
            
            if not font_found:
                logging.warning(f"기본 한글 폰트를 찾을 수 없습니다: {str(e)}")
                return False
            
            return font_found
    # macOS 환경인 경우
    elif platform.system() == 'Darwin':
        try:
            plt.rcParams['font.family'] = 'AppleGothic'
            plt.rcParams['axes.unicode_minus'] = False
            logging.info("한글 폰트 'AppleGothic'을(를) 사용합니다.")
            return True
        except:
            logging.warning("macOS 한글 폰트를 찾을 수 없습니다. 텍스트가 깨질 수 있습니다.")
            return False
    # Linux 환경인 경우
    else:
        font_names = [
            'NanumGothic',
            'NanumBarunGothic',
            'UnDotum',
            'UnBatang'
        ]
        
        font_found = False
        for font_name in font_names:
            try:
                plt.rcParams['font.family'] = 'sans-serif'
                if 'font.sans-serif' in plt.rcParams:
                    plt.rcParams['font.sans-serif'] = [font_name] + plt.rcParams['font.sans-serif']
                else:
                    plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                font_found = True
                logging.info(f"한글 폰트 '{font_name}'을(를) 사용합니다.")
                break
            except:
                continue
                
        if not font_found:
            logging.warning("기본 한글 폰트를 찾을 수 없습니다. 텍스트가 깨질 수 있습니다.")
            return False
        
        return font_found

# 클래스 시작 전에 폰트 설정 호출
configure_korean_font()

class VisualizationEngine:
    """
    데이터 시각화를 담당하는 엔진 클래스
    다양한 차트와 그래프를 생성하고 저장하는 기능 제공
    """
    
    def __init__(self, output_dir: str = 'output'):
        """
        VisualizationEngine 초기화
        
        Args:
            output_dir (str): 시각화 결과를 저장할 디렉토리
        """
        self.df = None
        self.output_dir = output_dir
        self.theme = 'default'
        self.figure_size = (10, 6)
        self.dpi = 100
        self.current_figure = None
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 한글 폰트 설정 재적용 (중요: 각 인스턴스 생성 시마다 적용)
        configure_korean_font()
        
        # 현재 설정된 폰트 확인 (디버깅용)
        self.logger.info(f"폰트 패밀리: {plt.rcParams['font.family']}")
        if 'font.sans-serif' in plt.rcParams:
            self.logger.info(f"sans-serif 폰트: {plt.rcParams['font.sans-serif']}")
            
        # 폰트 속성 객체 생성 (B 스타일 사용)
        self.korean_font_paths = {
            'B': r'C:\Users\ffgtt\AppData\Local\Microsoft\Windows\Fonts\현대하모니+B.ttf',  # 현대하모니 B 폰트
            'M': r'C:\Users\ffgtt\AppData\Local\Microsoft\Windows\Fonts\현대하모니+M.ttf',  # 현대하모니 M 폰트
            'L': r'C:\Users\ffgtt\AppData\Local\Microsoft\Windows\Fonts\현대하모니+L.ttf'   # 현대하모니 L 폰트
        }
        
        # 폰트 속성 객체 미리 생성
        self.font_props = {}
        for style, path in self.korean_font_paths.items():
            if os.path.exists(path):
                try:
                    self.font_props[style] = fm.FontProperties(fname=path)
                    self.logger.info(f"'{style}' 스타일 폰트 속성 객체 생성 성공")
                except Exception as e:
                    self.logger.warning(f"폰트 속성 객체 생성 실패 ({style}): {str(e)}")
    
    # 폰트 스타일 문자열로 FontProperties 객체 반환하는 헬퍼 메서드
    def get_font_prop(self, style='B'):
        """폰트 속성 객체 반환 (기본값: B 스타일)"""
        if style in self.font_props:
            return self.font_props[style]
        elif 'B' in self.font_props:
            return self.font_props['B']  # 기본 대체 스타일
        elif len(self.font_props) > 0:
            return list(self.font_props.values())[0]  # 첫 번째 사용 가능한 폰트
        else:
            return None  # 사용 가능한 폰트 없음
    
    def generate_visualizations(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        분석 결과를 기반으로 주요 시각화 생성
        
        Args:
            df (pd.DataFrame): 시각화할 데이터프레임
            analysis_results (Dict[str, Any]): 분석 결과 데이터
            
        Returns:
            List[Dict[str, Any]]: 생성된 시각화 정보 목록
        """
        # 한글 폰트 설정 확인 (각 그래프 생성 전에 확인)
        configure_korean_font()
        
        self.logger.info("주요 시각화 생성 시작")
        self.set_dataframe(df)
        
        visualizations = []
        
        try:
            # 1. 결측치 시각화
            missing_viz_path = self.plot_missing_data(
                filename="missing_data_chart.png",
                title="결측치 분포"
            )
            if missing_viz_path:
                visualizations.append({
                    'type': 'missing_data',
                    'title': '결측치 분포',
                    'path': missing_viz_path,
                    'description': '각 열의 결측치 비율을 보여주는 차트입니다.'
                })
            
            # 2. 수치형 변수 분포
            if 'basic_stats' in analysis_results:
                numeric_cols = list(analysis_results['basic_stats'].keys())[:5]  # 상위 5개만
                for col in numeric_cols:
                    hist_path = self.plot_histogram(
                        column=col,
                        filename=f"histogram_{col.replace(' ', '_')}.png",
                        title=f"{col} 분포"
                    )
                    if hist_path:
                        visualizations.append({
                            'type': 'histogram',
                            'title': f"{col} 분포",
                            'path': hist_path,
                            'description': f'{col} 열의 값 분포를 보여주는 히스토그램입니다.'
                        })
            
            # 3. 범주형 변수 분포
            if 'categorical_stats' in analysis_results:
                cat_cols = list(analysis_results['categorical_stats'].keys())[:5]  # 상위 5개만
                for col in cat_cols:
                    if col in df.columns:
                        bar_path = self.plot_bar(
                            column=col,
                            filename=f"bar_{col.replace(' ', '_')}.png",
                            title=f"{col} 범주별 빈도"
                        )
                        if bar_path:
                            visualizations.append({
                                'type': 'bar',
                                'title': f"{col} 범주별 빈도",
                                'path': bar_path,
                                'description': f'{col} 열의 범주별 빈도를 보여주는 막대 그래프입니다.'
                            })
            
            # 4. 상관관계 히트맵
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) >= 2:
                corr_path = self.plot_correlation_heatmap(
                    filename="correlation_heatmap.png",
                    title="변수 간 상관관계"
                )
                if corr_path:
                    visualizations.append({
                        'type': 'correlation',
                        'title': '변수 간 상관관계',
                        'path': corr_path,
                        'description': '수치형 변수 간 상관관계를 보여주는 히트맵입니다.'
                    })
            
            # 5. 페어플롯 (수치형 변수가 2~5개일 경우)
            if 2 <= len(numeric_cols) <= 5:
                pair_path = self.plot_pairplot(
                    columns=numeric_cols.tolist(),
                    filename="pairplot.png",
                    title="변수 간 산점도 행렬"
                )
                if pair_path:
                    visualizations.append({
                        'type': 'pairplot',
                        'title': '변수 간 산점도 행렬',
                        'path': pair_path,
                        'description': '수치형 변수 간의 관계를 보여주는 산점도 행렬입니다.'
                    })
            
            # 6. 범주형 변수별 수치형 변수 분포 (박스플롯)
            if 'categorical_stats' in analysis_results and len(numeric_cols) > 0:
                cat_cols = list(analysis_results['categorical_stats'].keys())
                valid_cat_cols = [col for col in cat_cols 
                                 if col in df.columns and 2 <= df[col].nunique() <= 10]
                
                if valid_cat_cols and len(numeric_cols) > 0:
                    # 첫 번째 범주형 변수와 첫 번째 수치형 변수로 박스플롯 생성
                    box_path = self.plot_boxplot(
                        column=numeric_cols[0],
                        by=valid_cat_cols[0],
                        filename=f"boxplot_{numeric_cols[0]}_{valid_cat_cols[0]}.png",
                        title=f"{valid_cat_cols[0]}별 {numeric_cols[0]} 분포"
                    )
                    if box_path:
                        visualizations.append({
                            'type': 'boxplot',
                            'title': f"{valid_cat_cols[0]}별 {numeric_cols[0]} 분포",
                            'path': box_path,
                            'description': f'{valid_cat_cols[0]}의 각 범주별 {numeric_cols[0]} 분포를 보여주는 박스플롯입니다.'
                        })
            
        except Exception as e:
            self.logger.error(f"시각화 생성 중 오류 발생: {str(e)}")
        
        self.logger.info(f"주요 시각화 생성 완료: {len(visualizations)}개 생성됨")
        return visualizations
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        시각화할 데이터프레임 설정
        
        Args:
            df (pd.DataFrame): 시각화 대상 데이터프레임
        """
        self.df = df
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
        self.logger.info(f"그림 크기 설정 완료: {width}x{height}, DPI: {dpi}")
        
    def _get_filename(self, prefix: str, ext: str = 'png') -> str:
        """
        시간 기반 고유한 파일명 생성
        
        Args:
            prefix (str): 파일명 접두사
            ext (str): 파일 확장자
            
        Returns:
            str: 생성된 파일명
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{prefix}_{timestamp}.{ext}"
        
    def _save_figure(self, filename: Optional[str] = None, tight_layout: bool = True) -> str:
        """
        현재 그림을 파일로 저장
        
        Args:
            filename (str, optional): 저장할 파일 이름
            tight_layout (bool): tight_layout 적용 여부
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.current_figure is None:
            return ""
        
        if tight_layout:
            plt.tight_layout()
        
        if filename is None:
            filename = self._get_filename("plot")
            
        # 파일 경로 생성 시 안전한 경로 처리 추가
        safe_filename = filename.replace('/', '_').replace('\\', '_')
        file_path = os.path.join(self.output_dir, safe_filename)
        
        try:
            plt.savefig(file_path, dpi=self.dpi, bbox_inches='tight')
            self.logger.info(f"그림 저장 완료: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"그림 저장 중 오류 발생: {str(e)}")
            return ""
            
    def _handle_column_with_timedelta(self, series: pd.Series) -> pd.Series:
        """
        Timedelta 타입 열을 초 단위 수치형으로 변환
        
        Args:
            series (pd.Series): 변환할 시리즈
            
        Returns:
            pd.Series: 변환된 시리즈
        """
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds()
        return series
            
    def plot_histogram(self, column: str, bins: int = 20, kde: bool = True, 
                      title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        지정한 열의 히스토그램 생성
        
        Args:
            column (str): 히스토그램을 생성할 열 이름
            bins (int): 구간 수
            kde (bool): 커널 밀도 추정 표시 여부
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        # 한글 폰트 설정 확인 (각 그래프 생성 전에 확인)
        configure_korean_font()
        
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        if column not in self.df.columns:
            self.logger.error(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        # 결측치가 아닌 데이터만 선택
        data = self.df[column].dropna()
        
        if len(data) == 0:
            self.logger.warning(f"열 '{column}'에 유효한 데이터가 없습니다.")
            return ""
            
        # Timedelta 타입 처리
        if pd.api.types.is_timedelta64_dtype(data):
            data = self._handle_column_with_timedelta(data)
            # 열 이름 수정
            column = f"{column} (seconds)"
            
        # 범주형 데이터 처리
        if isinstance(data.dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(data):
            self.logger.info(f"'{column}'은 범주형 데이터입니다. 막대 그래프로 대체합니다.")
            return self.plot_bar(column, filename=filename)
            
        try:
            # 그림 생성
            plt.figure(figsize=self.figure_size)
            
            # 히스토그램 그리기
            sns.histplot(data=data, kde=kde, bins=bins)
            
            # 폰트 속성 가져오기
            font_prop = self.get_font_prop('B')
            
            # 그래프 설정 - 직접 폰트 지정
            if title:
                plt.title(title, fontproperties=font_prop, fontsize=14)
            else:
                plt.title(f"Distribution of {column}", fontproperties=font_prop, fontsize=14)
                
            plt.xlabel(column, fontproperties=font_prop, fontsize=12)
            plt.ylabel("Frequency", fontproperties=font_prop, fontsize=12)
            plt.grid(True, alpha=0.3)
            
            # 눈금 레이블에도 폰트 적용
            plt.xticks(fontproperties=font_prop, fontsize=10)
            plt.yticks(fontproperties=font_prop, fontsize=10)
            
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                filename = self._get_filename(f"histogram_{column.replace(' ', '_')}")
                
            # 그림 저장
            file_path = self._save_figure(filename)
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"히스토그램 플롯 생성 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_bar(self, column: str, horizontal: bool = False, top_n: int = 10,
                title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        지정한 열의 막대 그래프 생성
        
        Args:
            column (str): 막대 그래프를 생성할 열 이름
            horizontal (bool): 수평 막대 그래프 여부
            top_n (int): 표시할 최대 항목 수
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        # 한글 폰트 설정 확인 (각 그래프 생성 전에 확인)
        configure_korean_font()
        
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        if column not in self.df.columns:
            self.logger.error(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        # 결측치가 아닌 데이터만 선택
        data = self.df[column].dropna()
        
        if len(data) == 0:
            self.logger.warning(f"열 '{column}'에 유효한 데이터가 없습니다.")
            return ""
            
        try:
            # 값 빈도 계산
            value_counts = data.value_counts().sort_values(ascending=False)
            
            # 상위 N개 값만 선택
            if len(value_counts) > top_n:
                value_counts = value_counts.head(top_n)
                
            # 그림 생성
            plt.figure(figsize=self.figure_size)
            
            # 폰트 속성 가져오기
            font_prop = self.get_font_prop('B')
            
            # 막대 그래프 그리기 (가로/세로)
            if horizontal:
                ax = sns.barplot(y=value_counts.index, x=value_counts.values)
                
                # 값 레이블 추가
                for i, v in enumerate(value_counts.values):
                    ax.text(v + 0.1, i, str(v), va='center', fontproperties=font_prop)
                    
                plt.xlabel("Count", fontproperties=font_prop, fontsize=12)
                plt.ylabel(column, fontproperties=font_prop, fontsize=12)
            else:
                ax = sns.barplot(x=value_counts.index, y=value_counts.values)
                
                # 값 레이블 추가
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v + 0.1, str(v), ha='center', fontproperties=font_prop)
                    
                plt.xlabel(column, fontproperties=font_prop, fontsize=12)
                plt.ylabel("Count", fontproperties=font_prop, fontsize=12)
                
                # 긴 레이블 회전
                if max([len(str(x)) for x in value_counts.index]) > 10:
                    plt.xticks(rotation=45, ha='right', fontproperties=font_prop, fontsize=10)
                else:
                    plt.xticks(fontproperties=font_prop, fontsize=10)
                
            # y축 레이블 폰트 설정
            plt.yticks(fontproperties=font_prop, fontsize=10)
                
            # 그래프 설정
            if title:
                plt.title(title, fontproperties=font_prop, fontsize=14)
            else:
                plt.title(f"Frequency of {column}", fontproperties=font_prop, fontsize=14)
                
            plt.grid(True, alpha=0.3)
            
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                orientation = "horizontal" if horizontal else "vertical"
                filename = self._get_filename(f"bar_{orientation}_{column.replace(' ', '_')}")
                
            # 그림 저장
            file_path = self._save_figure(filename)
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"막대 그래프 플롯 생성 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_scatter(self, x_column: str, y_column: str, hue_column: Optional[str] = None,
                    title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        두 열 간의 산점도 생성
        
        Args:
            x_column (str): x축 데이터 열 이름
            y_column (str): y축 데이터 열 이름
            hue_column (str, optional): 색상 구분을 위한 열 이름
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        # 한글 폰트 설정 확인 (각 그래프 생성 전에 확인)
        configure_korean_font()
        
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        if x_column not in self.df.columns:
            self.logger.error(f"열 '{x_column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        if y_column not in self.df.columns:
            self.logger.error(f"열 '{y_column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        if hue_column is not None and hue_column not in self.df.columns:
            self.logger.warning(f"열 '{hue_column}'이 데이터프레임에 존재하지 않습니다. hue 없이 진행합니다.")
            hue_column = None
            
        # 필요한 열만 선택하고 결측치 제거
        columns = [x_column, y_column]
        if hue_column:
            columns.append(hue_column)
            
        data = self.df[columns].dropna()
        
        if len(data) == 0:
            self.logger.warning("유효한 데이터가 없습니다.")
            return ""
            
        # Timedelta 타입 처리
        x_data = data[x_column]
        if pd.api.types.is_timedelta64_dtype(x_data):
            data[x_column] = self._handle_column_with_timedelta(x_data)
            x_label = f"{x_column} (seconds)"
        else:
            x_label = x_column
            
        y_data = data[y_column]
        if pd.api.types.is_timedelta64_dtype(y_data):
            data[y_column] = self._handle_column_with_timedelta(y_data)
            y_label = f"{y_column} (seconds)"
        else:
            y_label = y_column
            
        try:
            # 그림 생성
            plt.figure(figsize=self.figure_size)
            
            # 산점도 그리기
            if hue_column:
                scatter = sns.scatterplot(data=data, x=x_column, y=y_column, hue=hue_column)
                plt.legend(title=hue_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                scatter = sns.scatterplot(data=data, x=x_column, y=y_column)
                
            # 그래프 설정
            if title:
                plt.title(title)
            else:
                plt.title(f"{y_column} vs {x_column}")
                
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                filename = self._get_filename(f"scatter_{x_column}_{y_column}".replace(' ', '_'))
                
            # 그림 저장
            file_path = self._save_figure(filename)
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"산점도 플롯 생성 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_boxplot(self, column: str, by: Optional[str] = None, horizontal: bool = False,
                    title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        지정한 열의 박스플롯 생성
        
        Args:
            column (str): 박스플롯을 생성할 열 이름
            by (str, optional): 그룹화할 열 이름
            horizontal (bool): 수평 박스플롯 여부
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일 이름
            
        Returns:
            str: 저장된 파일 경로
        """
        # 한글 폰트 설정 확인 (각 그래프 생성 전에 확인)
        configure_korean_font()
        
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        if column not in self.df.columns:
            self.logger.error(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        if by is not None and by not in self.df.columns:
            self.logger.warning(f"열 '{by}'이 데이터프레임에 존재하지 않습니다. 그룹화 없이 진행합니다.")
            by = None
            
        # 결측치가 아닌 데이터만 선택
        if by:
            data = self.df[[column, by]].dropna()
        else:
            data = self.df[column].dropna()
            
        if len(data) == 0:
            self.logger.warning(f"열 '{column}'에 유효한 데이터가 없습니다.")
            return ""
            
        # Timedelta 타입 처리
        if pd.api.types.is_timedelta64_dtype(self.df[column]):
            if by:
                data[column] = self._handle_column_with_timedelta(data[column])
                # 열 이름 수정
                column_label = f"{column} (seconds)"
            else:
                data = self._handle_column_with_timedelta(data)
                column_label = f"{column} (seconds)"
        else:
            column_label = column
            
        try:
            # 그림 생성
            plt.figure(figsize=self.figure_size)
            
            # 박스플롯 그리기
            if by:
                if horizontal:
                    ax = sns.boxplot(data=data, y=by, x=column)
                    plt.ylabel(by)
                    plt.xlabel(column_label)
                else:
                    ax = sns.boxplot(data=data, x=by, y=column)
                    plt.xlabel(by)
                    plt.ylabel(column_label)
                    
                    # 긴 레이블 회전
                    if max([len(str(x)) for x in data[by].unique()]) > 10:
                        plt.xticks(rotation=45, ha='right')
            else:
                if horizontal:
                    ax = sns.boxplot(data=data, orient='h')
                    plt.xlabel(column_label)
                    plt.ylabel("")
                else:
                    ax = sns.boxplot(data=data)
                    plt.xlabel("")
                    plt.ylabel(column_label)
                    
            # 그래프 설정
            if title:
                plt.title(title)
            else:
                if by:
                    plt.title(f"Boxplot of {column} by {by}")
                else:
                    plt.title(f"Boxplot of {column}")
                    
            plt.grid(True, alpha=0.3)
            
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                if by:
                    filename = self._get_filename(f"boxplot_{column}_by_{by}".replace(' ', '_'))
                else:
                    filename = self._get_filename(f"boxplot_{column}".replace(' ', '_'))
                    
            # 그림 저장
            file_path = self._save_figure(filename)
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"박스플롯 생성 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_line(self, x_column: str, y_column: str, group_column: Optional[str] = None,
                 title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        선 그래프 생성
        
        Args:
            x_column (str): X축 데이터 열 이름
            y_column (str): Y축 데이터 열 이름
            group_column (str, optional): 그룹화 기준 열 이름
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        if x_column not in self.df.columns:
            self.logger.error(f"열 '{x_column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        if y_column not in self.df.columns:
            self.logger.error(f"열 '{y_column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        if group_column is not None and group_column not in self.df.columns:
            self.logger.warning(f"열 '{group_column}'이 데이터프레임에 존재하지 않습니다. 그룹화 없이 진행합니다.")
            group_column = None
            
        # 필요한 열만 선택하고 결측치 제거
        columns = [x_column, y_column]
        if group_column:
            columns.append(group_column)
            
        data = self.df[columns].dropna()
        
        if len(data) == 0:
            self.logger.warning("유효한 데이터가 없습니다.")
            return ""
            
        # 날짜/시간 데이터 확인 및 정렬
        if pd.api.types.is_datetime64_any_dtype(data[x_column]):
            data = data.sort_values(by=x_column)
            
        # Timedelta 타입 처리
        if pd.api.types.is_timedelta64_dtype(data[y_column]):
            data[y_column] = self._handle_column_with_timedelta(data[y_column])
            y_label = f"{y_column} (seconds)"
        else:
            y_label = y_column
            
        try:
            # 그림 생성
            plt.figure(figsize=self.figure_size)
            
            # 선 그래프 그리기
            if group_column:
                groups = data[group_column].unique()
                
                for group in groups:
                    group_data = data[data[group_column] == group]
                    plt.plot(group_data[x_column], group_data[y_column], marker='o', label=group)
                    
                plt.legend(title=group_column, bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.plot(data[x_column], data[y_column], marker='o')
                
            # 그래프 설정
            if title:
                plt.title(title)
            else:
                plt.title(f"{y_column} over {x_column}")
                
            plt.xlabel(x_column)
            plt.ylabel(y_label)
            plt.grid(True, alpha=0.3)
            
            # X축 레이블 조정
            if pd.api.types.is_datetime64_any_dtype(data[x_column]) or len(str(data[x_column].iloc[0])) > 10:
                plt.xticks(rotation=45, ha='right')
                
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                filename = self._get_filename(f"line_{x_column}_{y_column}".replace(' ', '_'))
                
            # 그림 저장
            file_path = self._save_figure(filename)
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"선 그래프 생성 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_correlation_heatmap(self, columns: Optional[List[str]] = None, 
                               min_corr: float = 0.0, method: str = 'pearson',
                               title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        상관관계 히트맵 생성
        
        Args:
            columns (List[str], optional): 분석할 열 목록. None인 경우 모든 수치형 열 사용
            min_corr (float): 최소 상관계수 임계값 (절대값 기준). 이 값보다 작은 상관관계는 0으로 설정
            method (str): 상관관계 계산 방법 ('pearson', 'spearman', 'kendall')
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        # 수치형 열만 선택
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            # Timedelta 열도 처리
            for col in self.df.select_dtypes(include=['timedelta64[ns]']).columns:
                numeric_cols.append(col)
        else:
            numeric_cols = [col for col in columns if col in self.df.columns]
            
        if not numeric_cols:
            self.logger.warning("분석할 수치형 열이 없습니다.")
            return ""
            
        # Timedelta 열 변환
        data = self.df[numeric_cols].copy()
        for col in data.columns:
            if pd.api.types.is_timedelta64_dtype(data[col]):
                data[col] = self._handle_column_with_timedelta(data[col])
                
        # 결측치 제거
        data = data.dropna()
        
        if len(data) == 0:
            self.logger.warning("유효한 데이터가 없습니다.")
            return ""
            
        try:
            # 상관관계 계산
            corr_matrix = data.corr(method=method)
            
            # 최소 상관계수 임계값 적용
            if min_corr > 0:
                corr_matrix = corr_matrix.where(abs(corr_matrix) >= min_corr, 0)
                
            # 그림 크기 동적 조정 (열 개수에 비례)
            n_cols = len(corr_matrix.columns)
            figsize = (max(10, n_cols * 0.8), max(8, n_cols * 0.7))
            
            # 그림 생성
            plt.figure(figsize=figsize)
            
            # 폰트 속성 가져오기
            font_prop = self.get_font_prop('B')
            
            # 히트맵 그리기
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 상단 삼각형 마스킹
            heatmap = sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                       mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5})
            
            # 히트맵 주석(annotation) 폰트 설정
            for text in heatmap.texts:
                text.set_fontproperties(font_prop)
                
            # x축과 y축 레이블 폰트 설정
            plt.xticks(fontproperties=font_prop, fontsize=10)
            plt.yticks(fontproperties=font_prop, fontsize=10)
                       
            # 그래프 설정
            if title:
                plt.title(title, fontproperties=font_prop, fontsize=14)
            else:
                plt.title(f"Correlation Heatmap ({method})", fontproperties=font_prop, fontsize=14)
                
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                filename = self._get_filename(f"correlation_heatmap_{method}".replace(' ', '_'))
                
            # 그림 저장
            file_path = self._save_figure(filename)
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"상관관계 히트맵 생성 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_pie(self, column: str, top_n: int = 5, show_others: bool = True,
                title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        파이 차트 생성
        
        Args:
            column (str): 데이터 열 이름
            top_n (int): 표시할 상위 범주 수
            show_others (bool): 상위 N개 이외의 범주를 'Others'로 묶어서 표시할지 여부
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        if column not in self.df.columns:
            self.logger.error(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
            return ""
            
        # 결측치가 아닌 데이터만 선택
        data = self.df[column].dropna()
        
        if len(data) == 0:
            self.logger.warning(f"열 '{column}'에 유효한 데이터가 없습니다.")
            return ""
            
        try:
            # 값 빈도 계산
            value_counts = data.value_counts()
            
            # 상위 N개 값 선택
            if len(value_counts) > top_n and show_others:
                top_values = value_counts.head(top_n)
                others_sum = value_counts[top_n:].sum()
                
                # 'Others' 항목 추가
                if others_sum > 0:
                    top_values = pd.concat([top_values, pd.Series([others_sum], index=['Others'])])
            else:
                top_values = value_counts.head(top_n)
                
            # 그림 생성
            plt.figure(figsize=self.figure_size)
            
            # 파이 차트 그리기
            plt.pie(top_values, labels=top_values.index, autopct='%1.1f%%', 
                   startangle=90, shadow=False)
                   
            # 원형 모양 유지
            plt.axis('equal')
            
            # 그래프 설정
            if title:
                plt.title(title)
            else:
                plt.title(f"Distribution of {column}")
                
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                filename = self._get_filename(f"pie_{column}".replace(' ', '_'))
                
            # 그림 저장
            file_path = self._save_figure(filename)
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"파이 차트 생성 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_missing_data(self, top_n: int = 30, figsize: Optional[Tuple[int, int]] = None,
                         title: Optional[str] = None, filename: Optional[str] = None) -> str:
        """
        결측 데이터 시각화
        
        Args:
            top_n (int): 표시할 최대 열 수
            figsize (Tuple[int, int], optional): 그림 크기. None인 경우 자동 계산
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        try:
            # 결측치 비율 계산
            missing_data = self.df.isna().mean().sort_values(ascending=False) * 100
            
            # 결측치가 있는 열만 선택
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) == 0:
                self.logger.info("결측 데이터가 없습니다.")
                return ""
                
            # 표시할 열 제한
            if len(missing_data) > top_n:
                missing_data = missing_data.head(top_n)
                
            # 그림 크기 계산
            if figsize is None:
                height = max(6, len(missing_data) * 0.25)
                figsize = (10, height)
                
            # 그림 생성
            plt.figure(figsize=figsize)
            
            # 결측 데이터 시각화
            ax = missing_data.plot(kind='barh', color='red')
            
            # 값 레이블 추가
            for i, v in enumerate(missing_data):
                ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
                
            # 그래프 설정
            if title:
                plt.title(title)
            else:
                plt.title("Missing Data Percentage")
                
            plt.xlabel("Percentage (%)")
            plt.ylabel("Columns")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # 현재 그림 저장
            self.current_figure = plt.gcf()
            
            # 자동 파일명 생성
            if filename is None:
                filename = self._get_filename("missing_data")
                
            # 그림 저장
            file_path = self._save_figure(filename, tight_layout=False)  # 이미 tight_layout 적용됨
            plt.close()
            
            return file_path
        except Exception as e:
            self.logger.error(f"결측 데이터 시각화 중 오류 발생: {str(e)}")
            plt.close()
            return ""
            
    def plot_pairplot(self, columns: Optional[List[str]] = None, hue: Optional[str] = None,
                     max_cols: int = 5, title: Optional[str] = None, 
                     filename: Optional[str] = None) -> str:
        """
        페어플롯 생성 (여러 변수 간의 산점도 매트릭스)
        
        Args:
            columns (List[str], optional): 분석할 열 목록. None인 경우 모든 수치형 열 사용 (최대 max_cols개)
            hue (str, optional): 색상 구분에 사용할 열 이름
            max_cols (int): 최대 열 수
            title (str, optional): 그래프 제목
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        # 수치형 열만 선택
        if columns is None:
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            # Timedelta 열도 추가
            timedelta_cols = self.df.select_dtypes(include=['timedelta64[ns]']).columns.tolist()
            numeric_cols.extend(timedelta_cols)
                
            # 열 수 제한
            if len(numeric_cols) > max_cols:
                self.logger.info(f"수치형 열이 {len(numeric_cols)}개로 많아 상위 {max_cols}개만 사용합니다.")
                numeric_cols = numeric_cols[:max_cols]
        else:
            numeric_cols = [col for col in columns if col in self.df.columns]
            
        # 중복 열 제거
        numeric_cols = list(dict.fromkeys(numeric_cols))
        
        if not numeric_cols:
            self.logger.warning("분석할 수치형 열이 없습니다.")
            return ""
            
        # 최소 2개 이상의 열이 필요함
        if len(numeric_cols) < 2:
            self.logger.warning("페어플롯을 위해서는 최소 2개 이상의 수치형 열이 필요합니다.")
            
            # 단일 열이 있는 경우 히스토그램으로 대체
            if len(numeric_cols) == 1:
                self.logger.info(f"열이 1개뿐이므로 히스토그램으로 대체합니다: {numeric_cols[0]}")
                return self.plot_histogram(
                    column=numeric_cols[0],
                    title=f"Distribution of {numeric_cols[0]}",
                    filename=filename
                )
            return ""
            
        # hue 열 확인
        if hue is not None and hue not in self.df.columns:
            self.logger.warning(f"열 '{hue}'이 데이터프레임에 존재하지 않습니다. hue 없이 진행합니다.")
            hue = None
            
        # 필요한 열만 선택
        if hue:
            data_cols = numeric_cols + [hue]
        else:
            data_cols = numeric_cols
            
        # Timedelta 열 변환
        data = self.df[data_cols].copy()
        for col in numeric_cols:
            if pd.api.types.is_timedelta64_dtype(data[col]):
                data[col] = self._handle_column_with_timedelta(data[col])
                
        # 결측치 제거
        data = data.dropna()
        
        if len(data) == 0:
            self.logger.warning("유효한 데이터가 없습니다.")
            return ""
            
        try:
            # 출력 디렉토리 확인
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            # 자동 파일명 생성
            if filename is None:
                filename = self._get_filename("pairplot")
                
            # 파일 경로 구성
            file_path = os.path.join(self.output_dir, filename)
                
            # Seaborn pairplot 생성
            g = sns.pairplot(data, vars=numeric_cols, hue=hue, height=3, aspect=1)
            
            # 그래프 설정
            if title:
                g.fig.suptitle(title, y=1.02)
                
            # 현재 그림 저장
            self.current_figure = g.fig
            
            # 그림 직접 저장
            g.savefig(file_path, bbox_inches='tight')
            plt.close(g.fig)
            
            # 파일이 실제로 생성되었는지 확인
            if not os.path.exists(file_path):
                self.logger.error(f"파일 '{file_path}'이 생성되지 않았습니다.")
                # 다른 방식으로 저장 시도
                plt.figure(figsize=(10, 8))
                plt.title("Fallback Pairplot")
                plt.savefig(file_path)
                plt.close()
                if not os.path.exists(file_path):
                    return ""
                
            return file_path
        except Exception as e:
            self.logger.error(f"페어플롯 생성 중 오류 발생: {str(e)}")
            
            # 오류 발생 시 대체 이미지 생성
            try:
                if filename is None:
                    filename = self._get_filename("pairplot_fallback")
                file_path = os.path.join(self.output_dir, filename)
                
                # 간단한 산점도 행렬 만들기
                fig, axes = plt.subplots(len(numeric_cols), len(numeric_cols), figsize=(10, 10))
                fig.suptitle("Pairplot (Fallback)", size=16)
                
                # 축 정리
                for i in range(len(numeric_cols)):
                    for j in range(len(numeric_cols)):
                        if i != j:  # 다른 변수 사이의 산점도
                            try:
                                axes[i, j].scatter(data[numeric_cols[j]], data[numeric_cols[i]], alpha=0.5)
                            except:
                                pass
                        else:  # 동일 변수에 대한 히스토그램
                            try:
                                axes[i, i].hist(data[numeric_cols[i]], bins=20)
                            except:
                                pass
                        axes[i, j].set_xticks([])
                        axes[i, j].set_yticks([])
                        
                # 라벨 추가
                for i, col in enumerate(numeric_cols):
                    axes[-1, i].set_xlabel(col, rotation=45)
                    axes[i, 0].set_ylabel(col, rotation=45)
                    
                plt.tight_layout()
                plt.savefig(file_path)
                plt.close()
                
                if os.path.exists(file_path):
                    return file_path
            except Exception as e2:
                self.logger.error(f"대체 이미지 생성 중 오류 발생: {str(e2)}")
                
            plt.close()
            return ""
            
    def create_dashboard(self, columns: Optional[List[str]] = None, max_cols: int = 10,
                        include_corr: bool = True, filename: Optional[str] = None) -> str:
        """
        여러 시각화를 포함하는 대시보드 생성
        
        Args:
            columns (List[str], optional): 분석할 열 목록. None인 경우 주요 열 자동 선택
            max_cols (int): 최대 열 수
            include_corr (bool): 상관관계 히트맵 포함 여부
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로 목록을 담은 문자열
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""
            
        # 분석할 열 선택
        if columns is None:
            # 수치형 열
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            
            # Timedelta 열
            for col in self.df.select_dtypes(include=['timedelta64[ns]']).columns:
                numeric_cols.append(col)
                
            # 범주형 열 (문자열/범주형)
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # 날짜/시간 열
            datetime_cols = self.df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # 열 수 제한
            if len(numeric_cols) > max_cols:
                numeric_cols = numeric_cols[:max_cols]
                
            if len(categorical_cols) > max_cols:
                categorical_cols = categorical_cols[:max_cols]
                
            # 모든 선택된 열 합치기
            all_cols = numeric_cols + categorical_cols + datetime_cols
        else:
            all_cols = [col for col in columns if col in self.df.columns]
            
            # 열 유형별 분류
            numeric_cols = [col for col in all_cols if pd.api.types.is_numeric_dtype(self.df[col]) or 
                          pd.api.types.is_timedelta64_dtype(self.df[col])]
                          
            categorical_cols = [col for col in all_cols if pd.api.types.is_object_dtype(self.df[col]) or 
                              isinstance(self.df[col].dtype, pd.CategoricalDtype)]
                              
            datetime_cols = [col for col in all_cols if pd.api.types.is_datetime64_any_dtype(self.df[col])]
            
        if not all_cols:
            self.logger.warning("분석할 열이 없습니다.")
            return ""
            
        # 생성된 모든 시각화 파일 경로를 저장할 리스트
        visualization_files = []
        
        try:
            # 1. 결측 데이터 시각화
            missing_data_path = self.plot_missing_data(
                title="Missing Data in Dataset",
                filename="dashboard_missing_data.png"
            )
            if missing_data_path:
                visualization_files.append(missing_data_path)
                
            # 2. 수치형 열에 대한 히스토그램
            for col in numeric_cols:
                hist_path = self.plot_histogram(
                    column=col,
                    title=f"Distribution of {col}",
                    filename=f"dashboard_hist_{col.replace(' ', '_')}.png"
                )
                if hist_path:
                    visualization_files.append(hist_path)
                    
            # 3. 범주형 열에 대한 막대 그래프
            for col in categorical_cols:
                bar_path = self.plot_bar(
                    column=col,
                    horizontal=True,  # 가로 막대 그래프
                    title=f"Frequency of {col}",
                    filename=f"dashboard_bar_{col.replace(' ', '_')}.png"
                )
                if bar_path:
                    visualization_files.append(bar_path)
                    
                # 주요 범주형 열에 대해 파이 차트도 생성
                if len(self.df[col].dropna().unique()) <= 10:
                    pie_path = self.plot_pie(
                        column=col,
                        title=f"Distribution of {col}",
                        filename=f"dashboard_pie_{col.replace(' ', '_')}.png"
                    )
                    if pie_path:
                        visualization_files.append(pie_path)
                        
            # 4. 수치형 열 간의 상관관계 (선택적)
            if include_corr and len(numeric_cols) >= 2:
                corr_path = self.plot_correlation_heatmap(
                    columns=numeric_cols,
                    title="Correlation Between Numeric Variables",
                    filename="dashboard_correlation.png"
                )
                if corr_path:
                    visualization_files.append(corr_path)
                    
            # 5. 수치형-수치형 열 간의 산점도 (주요 2-3쌍만)
            if len(numeric_cols) >= 2:
                # 처음 2-3쌍의 수치형 열만 선택
                for i in range(min(2, len(numeric_cols)-1)):
                    scatter_path = self.plot_scatter(
                        x_column=numeric_cols[i],
                        y_column=numeric_cols[i+1],
                        title=f"{numeric_cols[i+1]} vs {numeric_cols[i]}",
                        filename=f"dashboard_scatter_{numeric_cols[i]}_{numeric_cols[i+1]}.png"
                    )
                    if scatter_path:
                        visualization_files.append(scatter_path)
                        
            # 6. 범주형-수치형 열 간의 박스플롯 (주요 2-3쌍만)
            if categorical_cols and numeric_cols:
                for i in range(min(3, len(categorical_cols))):
                    for j in range(min(2, len(numeric_cols))):
                        if len(self.df[categorical_cols[i]].dropna().unique()) <= 10:  # 범주가 10개 이하일 때만
                            box_path = self.plot_boxplot(
                                column=numeric_cols[j],
                                by=categorical_cols[i],
                                title=f"{numeric_cols[j]} by {categorical_cols[i]}",
                                filename=f"dashboard_box_{numeric_cols[j]}_by_{categorical_cols[i]}.png"
                            )
                            if box_path:
                                visualization_files.append(box_path)
                                
            # 대시보드 생성 결과 반환
            if visualization_files:
                result = f"Total of {len(visualization_files)} visualizations created:\n"
                for i, file_path in enumerate(visualization_files, 1):
                    result += f"{i}. {os.path.basename(file_path)}\n"
                return result
            else:
                return "No visualizations were created."
                
        except Exception as e:
            self.logger.error(f"대시보드 생성 중 오류 발생: {str(e)}")
            return f"Error creating dashboard: {str(e)}"
            
    def export_all_visualizations(self, columns: Optional[List[str]] = None, 
                                 prefix: str = "analysis", format: str = "png") -> str:
        """
        모든 주요 시각화 유형을 생성하고 내보내기
        
        Args:
            columns (List[str], optional): 분석할 열 목록. None인 경우 주요 열 자동 선택
            prefix (str): 파일명 접두사
            format (str): 저장 형식 ('png', 'pdf', 'svg', 'jpg')
            
        Returns:
            str: 저장된 파일 경로 목록을 담은 문자열
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return ""

        # 출력 디렉토리가 존재하는지 확인하고 없으면 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 테마 설정
        self.set_theme('whitegrid')
        
        # 그림 크기 설정
        self.set_figure_size(12, 8, 100)
        
        # 기본 테스트 시각화 생성 (최소 1개의 파일 보장)
        if self.df is not None and len(self.df.columns) > 0:
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                self.plot_histogram(numeric_cols[0], 
                                filename=f"{prefix}_test_histogram.{format}")
        
        # 대시보드 생성
        result = self.create_dashboard(
            columns=columns,
            include_corr=True,
            filename=f"{prefix}_dashboard.{format}"
        )
        
        return result 

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