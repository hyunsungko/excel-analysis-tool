import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging
import tempfile
import shutil
from datetime import datetime, timedelta

# 로깅 비활성화
logging.disable(logging.CRITICAL)

# 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.visualization.visualization_engine import VisualizationEngine

class TestVisualizationEngine(unittest.TestCase):
    """VisualizationEngine 클래스에 대한 단위 테스트"""
    
    def setUp(self):
        """테스트 데이터 및 임시 출력 디렉토리 설정"""
        # 테스트용 데이터프레임 생성
        self.df = pd.DataFrame({
            'id': range(1, 11),
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'D'],
            'value1': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'value2': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            'date': pd.date_range(start='2023-01-01', periods=10),
            'duration': [timedelta(minutes=m) for m in range(10, 110, 10)],
            'text': ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr', 'stu', 'vwx', 'yz1', '234']
        })
        
        # NaN 값 추가
        self.df.loc[0, 'value1'] = np.nan
        self.df.loc[3, 'value2'] = np.nan
        self.df.loc[5, 'category'] = np.nan
        
        # 임시 출력 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        
        # VisualizationEngine 인스턴스 생성 및 데이터프레임 설정
        self.engine = VisualizationEngine(output_dir=self.temp_dir)
        self.engine.set_dataframe(self.df)
        
    def tearDown(self):
        """테스트 후 임시 디렉토리 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.temp_dir)
        
    def test_set_dataframe(self):
        """set_dataframe 메서드 테스트"""
        engine = VisualizationEngine(output_dir=self.temp_dir)
        engine.set_dataframe(self.df)
        
        # 데이터프레임이 올바르게 설정되었는지 확인
        self.assertIsNotNone(engine.get_dataframe())
        self.assertEqual(engine.get_dataframe().shape, self.df.shape)
        
    def test_set_theme(self):
        """set_theme 메서드 테스트"""
        # 유효한 테마 설정
        self.engine.set_theme('dark')
        self.assertEqual(self.engine.theme, 'dark')
        
        # 유효하지 않은 테마 설정 시도
        self.engine.set_theme('nonexistent_theme')
        self.assertEqual(self.engine.theme, 'default')  # 기본 테마로 설정됨
        
    def test_set_figure_size(self):
        """set_figure_size 메서드 테스트"""
        width, height, dpi = 15, 10, 120
        self.engine.set_figure_size(width, height, dpi)
        
        # 설정이 올바르게 적용되었는지 확인
        self.assertEqual(self.engine.figure_size, (width, height))
        self.assertEqual(self.engine.dpi, dpi)
        
    def test_plot_histogram(self):
        """plot_histogram 메서드 테스트"""
        # 수치형 열 히스토그램
        file_path = self.engine.plot_histogram('value1')
        self.assertTrue(os.path.exists(file_path))
        self.assertTrue(file_path.endswith('.png'))
        
        # 타임델타 열 히스토그램
        file_path = self.engine.plot_histogram('duration')
        self.assertTrue(os.path.exists(file_path))
        
        # 범주형 열 히스토그램 (막대 그래프로 대체됨)
        file_path = self.engine.plot_histogram('category')
        self.assertTrue(os.path.exists(file_path))
        
        # 존재하지 않는 열
        file_path = self.engine.plot_histogram('nonexistent')
        self.assertEqual(file_path, "")
        
    def test_plot_bar(self):
        """plot_bar 메서드 테스트"""
        # 범주형 열 막대 그래프
        file_path = self.engine.plot_bar('category')
        self.assertTrue(os.path.exists(file_path))
        
        # 가로 막대 그래프
        file_path = self.engine.plot_bar('category', horizontal=True)
        self.assertTrue(os.path.exists(file_path))
        
        # 수치형 열 막대 그래프 (자동으로 빈도수 계산)
        file_path = self.engine.plot_bar('value1')
        self.assertTrue(os.path.exists(file_path))
        
    def test_plot_scatter(self):
        """plot_scatter 메서드 테스트"""
        # 기본 산점도
        file_path = self.engine.plot_scatter('value1', 'value2')
        self.assertTrue(os.path.exists(file_path))
        
        # 색상 구분 열 추가
        file_path = self.engine.plot_scatter('value1', 'value2', hue_column='category')
        self.assertTrue(os.path.exists(file_path))
        
        # 타임델타 열 포함
        file_path = self.engine.plot_scatter('duration', 'value1')
        self.assertTrue(os.path.exists(file_path))
        
    def test_plot_boxplot(self):
        """plot_boxplot 메서드 테스트"""
        # 그룹 없는 박스플롯
        file_path = self.engine.plot_boxplot('value1')
        self.assertTrue(os.path.exists(file_path))
        
        # 그룹별 박스플롯
        file_path = self.engine.plot_boxplot('value1', by='category')
        self.assertTrue(os.path.exists(file_path))
        
        # 가로 박스플롯
        file_path = self.engine.plot_boxplot('value1', horizontal=True)
        self.assertTrue(os.path.exists(file_path))
        
        # 타임델타 열 박스플롯
        file_path = self.engine.plot_boxplot('duration')
        self.assertTrue(os.path.exists(file_path))
        
    def test_plot_line(self):
        """plot_line 메서드 테스트"""
        # 기본 선 그래프
        file_path = self.engine.plot_line('id', 'value1')
        self.assertTrue(os.path.exists(file_path))
        
        # 날짜 기반 선 그래프
        file_path = self.engine.plot_line('date', 'value1')
        self.assertTrue(os.path.exists(file_path))
        
        # 그룹별 선 그래프
        file_path = self.engine.plot_line('id', 'value1', group_column='category')
        self.assertTrue(os.path.exists(file_path))
        
    def test_plot_correlation_heatmap(self):
        """plot_correlation_heatmap 메서드 테스트"""
        # 기본 상관관계 히트맵
        file_path = self.engine.plot_correlation_heatmap()
        self.assertTrue(os.path.exists(file_path))
        
        # 특정 열만 포함
        file_path = self.engine.plot_correlation_heatmap(columns=['value1', 'value2'])
        self.assertTrue(os.path.exists(file_path))
        
        # 최소 상관계수 임계값 설정
        file_path = self.engine.plot_correlation_heatmap(min_corr=0.5)
        self.assertTrue(os.path.exists(file_path))
        
        # 다른 상관관계 계산 방법
        file_path = self.engine.plot_correlation_heatmap(method='spearman')
        self.assertTrue(os.path.exists(file_path))
        
    def test_plot_pie(self):
        """plot_pie 메서드 테스트"""
        # 범주형 열 파이 차트
        file_path = self.engine.plot_pie('category')
        self.assertTrue(os.path.exists(file_path))
        
        # 상위 N개만 표시
        file_path = self.engine.plot_pie('category', top_n=2, show_others=True)
        self.assertTrue(os.path.exists(file_path))
        
    def test_plot_missing_data(self):
        """plot_missing_data 메서드 테스트"""
        # 결측 데이터 시각화
        file_path = self.engine.plot_missing_data()
        self.assertTrue(os.path.exists(file_path))
        
        # 결측치가 없는 데이터에 대한 테스트
        df_no_missing = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        engine = VisualizationEngine(output_dir=self.temp_dir)
        engine.set_dataframe(df_no_missing)
        file_path = engine.plot_missing_data()
        self.assertEqual(file_path, "")  # 결측치가 없으면 파일 생성 안 함
        
    def test_plot_pairplot(self):
        """plot_pairplot 메서드 테스트"""
        # 기본 페어플롯
        file_path = self.engine.plot_pairplot()
        self.assertNotEqual(file_path, "")
        
        # 특정 열만 포함
        file_path = self.engine.plot_pairplot(columns=['value1', 'value2'])
        self.assertNotEqual(file_path, "")
        
        # 색상 구분 열 추가
        file_path = self.engine.plot_pairplot(hue='category')
        self.assertNotEqual(file_path, "")
        
    def test_create_dashboard(self):
        """create_dashboard 메서드 테스트"""
        # 대시보드 생성
        result = self.engine.create_dashboard()
        self.assertIn("visualizations created", result)
        
        # 생성된 파일 수 확인
        file_count = len([f for f in os.listdir(self.temp_dir) if f.startswith("dashboard_")])
        self.assertGreater(file_count, 0)
        
        # 특정 열만 포함
        result = self.engine.create_dashboard(columns=['value1', 'category'])
        self.assertIn("visualizations created", result)
        
    def test_export_all_visualizations(self):
        """export_all_visualizations 메서드 테스트"""
        # 모든 시각화 내보내기
        result = self.engine.export_all_visualizations(prefix="test_export")
        self.assertIn("visualizations created", result)
        
        # 생성된 파일 수 확인
        file_count = len([f for f in os.listdir(self.temp_dir) if f.startswith("test_export_")])
        self.assertGreater(file_count, 0)
        
if __name__ == '__main__':
    unittest.main() 