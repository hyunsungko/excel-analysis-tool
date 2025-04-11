import unittest
import pandas as pd
import numpy as np
import sys
import os
import logging

# 로깅 비활성화
logging.disable(logging.CRITICAL)

# 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.core.analysis_engine import AnalysisEngine

class TestAnalysisEngine(unittest.TestCase):
    """AnalysisEngine 클래스에 대한 단위 테스트"""
    
    def setUp(self):
        """테스트 데이터 설정"""
        # 테스트용 데이터프레임 생성
        self.df = pd.DataFrame({
            'id': range(1, 11),
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'D'],
            'value1': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'value2': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            'date': pd.date_range(start='2023-01-01', periods=10),
            'text': ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr', 'stu', 'vwx', 'yz1', '234']
        })
        
        # NaN 값 추가
        self.df.loc[0, 'value1'] = np.nan
        self.df.loc[3, 'value2'] = np.nan
        self.df.loc[5, 'category'] = np.nan
        
        # AnalysisEngine 인스턴스 생성 및 데이터프레임 설정
        self.engine = AnalysisEngine()
        self.engine.set_dataframe(self.df)
        
    def test_set_dataframe(self):
        """set_dataframe 메서드 테스트"""
        engine = AnalysisEngine()
        engine.set_dataframe(self.df)
        
        # 데이터프레임이 올바르게 설정되었는지 확인
        self.assertIsNotNone(engine.get_dataframe())
        self.assertEqual(engine.get_dataframe().shape, self.df.shape)
        
    def test_get_basic_stats(self):
        """get_basic_stats 메서드 테스트"""
        # 모든 수치형 열에 대한 기본 통계 계산
        stats = self.engine.get_basic_stats()
        
        # 통계 결과 검증
        self.assertIn('value1', stats)
        self.assertIn('value2', stats)
        
        # 통계 데이터 검증
        self.assertEqual(stats['value1']['count'], 9)  # 1개의 NaN 값
        self.assertEqual(stats['value1']['missing'], 1)
        self.assertEqual(stats['value2']['missing'], 1)
        
        # 특정 통계값 확인
        self.assertAlmostEqual(stats['value1']['mean'], self.df['value1'].mean())
        self.assertAlmostEqual(stats['value2']['median'], self.df['value2'].median())
        
        # 특정 열에 대한 통계 계산
        stats_value1 = self.engine.get_basic_stats(columns=['value1'])
        self.assertIn('value1', stats_value1)
        self.assertNotIn('value2', stats_value1)
        
    def test_get_categorical_stats(self):
        """get_categorical_stats 메서드 테스트"""
        # 범주형 열에 대한 통계 계산
        cat_stats = self.engine.get_categorical_stats()
        
        # 통계 결과 검증
        self.assertIn('category', cat_stats)
        self.assertIn('text', cat_stats)
        
        # 범주형 데이터 검증
        self.assertEqual(cat_stats['category']['unique_values'], 4)  # A, B, C, D
        self.assertEqual(cat_stats['category']['missing'], 1)  # 1개의 NaN 값
        
        # 상위 값 확인
        self.assertIn('A', str(cat_stats['category']['top_values'].keys()))
        
        # 특정 열에 대한 통계 계산
        cat_stats_cat = self.engine.get_categorical_stats(columns=['category'])
        self.assertIn('category', cat_stats_cat)
        self.assertNotIn('text', cat_stats_cat)
        
    def test_get_correlation_matrix(self):
        """get_correlation_matrix 메서드 테스트"""
        # 상관관계 계산
        corr_matrix = self.engine.get_correlation_matrix()
        
        # 상관관계 매트릭스 검증
        self.assertIn('value1', corr_matrix.columns)
        self.assertIn('value2', corr_matrix.columns)
        
        # value1과 value2의 상관관계 확인
        corr_value = corr_matrix.loc['value1', 'value2']
        expected_corr = self.df[['value1', 'value2']].corr().loc['value1', 'value2']
        self.assertAlmostEqual(corr_value, expected_corr)
        
        # 특정 방법으로 상관관계 계산
        spearman_corr = self.engine.get_correlation_matrix(method='spearman')
        expected_spearman = self.df[['value1', 'value2']].corr(method='spearman').loc['value1', 'value2']
        self.assertAlmostEqual(spearman_corr.loc['value1', 'value2'], expected_spearman)
        
        # 최소 상관계수 임계값 적용
        high_corr = self.engine.get_correlation_matrix(min_corr=0.9)
        # 상관관계가 0.9보다 작으면 0으로 설정되어야 함
        if abs(expected_corr) < 0.9:
            self.assertEqual(high_corr.loc['value1', 'value2'], 0)
        
    def test_get_strong_correlations(self):
        """get_strong_correlations 메서드 테스트"""
        # 강한 상관관계 추출 (기본 임계값: 0.7)
        strong_corr = self.engine.get_strong_correlations()
        
        # value1과 value2의 상관관계 확인
        expected_corr = self.df[['value1', 'value2']].corr().loc['value1', 'value2']
        
        # 상관관계 절대값이 0.7 이상인 경우만 결과에 포함
        if abs(expected_corr) >= 0.7:
            self.assertGreaterEqual(len(strong_corr), 1)
            # 결과 형식 확인 (변수1, 변수2, 상관계수)
            self.assertEqual(len(strong_corr[0]), 3)
        else:
            # 낮은 임계값으로 테스트
            lower_threshold = abs(expected_corr) - 0.1
            low_strong_corr = self.engine.get_strong_correlations(threshold=lower_threshold)
            self.assertGreaterEqual(len(low_strong_corr), 1)
        
    def test_group_analysis(self):
        """group_analysis 메서드 테스트"""
        # 범주별 그룹 분석
        group_result = self.engine.group_analysis(group_by='category')
        
        # 결과 검증
        self.assertFalse(group_result.empty)
        self.assertIn('value1', group_result.columns.levels[0])
        self.assertIn('value2', group_result.columns.levels[0])
        
        # 집계 함수 검증
        self.assertIn('mean', group_result.columns.levels[1])
        self.assertIn('median', group_result.columns.levels[1])
        
        # 특정 열과 집계 함수로 그룹 분석
        custom_agg = {'value1': ['mean', 'max'], 'value2': ['min', 'sum']}
        custom_result = self.engine.group_analysis(
            group_by='category',
            agg_columns=['value1', 'value2'],
            agg_funcs=custom_agg
        )
        
        # 커스텀 집계 함수 검증
        self.assertIn('mean', custom_result['value1'].columns)
        self.assertIn('max', custom_result['value1'].columns)
        self.assertIn('min', custom_result['value2'].columns)
        self.assertIn('sum', custom_result['value2'].columns)
        
    def test_time_series_analysis(self):
        """time_series_analysis 메서드 테스트"""
        # 일별 시계열 분석
        ts_result = self.engine.time_series_analysis(
            date_column='date',
            value_column='value1',
            freq='D',
            agg_func='mean'
        )
        
        # 결과 검증
        self.assertFalse(ts_result.empty)
        self.assertEqual(len(ts_result), len(self.df.dropna(subset=['value1'])))
        
        # 월별 집계
        monthly_result = self.engine.time_series_analysis(
            date_column='date',
            value_column='value1',
            freq='M',
            agg_func='sum'
        )
        
        # 월별 결과 검증 (1개월 데이터만 있으므로 결과는 1행)
        self.assertEqual(len(monthly_result), 1)
        
    def test_get_all_results(self):
        """get_all_results 메서드 테스트"""
        # 분석 수행 (데이터 생성)
        self.engine.get_basic_stats()
        self.engine.get_categorical_stats()
        self.engine.get_correlation_matrix()
        
        # 모든 결과 가져오기
        all_results = self.engine.get_all_results()
        
        # 결과 검증
        self.assertIn('basic_stats', all_results)
        self.assertIn('categorical_stats', all_results)
        self.assertIn('correlation', all_results)
        
if __name__ == '__main__':
    unittest.main() 