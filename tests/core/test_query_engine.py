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

from src.core.query_engine import QueryEngine

class TestQueryEngine(unittest.TestCase):
    """QueryEngine 클래스에 대한 단위 테스트"""
    
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
        
        # QueryEngine 인스턴스 생성 및 데이터프레임 설정
        self.engine = QueryEngine()
        self.engine.set_dataframe(self.df)
        
    def test_set_dataframe(self):
        """set_dataframe 메서드 테스트"""
        engine = QueryEngine()
        engine.set_dataframe(self.df)
        
        # 데이터프레임이 올바르게 설정되었는지 확인
        self.assertIsNotNone(engine.get_dataframe())
        self.assertEqual(engine.get_dataframe().shape, self.df.shape)
        self.assertEqual(engine.get_current_result().shape, self.df.shape)
        
    def test_query(self):
        """query 메서드 테스트"""
        # 쿼리 실행
        result = self.engine.query("value1 > 20")
        
        # 결과 검증 - NaN 값을 제외하고 value1 > 20인 행 수
        expected_count = len(self.df[self.df['value1'] > 20])
        self.assertEqual(len(result), expected_count)
        
        # 모든 값이 조건에 맞는지 확인 (NaN은 제외)
        non_nan_values = result['value1'].dropna()
        self.assertTrue(all(non_nan_values > 20))
        
        # 쿼리 히스토리 확인
        history = self.engine.get_query_history()
        self.assertIn("query: value1 > 20", history)
        
    def test_select_columns(self):
        """select_columns 메서드 테스트"""
        # 열 선택
        result = self.engine.select_columns(['id', 'category', 'value1'])
        
        # 결과 검증
        self.assertEqual(result.shape[1], 3)  # 3개 열 선택
        self.assertListEqual(list(result.columns), ['id', 'category', 'value1'])
        
        # 존재하지 않는 열 선택 시도
        result = self.engine.select_columns(['id', 'nonexistent'])
        self.assertEqual(result.shape[1], 1)  # 존재하는 열만 선택됨
        
    def test_filter_by_value(self):
        """filter_by_value 메서드 테스트"""
        # 값 포함 필터링
        result = self.engine.filter_by_value('category', ['A', 'B'])
        
        # 결과 검증
        self.assertEqual(len(result), 6)  # A 또는 B 카테고리 행은 6개
        self.assertTrue(all(result['category'].isin(['A', 'B'])))
        
        # 값 제외 필터링
        self.engine.reset_query()  # 쿼리 상태 초기화
        result = self.engine.filter_by_value('category', ['A', 'B'], include=False)
        
        # 결과 검증
        self.assertEqual(len(result), 4)  # A, B가 아닌 카테고리 행은 4개 (C, D, NaN 포함)
        for val in result['category']:
            if not pd.isna(val):  # NaN이 아닌 경우에만 검사
                self.assertNotIn(val, ['A', 'B'])
        
    def test_filter_by_range(self):
        """filter_by_range 메서드 테스트"""
        # 범위 필터링 (양쪽 포함)
        result = self.engine.filter_by_range('value1', 20, 40, inclusive='both')
        
        # 결과 검증
        self.assertEqual(len(result), 5)  # 20 <= value1 <= 40인 행은 5개
        self.assertTrue(all((result['value1'] >= 20) & (result['value1'] <= 40)))
        
        # 범위 필터링 (왼쪽만 포함)
        self.engine.reset_query()
        result = self.engine.filter_by_range('value1', 20, 40, inclusive='left')
        
        # 결과 검증
        self.assertEqual(len(result), 4)  # 20 <= value1 < 40인 행은 4개
        self.assertTrue(all((result['value1'] >= 20) & (result['value1'] < 40)))
        
    def test_sort_values(self):
        """sort_values 메서드 테스트"""
        # 오름차순 정렬
        result = self.engine.sort_values(['value1'])
        
        # 결과 검증
        self.assertTrue(result['value1'].equals(result['value1'].sort_values(ignore_index=False)))
        
        # 내림차순 정렬
        self.engine.reset_query()
        result = self.engine.sort_values(['value1'], ascending=False)
        
        # 결과 검증
        self.assertTrue(result['value1'].equals(result['value1'].sort_values(ascending=False, ignore_index=False)))
        
        # 여러 열로 정렬
        self.engine.reset_query()
        result = self.engine.sort_values(['category', 'value1'], ascending=[True, False])
        
        # 쿼리 히스토리 확인
        history = self.engine.get_query_history()
        self.assertTrue(any("sort_values" in item for item in history))
        
    def test_group_and_aggregate(self):
        """group_and_aggregate 메서드 테스트"""
        # 그룹별 집계
        agg_dict = {'value1': ['mean', 'max'], 'value2': ['min', 'count']}
        result = self.engine.group_and_aggregate('category', agg_dict)
        
        # 결과 검증
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.index.name, 'category')
        self.assertIn('value1', result.columns.levels[0])
        self.assertIn('mean', result.columns.levels[1])
        
        # 존재하지 않는 열로 그룹별 집계 시도
        result_empty = self.engine.group_and_aggregate('nonexistent', agg_dict)
        self.assertTrue(result_empty.equals(self.engine.get_current_result()))  # 오류 발생 시 현재 결과 유지
        
    def test_pivot_table(self):
        """pivot_table 메서드 테스트"""
        # 피벗 테이블 생성
        result = self.engine.pivot_table(
            index='category',
            columns='id',
            values='value1',
            aggfunc='mean'
        )
        
        # 결과 검증
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result.index.name, 'category')
        self.assertEqual(result.columns.name, 'id')
        
        # 다양한 집계 함수 사용
        self.engine.reset_query()
        result = self.engine.pivot_table(
            index='category',
            values=['value1', 'value2'],
            aggfunc={'value1': 'mean', 'value2': 'sum'}
        )
        
        # 결과 검증
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('value1', result.columns)
        self.assertIn('value2', result.columns)
        
    def test_search_text(self):
        """search_text 메서드 테스트"""
        # 정규식으로 텍스트 검색
        result = self.engine.search_text('text', '^[a-d]')
        
        # 결과 검증
        self.assertEqual(len(result), 2)  # 'abc', 'def'만 일치
        self.assertTrue(all(result['text'].str.match('^[a-d]')))
        
        # 일반 문자열 검색
        self.engine.reset_query()
        result = self.engine.search_text('text', 'abc', regex=False)
        
        # 결과 검증
        self.assertEqual(len(result), 1)  # 'abc'만 일치
        
        # 대소문자 구분 검색
        self.engine.reset_query()
        result = self.engine.search_text('text', 'ABC', case=True, regex=False)
        
        # 결과 검증
        self.assertEqual(len(result), 0)  # 일치하는 항목 없음
        
    def test_add_calculated_column(self):
        """add_calculated_column 메서드 테스트"""
        # 계산 열 추가
        result = self.engine.add_calculated_column('total', 'value1 + value2')
        
        # 결과 검증
        self.assertIn('total', result.columns)
        for i, row in result.iterrows():
            if pd.isna(row['value1']) or pd.isna(row['value2']):
                self.assertTrue(pd.isna(row['total']))
            else:
                self.assertEqual(row['total'], row['value1'] + row['value2'])
                
        # 복잡한 계산식 사용
        self.engine.reset_query()
        result = self.engine.add_calculated_column('ratio', 'value1 / value2 * 100')
        
        # 결과 검증
        self.assertIn('ratio', result.columns)
        
        # numpy 함수 사용
        self.engine.reset_query()
        result = self.engine.add_calculated_column('log_value', 'np.log(value1)')
        
        # 결과 검증
        self.assertIn('log_value', result.columns)
        
    def test_reset_query(self):
        """reset_query 메서드 테스트"""
        # 쿼리 수행 (필터링)
        self.engine.query("value1 > 30")
        filtered_result = self.engine.get_current_result()
        self.assertLess(len(filtered_result), len(self.df))
        
        # 쿼리 상태 초기화
        self.engine.reset_query()
        reset_result = self.engine.get_current_result()
        
        # 초기화 결과 검증
        self.assertEqual(len(reset_result), len(self.df))
        
    def test_get_query_history(self):
        """get_query_history 메서드 테스트"""
        # 여러 쿼리 수행
        self.engine.query("value1 > 30")
        self.engine.select_columns(['id', 'value1', 'value2'])
        self.engine.sort_values(['value1'], ascending=False)
        
        # 쿼리 히스토리 확인
        history = self.engine.get_query_history()
        self.assertEqual(len(history), 3)
        self.assertIn("query", history[0])
        self.assertIn("select_columns", history[1])
        self.assertIn("sort_values", history[2])
        
if __name__ == '__main__':
    unittest.main() 