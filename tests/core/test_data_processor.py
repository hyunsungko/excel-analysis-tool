#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DataProcessor 클래스에 대한 단위 테스트
"""

import os
import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import logging

# 테스트 중 로깅 비활성화
logging.disable(logging.CRITICAL)

# 모듈 경로 추가
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.core.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """
    DataProcessor 클래스 테스트
    """
    
    def setUp(self):
        """
        테스트를 위한 환경 설정 및 샘플 데이터 생성
        """
        # 샘플 데이터프레임 생성
        self.sample_df = pd.DataFrame({
            'id': range(1, 11),
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack'],
            'age': [25, np.nan, 35, 40, 45, 30, np.nan, 22, 38, 50],
            'salary': [50000, 60000, 70000, 80000, 90000, 55000, 65000, 45000, 85000, 95000],
            'department': ['HR', 'IT', np.nan, 'IT', 'HR', 'Finance', 'IT', 'HR', 'Finance', np.nan],
            'experience': [3, 5, 8, 10, 15, 4, 6, 2, 12, 20],
            'score': [75, 82, 90, 65, 95, 70, 88, 60, 85, 78]
        })
        
        # 데이터 프로세서 인스턴스 생성
        self.processor = DataProcessor(self.sample_df)
    
    def test_set_dataframe(self):
        """
        set_dataframe 메서드 테스트
        """
        # 새로운 데이터프레임 생성
        new_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        
        # 데이터프레임 설정
        self.processor.set_dataframe(new_df)
        
        # 설정된 데이터프레임 확인
        pd.testing.assert_frame_equal(self.processor.df, new_df)
        pd.testing.assert_frame_equal(self.processor.original_df, new_df)
    
    def test_add_preprocessing_step(self):
        """
        add_preprocessing_step 메서드 테스트
        """
        # 전처리 함수 정의
        def drop_nulls(df, columns=None):
            return df.dropna(subset=columns if columns else None)
        
        # 전처리 단계 추가
        self.processor.add_preprocessing_step('drop_nulls', drop_nulls, columns=['age', 'department'])
        
        # 추가된 전처리 단계 확인
        self.assertEqual(len(self.processor.preprocessing_steps), 1)
        self.assertEqual(self.processor.preprocessing_steps[0]['name'], 'drop_nulls')
        self.assertEqual(self.processor.preprocessing_steps[0]['function'], drop_nulls)
        self.assertEqual(self.processor.preprocessing_steps[0]['kwargs'], {'columns': ['age', 'department']})
    
    def test_apply_preprocessing(self):
        """
        apply_preprocessing 메서드 테스트
        """
        # 전처리 함수 정의
        def drop_nulls(df, columns=None):
            return df.dropna(subset=columns if columns else None)
            
        def multiply_column(df, column, factor):
            df = df.copy()
            df[column] = df[column] * factor
            return df
        
        # 전처리 단계 추가
        self.processor.add_preprocessing_step('drop_nulls', drop_nulls, columns=['age'])
        self.processor.add_preprocessing_step('multiply_salary', multiply_column, column='salary', factor=1.1)
        
        # 전처리 적용
        processed_df = self.processor.apply_preprocessing()
        
        # 적용 결과 확인
        # 1. age가 NaN인 행이 제거되었는지 확인
        self.assertEqual(len(processed_df), 8)  # 10 - 2(age가 NaN인 행)
        
        # 2. salary가 1.1배 증가했는지 확인
        for i, row in processed_df.iterrows():
            original_salary = self.sample_df.loc[i, 'salary']
            self.assertAlmostEqual(row['salary'], original_salary * 1.1)
    
    def test_handle_missing_values_drop_rows(self):
        """
        handle_missing_values 메서드 테스트 - drop_rows 전략
        """
        # 결측치 처리 - 행 삭제
        result_df = self.processor.handle_missing_values(
            self.sample_df,
            strategy='drop_rows',
            columns=['age', 'department'],
            threshold=0.0  # 모든 결측치 처리
        )
        
        # 결과 확인
        self.assertEqual(len(result_df), 6)  # 10 - 4(age 또는 department가 NaN인 행)
        
        # 결측치가 없는지 확인
        self.assertEqual(result_df['age'].isnull().sum(), 0)
        self.assertEqual(result_df['department'].isnull().sum(), 0)
    
    def test_handle_missing_values_drop_columns(self):
        """
        handle_missing_values 메서드 테스트 - drop_columns 전략
        """
        # 결측치 처리 - 열 삭제
        result_df = self.processor.handle_missing_values(
            self.sample_df,
            strategy='drop_columns',
            threshold=0.1  # 10% 이상의 결측치가 있는 열만 처리
        )
        
        # 결과 확인
        expected_columns = ['id', 'name', 'salary', 'experience', 'score']
        self.assertListEqual(list(result_df.columns), expected_columns)
    
    def test_handle_missing_values_fill_values(self):
        """
        handle_missing_values 메서드 테스트 - fill_values 전략
        """
        # 결측치 처리 - 값 채우기
        result_df = self.processor.handle_missing_values(
            self.sample_df,
            strategy='fill_values',
            columns=['age', 'department'],
            threshold=0.0,
            fill_values={'age': 30, 'department': 'Unknown'}
        )
        
        # 결과 확인
        self.assertEqual(result_df['age'].isnull().sum(), 0)
        self.assertEqual(result_df['department'].isnull().sum(), 0)
        
        # 채워진 값 확인
        self.assertEqual(result_df.loc[1, 'age'], 30)
        self.assertEqual(result_df.loc[6, 'age'], 30)
        self.assertEqual(result_df.loc[2, 'department'], 'Unknown')
        self.assertEqual(result_df.loc[9, 'department'], 'Unknown')
    
    def test_convert_data_types(self):
        """
        convert_data_types 메서드 테스트
        """
        # 테스트 데이터프레임
        test_df = pd.DataFrame({
            'str_col': ['1', '2', '3', '4', '5'],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'date_str': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
            'bool_str': ['true', 'false', 'yes', 'no', '1']
        })
        
        # 데이터 타입 변환
        result_df = self.processor.convert_data_types(
            test_df,
            type_conversions={
                'str_col': 'int',
                'float_col': 'str',
                'date_str': 'datetime',
                'bool_str': 'bool'
            }
        )
        
        # 결과 확인
        self.assertEqual(result_df['str_col'].dtype, 'Int64')
        self.assertEqual(result_df['float_col'].dtype, 'object')
        self.assertTrue(pd.api.types.is_datetime64_dtype(result_df['date_str']))
        
        # 변환된 값 확인
        self.assertEqual(result_df.loc[0, 'str_col'], 1)
        self.assertEqual(result_df.loc[0, 'float_col'], '1.1')
        self.assertEqual(result_df.loc[0, 'date_str'].year, 2020)
        self.assertEqual(result_df.loc[0, 'date_str'].month, 1)
        self.assertEqual(result_df.loc[0, 'date_str'].day, 1)
        self.assertEqual(result_df.loc[0, 'bool_str'], True)
        self.assertEqual(result_df.loc[1, 'bool_str'], False)
    
    def test_handle_outliers(self):
        """
        handle_outliers 메서드 테스트
        """
        # 테스트 데이터프레임 (outlier 포함)
        test_df = pd.DataFrame({
            'values': [10, 15, 12, 14, 100, 13, 11, 9, 150, 14]
        })
        
        # IQR 방법으로 이상치 제거
        result_df = self.processor.handle_outliers(
            test_df,
            columns=['values'],
            method='iqr',
            threshold=1.5,
            strategy='remove'
        )
        
        # 결과 확인 (이상치 100, 150 제거)
        self.assertEqual(len(result_df), 8)
        self.assertTrue(100 not in result_df['values'].values)
        self.assertTrue(150 not in result_df['values'].values)
        
        # IQR 방법으로 이상치 클리핑
        result_df = self.processor.handle_outliers(
            test_df,
            columns=['values'],
            method='iqr',
            threshold=1.5,
            strategy='clip'
        )
        
        # 결과 확인 (이상치가 상한값으로 대체)
        self.assertEqual(len(result_df), 10)  # 행 수 유지
        
        # 상한값 계산 (Q3 + 1.5*IQR)
        q1 = test_df['values'].quantile(0.25)
        q3 = test_df['values'].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        
        # 클리핑된 값 확인
        self.assertEqual(result_df.loc[4, 'values'], upper_bound)
        self.assertEqual(result_df.loc[8, 'values'], upper_bound)
    
    def test_normalize_data(self):
        """
        normalize_data 메서드 테스트
        """
        # 테스트 데이터프레임
        test_df = pd.DataFrame({
            'values': [10, 20, 30, 40, 50]
        })
        
        # Min-Max 정규화
        result_df = self.processor.normalize_data(
            test_df,
            columns=['values'],
            method='minmax'
        )
        
        # 결과 확인 (0~1 범위로 정규화)
        self.assertEqual(result_df.loc[0, 'values'], 0.0)
        self.assertEqual(result_df.loc[4, 'values'], 1.0)
        
        # Z-Score 정규화
        result_df = self.processor.normalize_data(
            test_df,
            columns=['values'],
            method='zscore'
        )
        
        # 결과 확인 (평균=0, 표준편차=약 1.118)
        # sklearn.preprocessing.StandardScaler와 pandas의 DataFrame 사용시 이 값이 나옴
        self.assertAlmostEqual(result_df['values'].mean(), 0, places=10)
        # 표본표준편차(n-1로 나눔)를 사용하기 때문에 정확히 1.118... 값이 나옴
        self.assertAlmostEqual(result_df['values'].std(), 1.1180339887498947, places=10)
    
    def test_add_derived_column(self):
        """
        add_derived_column 메서드 테스트
        """
        # 파생 열 추가 함수
        def calculate_bonus(df):
            return df['salary'] * 0.1
            
        # 파생 열 추가
        result_df = self.processor.add_derived_column(
            self.sample_df,
            new_column='bonus',
            formula=calculate_bonus
        )
        
        # 결과 확인
        self.assertIn('bonus', result_df.columns)
        
        # 계산된 값 확인
        for i, row in result_df.iterrows():
            self.assertEqual(row['bonus'], self.sample_df.loc[i, 'salary'] * 0.1)
            
        # 변환 히스토리 확인
        self.assertIn('bonus', self.processor.column_transformations)
        self.assertEqual(self.processor.column_transformations['bonus']['type'], 'derived')
    
    def test_filter_data(self):
        """
        filter_data 메서드 테스트
        """
        # 필터링 조건 함수
        def age_above_30(df):
            return df['age'] > 30
            
        def salary_above_70000(df):
            return df['salary'] > 70000
            
        # AND 조건으로 필터링
        result_df = self.processor.filter_data(
            self.sample_df,
            conditions=[age_above_30, salary_above_70000],
            combine='and'
        )
        
        # 결과 확인 (age > 30 AND salary > 70000)
        self.assertEqual(len(result_df), 4)  # 4개 행이 조건 만족
        
        # 모든 행이 조건을 만족하는지 확인
        for i, row in result_df.iterrows():
            self.assertTrue(row['age'] > 30 and row['salary'] > 70000)
            
        # OR 조건으로 필터링
        result_df = self.processor.filter_data(
            self.sample_df,
            conditions=[age_above_30, salary_above_70000],
            combine='or'
        )
        
        # 결과 확인 (age > 30 OR salary > 70000)
        # (age가 NaN인 경우는 False 반환)
        filtered_rows = self.sample_df[(self.sample_df['age'] > 30) | (self.sample_df['salary'] > 70000)]
        self.assertEqual(len(result_df), len(filtered_rows))

if __name__ == "__main__":
    unittest.main() 