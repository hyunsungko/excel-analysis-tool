#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DataProcessor 모듈: 데이터 전처리 및 변환 기능을 담당합니다.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Union, Optional, Tuple, Any, Callable
from datetime import datetime

# 로거 설정
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    데이터프레임의 전처리 및 변환을 담당하는 클래스
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None):
        """
        DataProcessor 클래스 초기화
        
        Parameters:
        -----------
        df : pandas.DataFrame, optional
            처리할 데이터프레임
        """
        self.df = df
        self.original_df = df.copy() if df is not None else None
        self.preprocessing_steps = []
        self.column_transformations = {}
        
    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        데이터 처리 메서드 - 전처리 및 변환 과정을 한번에 수행
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
            
        Returns:
        --------
        pandas.DataFrame
            처리된 데이터프레임
        """
        logger.info("데이터 처리 시작")
        start_time = datetime.now()
        
        # 데이터프레임 설정
        self.set_dataframe(df)
        
        # 처리할 데이터 크기 확인
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        rows, cols = df.shape
        
        logger.info(f"데이터 크기: {rows}행 x {cols}열 (약 {memory_usage_mb:.2f}MB)")
        
        # 기본 전처리 단계 설정
        if len(self.preprocessing_steps) == 0:
            # 결측치 처리 - 기본적으로 결측치가 50% 이상인 열만 처리
            self.add_preprocessing_step("결측치 처리", self.handle_missing_values, 
                                       strategy='fill_values', threshold=0.5)
            
            # 이상치 처리 - 수치형 열에 대해서만 (대용량 데이터에서는 처리 제한)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_cols and rows < 100000:  # 10만 행 미만인 경우만 이상치 처리
                self.add_preprocessing_step("이상치 처리", self.handle_outliers,
                                          columns=numeric_cols, method='iqr', strategy='remove')
        
        # 전처리 적용 (분할 처리 - 대용량 데이터 처리 최적화)
        if rows > 500000:  # 50만 행 이상인 경우 분할 처리
            # 데이터 프레임을 여러 청크로 분할하여 처리
            chunk_size = 100000  # 10만 행씩 처리
            chunks = []
            logger.info(f"대용량 데이터 분할 처리: {rows}행을 {chunk_size}행 단위로 처리")
            
            for i in range(0, rows, chunk_size):
                end_idx = min(i + chunk_size, rows)
                logger.info(f"청크 처리 중: {i}~{end_idx}행")
                chunk = self.df.iloc[i:end_idx].copy()
                
                # 청크별 전처리 적용
                for name, func, kwargs in self.preprocessing_steps:
                    chunk = func(chunk, **kwargs)
                
                chunks.append(chunk)
                
            # 처리된 청크 결합
            processed_df = pd.concat(chunks, ignore_index=True)
        else:
            # 일반 처리 (전체 데이터 한 번에 처리)
            processed_df = self.apply_preprocessing()
        
        end_time = datetime.now()
        process_time = (end_time - start_time).total_seconds()
        
        logger.info(f"데이터 처리 완료 (소요 시간: {process_time:.2f}초)")
        return processed_df
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        처리할 데이터프레임 설정
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        """
        self.df = df
        self.original_df = df.copy()
        
    def add_preprocessing_step(self, step_name: str, function: Callable, **kwargs) -> None:
        """
        전처리 단계 추가
        
        Parameters:
        -----------
        step_name : str
            전처리 단계 이름
        function : callable
            전처리 함수
        **kwargs
            전처리 함수에 전달할 인수
        """
        self.preprocessing_steps.append({
            'name': step_name,
            'function': function,
            'kwargs': kwargs
        })
        logger.info(f"전처리 단계 추가: {step_name}")
        
    def apply_preprocessing(self) -> pd.DataFrame:
        """
        모든 전처리 단계 적용
        
        Returns:
        --------
        pandas.DataFrame
            전처리된 데이터프레임
        """
        if self.df is None:
            raise ValueError("데이터프레임이 설정되지 않았습니다. set_dataframe()을 먼저 호출하세요.")
            
        # 원본 데이터 복사
        processed_df = self.df.copy()
        
        # 각 전처리 단계 적용
        for step in self.preprocessing_steps:
            try:
                logger.info(f"전처리 단계 적용: {step['name']}")
                processed_df = step['function'](processed_df, **step['kwargs'])
            except Exception as e:
                logger.error(f"전처리 단계 '{step['name']}' 적용 중 오류 발생: {str(e)}")
                raise
                
        return processed_df
    
    def handle_missing_values(self, 
                              df: pd.DataFrame, 
                              strategy: str = 'drop_rows', 
                              columns: Optional[List[str]] = None, 
                              threshold: float = 0.5,
                              fill_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        결측치 처리
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        strategy : str
            결측치 처리 전략 ('drop_rows', 'drop_columns', 'fill_values')
        columns : list of str, optional
            처리할 열 목록. None이면 모든 열 처리
        threshold : float
            결측치 비율 임계값 (0.0 ~ 1.0). 이 값을 초과하는 경우만 처리
        fill_values : dict, optional
            각 열의 결측치를 채울 값들의 딕셔너리
            
        Returns:
        --------
        pandas.DataFrame
            결측치 처리된 데이터프레임
        """
        result_df = df.copy()
        
        # 처리할 열 선택
        if columns is None:
            target_columns = df.columns
        else:
            target_columns = [col for col in columns if col in df.columns]
            
        # 각 열의 결측치 비율 계산
        missing_ratio = df[target_columns].isnull().mean()
        columns_to_process = missing_ratio[missing_ratio > threshold].index.tolist()
        
        if not columns_to_process:
            logger.info("임계값을 초과하는 결측치가 있는 열이 없습니다.")
            return result_df
            
        # 선택된 전략에 따라 결측치 처리
        if strategy == 'drop_rows':
            # 해당 열에 결측치가 있는 행 삭제
            result_df = result_df.dropna(subset=columns_to_process)
            logger.info(f"{len(df) - len(result_df)}개 행이 삭제되었습니다.")
            
        elif strategy == 'drop_columns':
            # 해당 열 삭제
            result_df = result_df.drop(columns=columns_to_process)
            logger.info(f"{len(columns_to_process)}개 열이 삭제되었습니다: {columns_to_process}")
            
        elif strategy == 'fill_values':
            # 지정된 값으로 결측치 채우기
            if fill_values:
                for col in columns_to_process:
                    if col in fill_values:
                        result_df[col] = result_df[col].fillna(fill_values[col])
                        logger.info(f"'{col}' 열의 결측치를 {fill_values[col]}로 채웠습니다.")
                    else:
                        # 채울 값이 지정되지 않은 경우 기본 전략 사용
                        if pd.api.types.is_numeric_dtype(result_df[col]):
                            # 수치형 데이터는 평균으로 채우기
                            result_df[col] = result_df[col].fillna(result_df[col].mean())
                            logger.info(f"'{col}' 열의 결측치를 평균값으로 채웠습니다.")
                        elif pd.api.types.is_categorical_dtype(result_df[col]) or pd.api.types.is_object_dtype(result_df[col]):
                            # 범주형 데이터는 최빈값으로 채우기
                            mode_value = result_df[col].mode()[0] if not result_df[col].mode().empty else None
                            result_df[col] = result_df[col].fillna(mode_value)
                            logger.info(f"'{col}' 열의 결측치를 최빈값으로 채웠습니다.")
                        else:
                            # 그 외 데이터는 처리하지 않음
                            logger.warning(f"'{col}' 열의 결측치를 처리하지 않았습니다. 지원되지 않는 데이터 타입입니다.")
            else:
                logger.warning("fill_values 전략이 선택되었지만 채울 값이 제공되지 않았습니다.")
        else:
            logger.warning(f"지원되지 않는 결측치 처리 전략: {strategy}")
            
        return result_df
    
    def convert_data_types(self, 
                          df: pd.DataFrame, 
                          type_conversions: Dict[str, str],
                          handle_errors: str = 'coerce') -> pd.DataFrame:
        """
        데이터 타입 변환
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        type_conversions : dict
            변환할 열과 타입의 딕셔너리 {'column_name': 'target_type'}
            지원되는 타입: 'int', 'float', 'str', 'bool', 'datetime', 'category'
        handle_errors : str
            오류 처리 방식 ('ignore', 'raise', 'coerce')
            
        Returns:
        --------
        pandas.DataFrame
            타입 변환된 데이터프레임
        """
        result_df = df.copy()
        
        for column, target_type in type_conversions.items():
            if column not in df.columns:
                logger.warning(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
                continue
                
            try:
                if target_type == 'int':
                    # 정수형 변환
                    if handle_errors == 'coerce':
                        result_df[column] = pd.to_numeric(result_df[column], errors='coerce').astype('Int64')
                    else:
                        result_df[column] = result_df[column].astype('int64')
                        
                elif target_type == 'float':
                    # 실수형 변환
                    result_df[column] = pd.to_numeric(result_df[column], errors=handle_errors)
                    
                elif target_type == 'str':
                    # 문자열 변환
                    result_df[column] = result_df[column].astype(str)
                    
                elif target_type == 'bool':
                    # 불리언 변환
                    if handle_errors == 'coerce':
                        # 일반적인 불리언 문자열 처리
                        bool_map = {'true': True, 'false': False, 
                                   'yes': True, 'no': False, 
                                   '1': True, '0': False,
                                   1: True, 0: False,
                                   'y': True, 'n': False}
                        result_df[column] = result_df[column].map(
                            lambda x: bool_map.get(str(x).lower(), None)
                        )
                    else:
                        result_df[column] = result_df[column].astype(bool)
                        
                elif target_type == 'datetime':
                    # 날짜/시간 변환
                    result_df[column] = pd.to_datetime(result_df[column], errors=handle_errors)
                    
                elif target_type == 'category':
                    # 범주형 변환
                    result_df[column] = result_df[column].astype('category')
                    
                else:
                    logger.warning(f"지원되지 않는 데이터 타입: {target_type}")
                    
                logger.info(f"'{column}' 열을 {target_type} 타입으로 변환했습니다.")
                
            except Exception as e:
                if handle_errors == 'raise':
                    raise
                else:
                    logger.warning(f"'{column}' 열 변환 중 오류 발생: {str(e)}")
                    
        return result_df
    
    def handle_outliers(self, 
                       df: pd.DataFrame, 
                       columns: Optional[List[str]] = None,
                       method: str = 'iqr',
                       threshold: float = 1.5,
                       strategy: str = 'remove') -> pd.DataFrame:
        """
        이상치 처리
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        columns : list of str, optional
            처리할 열 목록. None이면 모든 수치형 열 처리
        method : str
            이상치 탐지 방법 ('iqr', 'zscore', 'percentile')
        threshold : float
            이상치 탐지 임계값
        strategy : str
            이상치 처리 전략 ('remove', 'clip', 'replace_mean', 'replace_median')
            
        Returns:
        --------
        pandas.DataFrame
            이상치 처리된 데이터프레임
        """
        result_df = df.copy()
        
        # 처리할 열 선택
        if columns is None:
            # 모든 수치형 열 선택
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            columns = numeric_columns
        else:
            # 수치형 열만 필터링
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
        if not columns:
            logger.info("처리할 수치형 열이 없습니다.")
            return result_df
            
        # 각 열에 대해 이상치 처리
        for column in columns:
            # 이상치 탐지
            if method == 'iqr':
                # IQR 방법
                Q1 = result_df[column].quantile(0.25)
                Q3 = result_df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outliers = (result_df[column] < lower_bound) | (result_df[column] > upper_bound)
                
            elif method == 'zscore':
                # Z-Score 방법
                from scipy import stats
                z_scores = np.abs(stats.zscore(result_df[column], nan_policy='omit'))
                outliers = z_scores > threshold
                
            elif method == 'percentile':
                # 백분위수 방법
                lower_bound = result_df[column].quantile(threshold / 100)
                upper_bound = result_df[column].quantile(1 - threshold / 100)
                outliers = (result_df[column] < lower_bound) | (result_df[column] > upper_bound)
                
            else:
                logger.warning(f"지원되지 않는 이상치 탐지 방법: {method}")
                continue
                
            # 이상치 개수 로깅
            outlier_count = outliers.sum()
            if outlier_count > 0:
                logger.info(f"'{column}' 열에서 {outlier_count}개의 이상치 발견")
                
                # 이상치 처리 전략 적용
                if strategy == 'remove':
                    # 이상치가 있는 행 제거
                    result_df = result_df[~outliers]
                    logger.info(f"이상치가 있는 {outlier_count}개 행 제거")
                    
                elif strategy == 'clip':
                    # 이상치 자르기 (경계값으로 대체)
                    if method == 'iqr' or method == 'percentile':
                        result_df.loc[result_df[column] < lower_bound, column] = lower_bound
                        result_df.loc[result_df[column] > upper_bound, column] = upper_bound
                        logger.info(f"'{column}' 열의 이상치를 {lower_bound}와 {upper_bound} 사이로 자름")
                    else:
                        # z-score 방법에서는 평균에서 threshold 표준편차 밖으로 나간 값을 자름
                        mean = result_df[column].mean()
                        std = result_df[column].std()
                        lower_bound = mean - threshold * std
                        upper_bound = mean + threshold * std
                        result_df.loc[result_df[column] < lower_bound, column] = lower_bound
                        result_df.loc[result_df[column] > upper_bound, column] = upper_bound
                        logger.info(f"'{column}' 열의 이상치를 {lower_bound}와 {upper_bound} 사이로 자름")
                        
                elif strategy == 'replace_mean':
                    # 이상치를 평균으로 대체
                    mean_value = result_df.loc[~outliers, column].mean()
                    result_df.loc[outliers, column] = mean_value
                    logger.info(f"'{column}' 열의 이상치를 평균값 {mean_value}으로 대체")
                    
                elif strategy == 'replace_median':
                    # 이상치를 중앙값으로 대체
                    median_value = result_df.loc[~outliers, column].median()
                    result_df.loc[outliers, column] = median_value
                    logger.info(f"'{column}' 열의 이상치를 중앙값 {median_value}으로 대체")
                    
                else:
                    logger.warning(f"지원되지 않는 이상치 처리 전략: {strategy}")
                    
            else:
                logger.info(f"'{column}' 열에서 이상치가 발견되지 않았습니다.")
                
        return result_df
    
    def normalize_data(self, 
                      df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      method: str = 'minmax',
                      custom_range: Optional[Tuple[float, float]] = None) -> pd.DataFrame:
        """
        데이터 정규화
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        columns : list of str, optional
            처리할 열 목록. None이면 모든 수치형 열 처리
        method : str
            정규화 방법 ('minmax', 'zscore', 'robust', 'log')
        custom_range : tuple of float, optional
            minmax 정규화의 범위 (기본값: (0, 1))
            
        Returns:
        --------
        pandas.DataFrame
            정규화된 데이터프레임
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        
        result_df = df.copy()
        
        # 처리할 열 선택
        if columns is None:
            # 모든 수치형 열 선택
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            columns = numeric_columns
        else:
            # 수치형 열만 필터링
            columns = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            
        if not columns:
            logger.info("처리할 수치형 열이 없습니다.")
            return result_df
            
        # 각 정규화 방법 적용
        if method == 'minmax':
            # Min-Max 정규화
            if custom_range:
                scaler = MinMaxScaler(feature_range=custom_range)
            else:
                scaler = MinMaxScaler()
                
            result_df[columns] = scaler.fit_transform(result_df[columns])
            logger.info(f"{len(columns)}개 열에 Min-Max 정규화 적용")
            
        elif method == 'zscore':
            # Z-Score 정규화 (평균=0, 표준편차=1)
            scaler = StandardScaler()
            result_df[columns] = scaler.fit_transform(result_df[columns])
            logger.info(f"{len(columns)}개 열에 Z-Score 정규화 적용")
            
        elif method == 'robust':
            # Robust 정규화 (중앙값=0, IQR=1)
            scaler = RobustScaler()
            result_df[columns] = scaler.fit_transform(result_df[columns])
            logger.info(f"{len(columns)}개 열에 Robust 정규화 적용")
            
        elif method == 'log':
            # 로그 변환
            for col in columns:
                # 0 이하의 값이 있는지 확인
                min_value = result_df[col].min()
                if min_value <= 0:
                    # 모든 값을 양수로 만들기 위해 오프셋 추가
                    offset = abs(min_value) + 1
                    result_df[col] = np.log(result_df[col] + offset)
                    logger.info(f"'{col}' 열에 로그 변환 적용 (오프셋: {offset})")
                else:
                    result_df[col] = np.log(result_df[col])
                    logger.info(f"'{col}' 열에 로그 변환 적용")
                    
        else:
            logger.warning(f"지원되지 않는 정규화 방법: {method}")
            
        return result_df
    
    def add_derived_column(self, 
                          df: pd.DataFrame, 
                          new_column: str,
                          formula: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
        """
        파생 열 추가
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        new_column : str
            새 열 이름
        formula : callable
            새 열의 값을 계산하는 함수
            
        Returns:
        --------
        pandas.DataFrame
            새 열이 추가된 데이터프레임
        """
        result_df = df.copy()
        
        try:
            result_df[new_column] = formula(result_df)
            logger.info(f"'{new_column}' 파생 열이 추가되었습니다.")
            
            # 열 변환 기록
            self.column_transformations[new_column] = {'type': 'derived', 'formula': formula.__name__}
            
        except Exception as e:
            logger.error(f"파생 열 추가 중 오류 발생: {str(e)}")
            raise
            
        return result_df
    
    def filter_data(self, 
                   df: pd.DataFrame, 
                   conditions: List[Callable[[pd.DataFrame], pd.Series]],
                   combine: str = 'and') -> pd.DataFrame:
        """
        조건에 따라 데이터 필터링
        
        Parameters:
        -----------
        df : pandas.DataFrame
            처리할 데이터프레임
        conditions : list of callable
            필터링 조건 함수 목록
        combine : str
            조건 결합 방식 ('and' 또는 'or')
            
        Returns:
        --------
        pandas.DataFrame
            필터링된 데이터프레임
        """
        result_df = df.copy()
        
        if not conditions:
            logger.warning("필터링 조건이 제공되지 않았습니다.")
            return result_df
            
        # 각 조건 적용
        mask = None
        for i, condition in enumerate(conditions):
            try:
                current_mask = condition(result_df)
                
                if mask is None:
                    mask = current_mask
                else:
                    if combine.lower() == 'and':
                        mask = mask & current_mask
                    elif combine.lower() == 'or':
                        mask = mask | current_mask
                    else:
                        logger.warning(f"지원되지 않는 조건 결합 방식: {combine}")
                        return result_df
                        
            except Exception as e:
                logger.error(f"조건 {i+1} 적용 중 오류 발생: {str(e)}")
                raise
                
        # 필터링 적용
        filtered_df = result_df[mask]
        logger.info(f"{len(result_df) - len(filtered_df)}개 행이 필터링되었습니다. (원본: {len(result_df)}, 결과: {len(filtered_df)})")
        
        return filtered_df
    
    def get_transformation_history(self) -> List[Dict]:
        """
        전처리 및 변환 히스토리 반환
        
        Returns:
        --------
        list of dict
            전처리 및 변환 단계 목록
        """
        return self.preprocessing_steps 