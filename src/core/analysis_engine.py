#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
import os
import sys

# 현재 디렉토리를 경로에 추가 (상대 경로 import를 위해)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 주관식 분석을 위한 TextAnalyzer 임포트
from analysis.text_analyzer import TextAnalyzer

# 로거 설정
logger = logging.getLogger(__name__)

class AnalysisEngine:
    """
    데이터 분석을 수행하는 엔진 클래스
    기술통계, 상관관계 분석, 그룹별 분석 등 다양한 분석 기능 제공
    """
    
    def __init__(self):
        """AnalysisEngine 초기화"""
        self.df = None
        self.results = {}
        self.subjective_columns = []  # 주관식 열 목록
        self.text_analyzer = TextAnalyzer()  # 텍스트 분석기 초기화
        
        # 로그 디렉토리 확인
        os.makedirs("logs", exist_ok=True)
    
    def set_subjective_columns(self, columns: List[str]) -> None:
        """
        주관식 응답으로 처리할 열 목록 설정
        
        Args:
            columns (List[str]): 주관식 응답으로 처리할 열 이름 목록
        """
        self.subjective_columns = columns
        logger.info(f"주관식 응답 열로 {len(columns)}개 설정됨: {columns}")
        
        # 데이터프레임이 이미 있다면 분석 업데이트
        if self.df is not None:
            self.analyze_subjective_text()
    
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        데이터프레임에 대한 포괄적인 분석 수행
        
        Args:
            df (pd.DataFrame): 분석할 데이터프레임
            
        Returns:
            Dict[str, Any]: 종합 분석 결과
        """
        if df is None or df.empty:
            logger.warning("분석할 데이터가 없습니다")
            return self.results
            
        logger.info(f"데이터 분석 시작: {df.shape[0]} 행, {df.shape[1]} 열")
        
        # 분석 결과 초기화
        self.results = {}
        
        try:
            # 기본 통계량 계산
            self.get_basic_stats()
            
            # 수치형 통계량 계산
            self.get_numeric_stats()
            
            # 범주형 통계량 계산
            self.get_categorical_stats()
            
            # 상관관계 분석 (수치형 데이터가 2개 이상일 경우)
            num_cols = df.select_dtypes(include=['number']).columns
            if len(num_cols) >= 2:
                self.get_correlation_matrix()
                self.get_strong_correlations()
            
            # 가장 많은 고유값을 가진 상위 2개 범주형 열에 대한 그룹 분석
            cat_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
            if cat_cols and num_cols.any():
                # 고유값 수가 많은 순으로 범주형 열 정렬 (너무 많은 고유값은 제외)
                valid_cat_cols = [col for col in cat_cols if 2 <= df[col].nunique() <= 10]
                if valid_cat_cols:
                    # 고유값 수 기준으로 정렬
                    sorted_cat_cols = sorted(valid_cat_cols, 
                                            key=lambda col: df[col].nunique(), 
                                            reverse=True)
                    # 첫 번째 열에 대한 그룹 분석
                    if sorted_cat_cols:
                        self.group_analysis(group_by=sorted_cat_cols[0])
            
            # 주관식 데이터 분석 (주관식 열이 지정된 경우)
            if self.subjective_columns:
                self.analyze_subjective_text()
            
            logger.info("데이터 분석 완료")
            
        except Exception as e:
            logger.error(f"분석 중 오류 발생: {str(e)}")
            # 오류 발생 시 추적 정보 로깅
            import traceback
            logger.error(traceback.format_exc())
            
        return self.results
    
    def analyze_subjective_text(self) -> Dict[str, Any]:
        """
        주관식 텍스트 데이터 분석 수행
        
        Returns:
            Dict[str, Any]: 텍스트 분석 결과
        """
        if self.df is None or not self.subjective_columns:
            return None
        
        logger.info(f"주관식 텍스트 분석 시작: {len(self.subjective_columns)}개 열")
        
        text_analysis = {}
        
        for col in self.subjective_columns:
            if col in self.df.columns:
                logger.info(f"'{col}' 열 텍스트 분석 중...")
                text_result = self.text_analyzer.summarize_text_column(self.df, col)
                if text_result:
                    text_analysis[col] = text_result
                    logger.info(f"'{col}' 열 분석 완료: {text_result['response_count']}개 응답, "
                               f"{len(text_result['keywords'])}개 키워드 추출")
                else:
                    logger.warning(f"'{col}' 열 분석 결과 없음")
            else:
                logger.warning(f"지정된 주관식 열 '{col}'이(가) 데이터프레임에 없습니다")
        
        self.results['text_analysis'] = text_analysis
        logger.info(f"주관식 텍스트 분석 완료: {len(text_analysis)}개 열 처리됨")
        
        return text_analysis
    
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        분석할 데이터프레임 설정
        
        Args:
            df (pd.DataFrame): 분석 대상 데이터프레임
        """
        # 데이터프레임 복사 (원본 변경 방지)
        df = df.copy()
        
        # 데이터 타입 자동 변환 시도
        logger.info("데이터 타입 자동 감지 및 변환 시작")
        
        for col in df.columns:
            # 열의 현재 데이터 타입 확인
            orig_type = df[col].dtype
            logger.info(f"열: {col}, 원본 타입: {orig_type}")
            
            # 이미 수치형이면 건너뛰기
            if pd.api.types.is_numeric_dtype(orig_type):
                continue
                
            # 빈 값 또는 NaN이 아닌 행만 가져오기
            non_empty = df[col].dropna()
            if non_empty.empty:
                continue
                
            # 숫자 또는 숫자+기호로 이루어진 문자열인지 확인
            try:
                # 수치형으로 변환 시도
                numeric_values = pd.to_numeric(non_empty, errors='coerce')
                # NaN이 된 값의 비율이 30% 미만이면 수치형으로 간주
                nan_ratio = numeric_values.isna().mean()
                
                if nan_ratio < 0.3:  # 70% 이상이 숫자로 변환 가능하면
                    # 실제로 변환 적용
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    new_type = df[col].dtype
                    logger.info(f"열 '{col}' 타입 변환 성공: {orig_type} → {new_type}")
            except Exception as e:
                logger.debug(f"열 '{col}' 수치형 변환 시도 중 오류: {str(e)}")
        
        # 변환 후 데이터 타입 요약 출력
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        categorical_cols = list(df.select_dtypes(include=['object', 'category']).columns)
        logger.info(f"타입 변환 후 - 수치형 열: {len(numeric_cols)}개, 범주형 열: {len(categorical_cols)}개")
        
        self.df = df
        logger.info(f"데이터프레임 설정됨: 행={df.shape[0]}, 열={df.shape[1]}")
        
        # 데이터프레임 설정 시 기본 분석 수행
        self.analyze(df)
    
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """현재 데이터프레임 반환"""
        return self.df
    
    def get_basic_stats(self):
        """기본 통계 정보 수집"""
        if self.df is None:
            return None
            
        try:
            # 기본 정보 수집
            numeric_cols = list(self.df.select_dtypes(include=['number']).columns)
            categorical_cols = list(self.df.select_dtypes(include=['object', 'category']).columns)
            
            logger.info(f"수치형 열 감지: {len(numeric_cols)}개 ({numeric_cols})")
            logger.info(f"범주형 열 감지: {len(categorical_cols)}개 ({categorical_cols})")
            
            stats = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'column_names': list(self.df.columns),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'missing_values': self.df.isna().sum().to_dict()
            }
            
            # 결과 저장
            self.results['basic_stats'] = stats
            
            return stats
        except Exception as e:
            logger.error(f"기본 통계 수집 중 오류: {str(e)}")
            raise
    
    def get_numeric_stats(self):
        """수치형 통계 정보 수집"""
        if self.df is None:
            return None
            
        try:
            # 수치형 열만 선택
            numeric_df = self.df.select_dtypes(include=['number'])
            if numeric_df.empty:
                logger.warning("수치형 열이 없습니다")
                return None
                
            # 기본 통계량 계산 (열 중심 구조)
            numeric_stats = {}
            
            # 각 열에 대한 통계 계산
            for col in numeric_df.columns:
                col_data = numeric_df[col].dropna()
                
                # 기본 통계량
                stats = {
                    'count': int(col_data.count()),
                    'missing': int(self.df[col].isna().sum()),
                    'missing_pct': float(round(self.df[col].isna().mean() * 100, 2)),
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'var': float(col_data.var()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'range': float(col_data.max() - col_data.min()),
                    'unique_values': int(col_data.nunique())
                }
                
                # 분위수 계산
                for q in [0.1, 0.25, 0.5, 0.75, 0.9]:
                    stats[f'q{int(q*100)}'] = float(col_data.quantile(q))
                
                # 왜도와 첨도 계산
                try:
                    stats['skew'] = float(col_data.skew())
                    stats['kurtosis'] = float(col_data.kurtosis())
                except:
                    stats['skew'] = None
                    stats['kurtosis'] = None
                
                # 히스토그램 구간 정보
                try:
                    hist_values, hist_bins = np.histogram(col_data, bins=10)
                    stats['histogram'] = {
                        'values': hist_values.tolist(),
                        'bins': hist_bins.tolist()
                    }
                except:
                    stats['histogram'] = None
                
                # 열별 통계 저장
                numeric_stats[col] = stats
            
            # 결과 저장
            self.results['numeric_stats'] = numeric_stats
            
            logger.info(f"수치형 통계 계산 완료: {len(numeric_stats)}개 열")
            return numeric_stats
            
        except Exception as e:
            logger.error(f"수치형 통계 수집 중 오류: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_categorical_stats(self, columns: Optional[List[str]] = None, 
                             top_n: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        범주형 데이터에 대한 통계 정보 계산
        
        Args:
            columns (List[str], optional): 분석할 열 목록. None인 경우 모든 범주형/객체형 열 분석
            top_n (int): 출력할 최대 범주 수
            
        Returns:
            Dict[str, Dict[str, Any]]: 열별 범주형 통계 정보
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return {}
            
        if columns is None:
            # 범주형 또는 객체형 열 선택
            cat_cols = self.df.select_dtypes(include=['category', 'object']).columns.tolist()
            columns = cat_cols
            
        cat_stats = {}
        for col in columns:
            if col not in self.df.columns:
                self.logger.warning(f"열 '{col}'이 데이터프레임에 존재하지 않습니다.")
                continue
                
            try:
                # 해당 열이 범주형 또는 객체형인 경우에만 처리
                if isinstance(self.df[col].dtype, pd.CategoricalDtype) or pd.api.types.is_object_dtype(self.df[col]):
                    # 범주별 빈도수 계산
                    value_counts = self.df[col].value_counts()
                    top_values = value_counts.head(top_n)
                    
                    # 백분율 계산
                    total_valid = self.df[col].count()
                    pct_values = (top_values / total_valid * 100).round(2)
                    
                    # 결과 저장
                    col_stats = {
                        'count': total_valid,
                        'missing': self.df[col].isna().sum(),
                        'missing_pct': round(self.df[col].isna().mean() * 100, 2),
                        'unique_values': self.df[col].nunique(),
                        'top_values': {
                            str(category): {
                                'count': int(count),
                                'percentage': float(pct_values.get(category, 0))
                            } for category, count in top_values.items()
                        },
                        'data_type': str(self.df[col].dtype)
                    }
                    
                    cat_stats[col] = col_stats
                else:
                    self.logger.info(f"열 '{col}'은 범주형이 아니므로 범주형 통계에서 제외됩니다.")
            except Exception as e:
                self.logger.error(f"열 '{col}' 범주형 통계 계산 중 오류 발생: {str(e)}")
                
        self.results['categorical_stats'] = cat_stats
        return cat_stats
    
    def get_correlation_matrix(self, method: str = 'pearson', 
                              columns: Optional[List[str]] = None,
                              min_corr: float = 0.0) -> pd.DataFrame:
        """
        수치형 열 간의 상관관계 매트릭스 계산
        
        Args:
            method (str): 상관관계 계산 방법 ('pearson', 'spearman', 'kendall')
            columns (List[str], optional): 분석할 열 목록. None인 경우 모든 수치형 열 분석
            min_corr (float): 최소 상관계수 임계값 (절대값 기준). 이 값보다 작은 상관관계는 0으로 설정
            
        Returns:
            pd.DataFrame: 상관관계 매트릭스
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        # 수치형 열만 선택
        if columns is None:
            numeric_df = self.df.select_dtypes(include=['number'])
        else:
            # 지정된 열 중 수치형 열만 필터링
            valid_columns = [col for col in columns if col in self.df.columns 
                             and pd.api.types.is_numeric_dtype(self.df[col])]
            
            if not valid_columns:
                self.logger.warning("유효한 수치형 열이 없습니다.")
                return pd.DataFrame()
                
            numeric_df = self.df[valid_columns]
            
        # 상관관계 계산
        try:
            corr_matrix = numeric_df.corr(method=method)
            
            # 최소 상관계수 임계값 적용
            if min_corr > 0:
                corr_matrix = corr_matrix.where(abs(corr_matrix) >= min_corr, 0)
                
            self.results['correlation'] = corr_matrix
            return corr_matrix
        except Exception as e:
            self.logger.error(f"상관관계 계산 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def get_strong_correlations(self, method: str = 'pearson', 
                               threshold: float = 0.7,
                               exclude_self: bool = True) -> List[Tuple[str, str, float]]:
        """
        강한 상관관계가 있는 변수 쌍 추출
        
        Args:
            method (str): 상관관계 계산 방법 ('pearson', 'spearman', 'kendall')
            threshold (float): 강한 상관관계로 간주할 임계값
            exclude_self (bool): 자기 자신과의 상관관계(항상 1.0) 제외 여부
            
        Returns:
            List[Tuple[str, str, float]]: (변수1, 변수2, 상관계수) 형태의 리스트
        """
        corr_matrix = self.get_correlation_matrix(method=method)
        if corr_matrix.empty:
            return []
            
        # 상관관계가 강한 변수 쌍 추출
        strong_correlations = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i if exclude_self else i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) >= threshold:
                    strong_correlations.append((col1, col2, corr_value))
                    
        # 상관계수 절대값 기준 내림차순 정렬
        strong_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return strong_correlations
    
    def group_analysis(self, group_by: Union[str, List[str]], 
                      agg_columns: Optional[List[str]] = None,
                      agg_funcs: Optional[Dict[str, List[str]]] = None) -> pd.DataFrame:
        """
        그룹별 집계 분석 수행
        
        Args:
            group_by (Union[str, List[str]]): 그룹화 기준 열
            agg_columns (List[str], optional): 집계할 열 목록. None인 경우 모든 수치형 열 사용
            agg_funcs (Dict[str, List[str]], optional): 열별 집계 함수 지정
                                                      예: {'age': ['mean', 'std'], 'salary': ['median', 'min', 'max']}
                                                      None인 경우 모든 열에 대해 기본 집계 함수 사용
        
        Returns:
            pd.DataFrame: 그룹별 집계 결과
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        # 그룹화 기준 열 확인
        if isinstance(group_by, str):
            group_by = [group_by]
            
        for col in group_by:
            if col not in self.df.columns:
                self.logger.error(f"그룹화 기준 열 '{col}'이 데이터프레임에 존재하지 않습니다.")
                return pd.DataFrame()
                
        # 집계할 열 선택
        if agg_columns is None:
            # 수치형 열만 자동 선택 (그룹화 기준 열 제외)
            numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
            agg_columns = [col for col in numeric_cols if col not in group_by]
        else:
            # 유효한 열만 필터링
            agg_columns = [col for col in agg_columns if col in self.df.columns and col not in group_by]
            
        if not agg_columns:
            self.logger.warning("집계할 유효한 열이 없습니다.")
            return pd.DataFrame()
            
        # 기본 집계 함수 정의
        default_agg_funcs = ['count', 'mean', 'median', 'std', 'min', 'max']
        
        # 집계 함수 설정
        if agg_funcs is None:
            # 모든 집계 열에 동일한 기본 집계 함수 적용
            agg_funcs = {col: default_agg_funcs for col in agg_columns}
        else:
            # 사용자 지정 집계 함수 사용
            for col in agg_columns:
                if col not in agg_funcs:
                    # 집계 함수가 지정되지 않은 열에 대해 기본 집계 함수 적용
                    agg_funcs[col] = default_agg_funcs
                    
        try:
            # 그룹별 집계 수행
            grouped = self.df.groupby(group_by)
            result = grouped.agg(agg_funcs)
            
            # 결과 저장
            group_key = '_'.join(group_by)
            self.results[f'group_by_{group_key}'] = result
            
            return result
        except Exception as e:
            self.logger.error(f"그룹별 분석 중 오류 발생: {str(e)}")
            return pd.DataFrame()
    
    def time_series_analysis(self, date_column: str, 
                           value_column: str,
                           freq: str = 'ME',
                           agg_func: str = 'mean') -> pd.DataFrame:
        """
        시계열 데이터 분석 수행
        
        Args:
            date_column (str): 날짜/시간 열
            value_column (str): 분석할 값 열
            freq (str): 리샘플링 주기 ('D': 일별, 'W': 주별, 'ME': 월별, 'QE': 분기별, 'YE': 연별)
            agg_func (str): 집계 함수 ('mean', 'sum', 'count', 'min', 'max' 등)
            
        Returns:
            pd.DataFrame: 시계열 분석 결과
        """
        if self.df is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        # 열 유효성 검사
        if date_column not in self.df.columns:
            self.logger.error(f"날짜 열 '{date_column}'이 데이터프레임에 존재하지 않습니다.")
            return pd.DataFrame()
            
        if value_column not in self.df.columns:
            self.logger.error(f"값 열 '{value_column}'이 데이터프레임에 존재하지 않습니다.")
            return pd.DataFrame()
            
        # 날짜 열이 datetime 타입인지 확인
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_column]):
            try:
                # datetime으로 변환 시도
                self.logger.info(f"'{date_column}' 열을 datetime 형식으로 변환합니다.")
                date_series = pd.to_datetime(self.df[date_column])
            except Exception as e:
                self.logger.error(f"날짜 열 변환 중 오류 발생: {str(e)}")
                return pd.DataFrame()
        else:
            date_series = self.df[date_column]
            
        try:
            # 시계열 데이터 준비
            ts_df = pd.DataFrame({
                'date': date_series,
                'value': self.df[value_column]
            }).dropna()
            
            # 시계열 인덱스 설정
            ts_df.set_index('date', inplace=True)
            
            # 시계열 리샘플링 및 집계
            if agg_func == 'mean':
                resampled = ts_df.resample(freq).mean()
            elif agg_func == 'sum':
                resampled = ts_df.resample(freq).sum()
            elif agg_func == 'count':
                resampled = ts_df.resample(freq).count()
            elif agg_func == 'min':
                resampled = ts_df.resample(freq).min()
            elif agg_func == 'max':
                resampled = ts_df.resample(freq).max()
            else:
                self.logger.warning(f"지원되지 않는 집계 함수 '{agg_func}'입니다. 'mean'을 사용합니다.")
                resampled = ts_df.resample(freq).mean()
                
            # 결과 저장
            self.results[f'time_series_{date_column}_{value_column}_{freq}'] = resampled
            
            return resampled
        except Exception as e:
            self.logger.error(f"시계열 분석 중 오류 발생: {str(e)}")
            return pd.DataFrame()
            
    def get_all_results(self):
        """모든 분석 결과 반환"""
        return self.results 