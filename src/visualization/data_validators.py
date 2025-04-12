import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional, Any, Set

# 로깅 설정
logger = logging.getLogger(__name__)

def validate_dataframe(df: pd.DataFrame) -> bool:
    """
    데이터프레임 유효성 검증
    
    Args:
        df (pd.DataFrame): 검증할 데이터프레임
        
    Returns:
        bool: 유효한 데이터프레임인지 여부
    
    Raises:
        ValueError: 데이터프레임이 None이거나 빈 경우
    """
    if df is None:
        raise ValueError("데이터프레임이 None입니다.")
    
    if len(df) == 0:
        raise ValueError("데이터프레임이 비어 있습니다.")
    
    return True

def validate_column(df: pd.DataFrame, column: str) -> bool:
    """
    열 존재 여부 검증
    
    Args:
        df (pd.DataFrame): 데이터프레임
        column (str): 검증할 열 이름
        
    Returns:
        bool: 열이 존재하는지 여부
    
    Raises:
        ValueError: 열이 존재하지 않는 경우
    """
    if column not in df.columns:
        raise ValueError(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
    
    return True

def validate_column_data(df: pd.DataFrame, column: str) -> pd.Series:
    """
    열 데이터 검증 및 반환
    
    Args:
        df (pd.DataFrame): 데이터프레임
        column (str): 검증할 열 이름
        
    Returns:
        pd.Series: 검증된 열 데이터
        
    Raises:
        ValueError: 열이 존재하지 않거나 모든 값이 결측치인 경우
    """
    # 열 존재 여부 확인
    validate_column(df, column)
    
    # 열 데이터 추출
    series = df[column]
    
    # 모든 값이 결측치인지 확인
    if series.isna().all():
        raise ValueError(f"열 '{column}'의 모든 값이 결측치입니다.")
    
    return series

def validate_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    수치형 열 데이터 검증 및 반환
    
    Args:
        df (pd.DataFrame): 데이터프레임
        column (str): 검증할 열 이름
        
    Returns:
        pd.Series: 검증된 수치형 열 데이터
        
    Raises:
        ValueError: 열이 존재하지 않거나 수치형이 아닌 경우
    """
    # 열 데이터 검증
    series = validate_column_data(df, column)
    
    # 수치형 변환 가능 여부 확인
    try:
        # Timedelta 처리
        if pd.api.types.is_timedelta64_dtype(series):
            return series.dt.total_seconds()
        
        # 이미 수치형인 경우
        if pd.api.types.is_numeric_dtype(series):
            return series
        
        # 수치형으로 변환 시도
        numeric_series = pd.to_numeric(series, errors='coerce')
        
        # 변환 후 결측치가 너무 많은 경우
        missing_after_conversion = numeric_series.isna().mean()
        if missing_after_conversion > 0.5:  # 50% 이상이 변환 실패
            raise ValueError(f"열 '{column}'의 값 중 {missing_after_conversion:.1%}가 수치형으로 변환할 수 없습니다.")
        
        return numeric_series
    except Exception as e:
        raise ValueError(f"열 '{column}'을 수치형으로 처리할 수 없습니다: {str(e)}")

def validate_categorical_column(df: pd.DataFrame, column: str, top_n: Optional[int] = None) -> Tuple[List[str], List[int]]:
    """
    범주형 열 데이터 검증 및 상위 범주 반환
    
    Args:
        df (pd.DataFrame): 데이터프레임
        column (str): 검증할 열 이름
        top_n (int, optional): 상위 범주 수
        
    Returns:
        Tuple[List[str], List[int]]: (범주 목록, 카운트 목록)
        
    Raises:
        ValueError: 열이 존재하지 않거나 처리할 수 없는 경우
    """
    # 열 데이터 검증
    series = validate_column_data(df, column)
    
    try:
        # 값 카운트
        value_counts = series.value_counts()
        
        # 범주가 너무 많은 경우
        if len(value_counts) > 100 and top_n is None:
            raise ValueError(f"열 '{column}'의 고유 값이 {len(value_counts)}개로 너무 많습니다.")
        
        # 상위 N개 범주만 선택
        if top_n is not None and len(value_counts) > top_n:
            # 상위 N개 선택
            top_counts = value_counts.head(top_n)
            
            # 기타 범주 추가
            other_count = value_counts.sum() - top_counts.sum()
            
            if other_count > 0:
                categories = top_counts.index.tolist() + ['기타']
                counts = top_counts.values.tolist() + [other_count]
            else:
                categories = top_counts.index.tolist()
                counts = top_counts.values.tolist()
        else:
            categories = value_counts.index.tolist()
            counts = value_counts.values.tolist()
        
        # 문자열로 변환
        categories = [str(cat) for cat in categories]
        
        return categories, counts
    except Exception as e:
        raise ValueError(f"열 '{column}'을 범주형으로 처리할 수 없습니다: {str(e)}") 