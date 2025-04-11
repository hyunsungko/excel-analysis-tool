#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DataLoader 모듈: 엑셀 파일 로드 및 데이터 형식 감지를 담당합니다.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# 로거 설정
logger = logging.getLogger(__name__)

class DataLoader:
    """
    다양한 형태의 엑셀 파일을 로드하고 데이터 형식을 자동으로 감지하는 클래스
    """
    
    def __init__(self):
        """
        DataLoader 클래스 초기화
        """
        self.data_dir = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) / 'data'
        self.current_file = None
        self.df = None
        self.file_info = {}
        self.column_types = {}
        
    def list_data_files(self, pattern='*.xlsx'):
        """
        data 디렉토리에서 지정된 패턴과 일치하는 파일 목록을 반환합니다.
        
        Parameters:
        -----------
        pattern : str
            검색할 파일 패턴 (기본값: '*.xlsx')
            
        Returns:
        --------
        list
            파일 경로 목록
        """
        try:
            if not self.data_dir.exists():
                logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")
                return []
            
            # 지정된 패턴과 일치하는 파일 목록 반환
            files = list(self.data_dir.glob(pattern))
            
            # 임시 파일 필터링 (엑셀에서 생성하는 임시 파일 제외)
            files = [f for f in files if not f.name.startswith('~$')]
            
            return files
        except Exception as e:
            logger.error(f"파일 목록을 가져오는 중 오류 발생: {str(e)}")
            return []
    
    def load_file(self, file_path=None, sheet_name=0):
        """
        지정된 엑셀 파일을 로드합니다.
        
        Parameters:
        -----------
        file_path : str or Path, optional
            로드할 파일 경로. None인 경우 data 디렉토리의 첫 번째 엑셀 파일을 사용
        sheet_name : str or int, optional
            로드할 시트 이름 또는 인덱스 (기본값: 0, 첫 번째 시트)
            
        Returns:
        --------
        pandas.DataFrame
            로드된 데이터프레임
            
        Raises:
        -------
        FileNotFoundError
            파일을 찾을 수 없는 경우
        ValueError
            파일 로드 중 오류가 발생한 경우
        """
        try:
            # 파일 경로가 None인 경우 첫 번째 엑셀 파일 사용
            if file_path is None:
                files = self.list_data_files()
                if not files:
                    raise FileNotFoundError("data 디렉토리에 엑셀 파일이 없습니다.")
                file_path = files[0]
            
            # 문자열 경로를 Path 객체로 변환
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            # 파일 존재 확인
            if not file_path.exists():
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
                
            # 파일 크기 확인 (MB 단위)
            file_size = file_path.stat().st_size / (1024 * 1024)
            use_efficient_loading = file_size > 100  # 100MB 이상이면 효율적 로딩 사용
            
            start_time = datetime.now()
            extension = file_path.suffix.lower()
            
            # 파일 타입에 따라 다른 로딩 방식 사용
            if extension == '.csv':
                # CSV 파일 로드
                if use_efficient_loading:
                    # 열 타입 감지를 위해 샘플 로드
                    sample = pd.read_csv(file_path, nrows=1000)
                    dtypes = {col: sample[col].dtype for col in sample.columns}
                    
                    # 문자열 열에 대해 범주형 데이터 타입 사용
                    for col, dtype in dtypes.items():
                        if dtype == 'object':
                            dtypes[col] = 'category'
                    
                    # 청크 단위로 로드하고 결합
                    chunks = []
                    for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=10000):
                        chunks.append(chunk)
                    self.df = pd.concat(chunks, ignore_index=True)
                else:
                    self.df = pd.read_csv(file_path)
            elif extension in ['.xlsx', '.xls']:
                # Excel 파일 로드
                if use_efficient_loading:
                    # 엑셀 엔진 선택
                    excel_engine = 'openpyxl' if extension == '.xlsx' else 'xlrd'
                    
                    # 시트 이름 목록 가져오기
                    with pd.ExcelFile(file_path, engine=excel_engine) as xl:
                        available_sheets = xl.sheet_names
                    
                    # 시트 선택 (인덱스 또는 이름)
                    if isinstance(sheet_name, int) and sheet_name < len(available_sheets):
                        sheet_to_load = available_sheets[sheet_name]
                    elif isinstance(sheet_name, str) and sheet_name in available_sheets:
                        sheet_to_load = sheet_name
                    else:
                        if available_sheets:
                            sheet_to_load = available_sheets[0]
                            logger.warning(f"지정한 시트가 없어 첫 번째 시트를 사용합니다: {sheet_to_load}")
                        else:
                            raise ValueError("엑셀 파일에 시트가 없습니다.")
                    
                    # 선택한 시트 로드
                    self.df = pd.read_excel(
                        file_path, 
                        sheet_name=sheet_to_load,
                        engine=excel_engine
                    )
                else:
                    self.df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}")
            
            load_time = (datetime.now() - start_time).total_seconds()
            
            # 파일 정보 저장
            self.current_file = file_path
            self.file_info = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_path.stat().st_size,
                'file_size_mb': round(file_size, 2),
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'sheet_name': sheet_name,
                'load_time_seconds': load_time,
                'row_count': len(self.df),
                'column_count': len(self.df.columns)
            }
            
            # 메모리 최적화
            self._optimize_dataframe()
            
            # 열 데이터 유형 감지
            self._detect_column_types()
            
            logger.info(f"파일 로드 완료: {file_path.name} ({len(self.df)} 행, {len(self.df.columns)} 열)")
            return self.df
            
        except FileNotFoundError as e:
            logger.error(f"파일 찾기 오류: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"파일 로드 중 오류 발생: {str(e)}")
            raise ValueError(f"파일 로드 중 오류 발생: {str(e)}")
    
    def _optimize_dataframe(self):
        """데이터프레임 메모리 사용량 최적화"""
        if self.df is None or self.df.empty:
            return
        
        # 메모리 사용량 확인 전
        mem_usage_before = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # 열 데이터 타입 최적화
        for col in self.df.columns:
            # 정수형 열
            if pd.api.types.is_integer_dtype(self.df[col]):
                # 값의 범위에 따라 더 작은 정수 타입 사용
                col_min, col_max = self.df[col].min(), self.df[col].max()
                
                if col_min >= 0:  # 양수
                    if col_max < 255:
                        self.df[col] = self.df[col].astype(np.uint8)
                    elif col_max < 65535:
                        self.df[col] = self.df[col].astype(np.uint16)
                    elif col_max < 4294967295:
                        self.df[col] = self.df[col].astype(np.uint32)
                else:  # 음수/양수
                    if col_min > -128 and col_max < 127:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif col_min > -32768 and col_max < 32767:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif col_min > -2147483648 and col_max < 2147483647:
                        self.df[col] = self.df[col].astype(np.int32)
                    
            # 부동소수점 열
            elif pd.api.types.is_float_dtype(self.df[col]):
                # 작은 부동소수점 유형 사용
                self.df[col] = self.df[col].astype(np.float32)
                
            # 문자열 열
            elif pd.api.types.is_object_dtype(self.df[col]):
                # 고유 값이 적은 경우 범주형으로 변환
                unique_count = self.df[col].nunique()
                unique_ratio = unique_count / len(self.df) if len(self.df) > 0 else 0
                
                if unique_ratio < 0.5 and unique_count > 0:  # 고유 값이 전체 데이터의 50% 미만인 경우
                    self.df[col] = self.df[col].astype('category')
        
        # 메모리 사용량 확인 후
        mem_usage_after = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        memory_saved = mem_usage_before - mem_usage_after
        
        # 메모리 최적화 정보 기록
        self.file_info['memory_before_optimization'] = round(mem_usage_before, 2)
        self.file_info['memory_after_optimization'] = round(mem_usage_after, 2)
        self.file_info['memory_saved'] = round(memory_saved, 2)
        self.file_info['memory_reduction_percent'] = round((memory_saved / mem_usage_before) * 100, 2) if mem_usage_before > 0 else 0
        
        logger.info(f"데이터프레임 최적화 완료: {mem_usage_before:.2f}MB → {mem_usage_after:.2f}MB ({memory_saved:.2f}MB 절약)")
    
    def _detect_column_types(self):
        """
        데이터프레임의 각 열에 대한 데이터 유형을 감지합니다.
        """
        if self.df is None:
            return
        
        self.column_types = {}
        
        for col in self.df.columns:
            col_type = self.df[col].dtype
            
            # 기본 데이터 유형 카테고리 지정
            if pd.api.types.is_numeric_dtype(col_type):
                if pd.api.types.is_integer_dtype(col_type) or col_type == np.int64:
                    category = "integer"
                elif pd.api.types.is_float_dtype(col_type) or col_type == np.float64:
                    category = "float"
                else:
                    category = "numeric"
            elif pd.api.types.is_datetime64_dtype(col_type):
                category = "datetime"
            elif pd.api.types.is_timedelta64_dtype(col_type):
                category = "timedelta"
            elif isinstance(col_type, pd.CategoricalDtype):  # is_categorical_dtype 대신 직접 타입 검사
                category = "categorical"
            elif pd.api.types.is_object_dtype(col_type):
                # 객체 유형에 대한 추가 분석
                unique_values = self.df[col].nunique()
                sample_values = self.df[col].dropna().head(5).tolist()
                
                if unique_values <= min(10, len(self.df) // 10):
                    category = "categorical"
                else:
                    category = "text"
            else:
                category = "other"
            
            # 결측치 정보 추가
            missing_count = self.df[col].isnull().sum()
            missing_percent = missing_count / len(self.df) * 100 if len(self.df) > 0 else 0
            
            # 고유값 정보 추가
            unique_count = self.df[col].nunique()
            unique_percent = unique_count / len(self.df) * 100 if len(self.df) > 0 else 0
            
            # 열 유형 정보 저장
            self.column_types[col] = {
                'dtype': str(col_type),
                'category': category,
                'missing_count': missing_count,
                'missing_percent': round(missing_percent, 2),
                'unique_count': unique_count,
                'unique_percent': round(unique_percent, 2)
            }
    
    def get_file_info(self):
        """
        현재 로드된 파일의 정보를 반환합니다.
        
        Returns:
        --------
        dict
            파일 정보
        """
        return self.file_info
    
    def get_column_types(self):
        """
        현재 로드된 데이터의 열 유형 정보를 반환합니다.
        
        Returns:
        --------
        dict
            열 유형 정보
        """
        return self.column_types
    
    def get_sheet_names(self, file_path=None):
        """
        엑셀 파일의 시트 이름 목록을 반환합니다.
        
        Parameters:
        -----------
        file_path : str or Path, optional
            확인할 파일 경로. None인 경우 현재 로드된 파일 사용
            
        Returns:
        --------
        list
            시트 이름 목록
        """
        try:
            if file_path is None:
                if self.current_file is None:
                    raise ValueError("로드된 파일이 없습니다.")
                file_path = self.current_file
            
            # 문자열 경로를 Path 객체로 변환
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            # 엑셀 파일 시트 이름 읽기 - 리소스를 명시적으로 닫기
            with pd.ExcelFile(file_path) as xls:
                return xls.sheet_names
        except Exception as e:
            logger.error(f"시트 이름을 가져오는 중 오류 발생: {str(e)}")
            return []
    
    def get_preview(self, max_rows=5):
        """
        현재 로드된 데이터의 미리보기를 반환합니다.
        
        Parameters:
        -----------
        max_rows : int, optional
            미리보기 행 수 (기본값: 5)
            
        Returns:
        --------
        pandas.DataFrame
            미리보기 데이터
        """
        if self.df is None:
            return None
        
        return self.df.head(max_rows)
    
    def get_summary(self):
        """
        현재 로드된 데이터의 기본 요약 정보를 반환합니다.
        
        Returns:
        --------
        dict
            요약 정보
        """
        if self.df is None:
            return {}
        
        # 기본 정보
        summary = {
            'file_info': self.get_file_info(),
            'row_count': len(self.df),
            'column_count': len(self.df.columns),
            'columns': list(self.df.columns),
            'column_types': self.get_column_types(),
            'missing_values': self.df.isnull().sum().sum(),
            'missing_percent': round(self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100, 2) if len(self.df) > 0 and len(self.df.columns) > 0 else 0
        }
        
        # 수치형 열의 요약 통계
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        if len(numeric_cols) > 0:
            summary['numeric_columns'] = {
                'count': len(numeric_cols),
                'names': list(numeric_cols)
            }
        
        # 범주형 열의 요약 통계
        categorical_cols = [col for col, info in self.column_types.items() if info['category'] in ['categorical', 'text']]
        if len(categorical_cols) > 0:
            summary['categorical_columns'] = {
                'count': len(categorical_cols),
                'names': list(categorical_cols)
            }
        
        # 날짜/시간 열의 요약 통계
        datetime_cols = [col for col, info in self.column_types.items() if info['category'] in ['datetime', 'timedelta']]
        if len(datetime_cols) > 0:
            summary['datetime_columns'] = {
                'count': len(datetime_cols),
                'names': list(datetime_cols)
            }
        
        return summary 