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
from typing import Optional, List, Dict, Union, Tuple, Any

# 로거 설정
logger = logging.getLogger(__name__)

class DataLoader:
    """
    다양한 형태의 엑셀/CSV 파일을 로드하고 데이터 형식을 자동으로 감지하는 클래스
    대용량 파일 처리를 위한 최적화 기능 포함
    """
    
    def __init__(self):
        """
        DataLoader 클래스 초기화
        """
        # 프로젝트 루트 디렉토리 및 데이터 디렉토리 설정
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.data_dir = self.project_root / 'data'
        
        # 상태 변수 초기화
        self.current_file = None
        self.df = None
        self.original_df = None
        self.file_info = {}
        self.column_types = {}
        
    def list_data_files(self, pattern: str = '*.*') -> List[Path]:
        """
        지정된 디렉토리에서 패턴과 일치하는 파일 목록을 반환합니다.
        
        Args:
            pattern (str): 검색할 파일 패턴 (기본값: '*.*')
            
        Returns:
            List[Path]: 파일 경로 목록
        """
        try:
            if not self.data_dir.exists():
                logger.warning(f"데이터 디렉토리가 존재하지 않습니다: {self.data_dir}")
                return []
            
            # 지정된 패턴과 일치하는 파일 목록 반환
            files = list(self.data_dir.glob(pattern))
            
            # 임시 파일 필터링 (엑셀에서 생성하는 임시 파일 제외)
            files = [f for f in files if not f.name.startswith('~$')]
            
            # 지원하는 파일 형식 필터링
            supported_extensions = {'.csv', '.xlsx', '.xls'}
            files = [f for f in files if f.suffix.lower() in supported_extensions]
            
            return files
        except Exception as e:
            logger.error(f"파일 목록을 가져오는 중 오류 발생: {str(e)}")
            return []
    
    def load_file(self, file_path: Optional[Union[str, Path]] = None, sheet_name: Union[str, int] = 0) -> pd.DataFrame:
        """
        지정된 엑셀/CSV 파일을 로드합니다.
        
        Args:
            file_path (str or Path, optional): 로드할 파일 경로
            sheet_name (str or int, optional): 로드할 시트 이름 또는 인덱스 (기본값: 0)
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
            
        Raises:
            FileNotFoundError: 파일을 찾을 수 없는 경우
            ValueError: 파일 로드 중 오류가 발생한 경우
        """
        try:
            # 파일 경로가 None인 경우 첫 번째 데이터 파일 사용
            if file_path is None:
                files = self.list_data_files()
                if not files:
                    raise FileNotFoundError("데이터 디렉토리에 지원하는 파일이 없습니다.")
                file_path = files[0]
            
            # 문자열 경로를 Path 객체로 통일
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            # 파일 존재 확인
            if not file_path.exists():
                raise FileNotFoundError(f"파일이 존재하지 않습니다: {file_path}")
                
            # 파일 크기 확인 (MB 단위)
            file_size_bytes = file_path.stat().st_size
            file_size_mb = file_size_bytes / (1024 * 1024)
            use_efficient_loading = file_size_mb > 100  # 100MB 이상이면 효율적 로딩 사용
            
            # 로딩 시작 시간 기록
            start_time = datetime.now()
            extension = file_path.suffix.lower()
            
            # 파일 타입에 따라 다른 로딩 방식 사용
            if extension == '.csv':
                self.df = self._load_csv(file_path, use_efficient_loading)
                file_format = 'CSV'
            elif extension in ['.xlsx', '.xls']:
                self.df = self._load_excel(file_path, sheet_name, use_efficient_loading)
                file_format = 'Excel'
            else:
                raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}")
            
            # 로드 완료 시간 및 소요 시간 계산
            end_time = datetime.now()
            load_time = (end_time - start_time).total_seconds()
            
            # 원본 데이터프레임 백업 (필요시 복원용)
            self.original_df = self.df.copy() if self.df is not None else None
            
            # NaN 값 개수 계산
            nan_count = self.df.isna().sum().sum() if self.df is not None else 0
            
            # 파일 정보 저장
            self.current_file = file_path
            self.file_info = {
                'file_path': str(file_path),
                'file_name': file_path.name,
                'file_size': file_size_bytes,
                'file_size_mb': round(file_size_mb, 2),
                'format': file_format,
                'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'sheet_name': sheet_name,
                'load_time_seconds': load_time,
                'row_count': len(self.df) if self.df is not None else 0,
                'column_count': len(self.df.columns) if self.df is not None else 0,
                'nan_count': nan_count
            }
            
            # 메모리 최적화
            self._optimize_dataframe()
            
            # 열 데이터 유형 감지
            self._detect_column_types()
            
            logger.info(f"파일 로드 완료: {file_path.name} ({len(self.df)} 행, {len(self.df.columns)} 열, {load_time:.2f}초)")
            return self.df
            
        except FileNotFoundError as e:
            logger.error(f"파일 찾기 오류: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"파일 로드 중 오류 발생: {str(e)}")
            raise ValueError(f"파일 로드 중 오류 발생: {str(e)}")
    
    def _load_csv(self, file_path: Path, use_efficient_loading: bool) -> pd.DataFrame:
        """
        CSV 파일을 로드하는 내부 메서드
        
        Args:
            file_path (Path): CSV 파일 경로
            use_efficient_loading (bool): 효율적인 로딩 사용 여부
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
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
            logger.info(f"대용량 CSV 파일 로드 중: {file_path.name}")
            
            for chunk in pd.read_csv(file_path, dtype=dtypes, chunksize=50000):
                chunks.append(chunk)
            
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(file_path)
    
    def _load_excel(self, file_path: Path, sheet_name: Union[str, int], use_efficient_loading: bool) -> pd.DataFrame:
        """
        Excel 파일을 로드하는 내부 메서드
        
        Args:
            file_path (Path): Excel 파일 경로
            sheet_name (str or int): 로드할 시트 이름 또는 인덱스
            use_efficient_loading (bool): 효율적인 로딩 사용 여부
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        # 엑셀 엔진 선택
        excel_engine = 'openpyxl' if file_path.suffix.lower() == '.xlsx' else 'xlrd'
        
        # 시트 이름 목록 가져오기
        with pd.ExcelFile(file_path, engine=excel_engine) as xl:
            available_sheets = xl.sheet_names
            self.file_info['available_sheets'] = available_sheets
        
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
        return pd.read_excel(
            file_path, 
            sheet_name=sheet_to_load,
            engine=excel_engine
        )
    
    def _optimize_dataframe(self) -> None:
        """
        데이터프레임 메모리 사용량 최적화
        더 효율적인 데이터 타입으로 변환하여 메모리 사용량 감소
        """
        if self.df is None or self.df.empty:
            return
        
        # 메모리 사용량 확인 전
        mem_usage_before = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        # 열 데이터 타입 최적화 수행
        self._optimize_numeric_columns()
        self._optimize_object_columns()
        self._optimize_datetime_columns()
        
        # 메모리 사용량 확인 후
        mem_usage_after = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        memory_saved = mem_usage_before - mem_usage_after
        
        # 메모리 최적화 정보 기록
        self.file_info['memory_before_optimization'] = round(mem_usage_before, 2)
        self.file_info['memory_after_optimization'] = round(mem_usage_after, 2)
        self.file_info['memory_saved'] = round(memory_saved, 2)
        self.file_info['memory_reduction_percent'] = round((memory_saved / mem_usage_before) * 100, 2) if mem_usage_before > 0 else 0
        
        logger.info(f"데이터프레임 최적화 완료: {mem_usage_before:.2f}MB → {mem_usage_after:.2f}MB ({memory_saved:.2f}MB 절약)")
    
    def _optimize_numeric_columns(self) -> None:
        """
        수치형 열(정수형 및 부동소수점)의 데이터 타입을 최적화합니다.
        """
        for col in self.df.columns:
            # 결측값 확인
            has_null = self.df[col].isna().any()
            
            # 정수형 열 최적화
            if pd.api.types.is_integer_dtype(self.df[col]):
                self._optimize_integer_column(col, has_null)
            # 부동소수점 열 최적화
            elif pd.api.types.is_float_dtype(self.df[col]):
                self._optimize_float_column(col, has_null)
    
    def _optimize_integer_column(self, col: str, has_null: bool) -> None:
        """
        정수형 열의 데이터 타입을 최적화합니다.
        
        Args:
            col (str): 열 이름
            has_null (bool): 결측값 존재 여부
        """
        # 결측값이 있는 경우 Pandas의 nullable 정수 타입 사용
        if has_null:
            self.df[col] = self.df[col].astype('Int64')
            return
            
        # 값의 범위에 따라 더 작은 정수 타입 사용
        col_min, col_max = self.df[col].min(), self.df[col].max()
        
        # 양수만 있는 경우 (unsigned)
        if col_min >= 0:
            if col_max < 255:
                self.df[col] = self.df[col].astype(np.uint8)
            elif col_max < 65535:
                self.df[col] = self.df[col].astype(np.uint16)
            elif col_max < 4294967295:
                self.df[col] = self.df[col].astype(np.uint32)
        # 음수/양수 모두 있는 경우 (signed)
        else:
            if col_min > -128 and col_max < 127:
                self.df[col] = self.df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                self.df[col] = self.df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                self.df[col] = self.df[col].astype(np.int32)
    
    def _optimize_float_column(self, col: str, has_null: bool) -> None:
        """
        부동소수점 열의 데이터 타입을 최적화합니다.
        
        Args:
            col (str): 열 이름
            has_null (bool): 결측값 존재 여부
        """
        # 결측값이 있더라도 정밀도 손실이 문제가 되지 않으면 float32로 변환
        # float32는 대략 7자리의 정밀도를 가지며, 메모리 사용량을 절반으로 줄일 수 있음
        
        # 먼저 데이터가 float32 범위 내에 있는지 확인
        if has_null:
            non_null_values = self.df[col].dropna()
            if non_null_values.empty:
                return
                
            col_min, col_max = non_null_values.min(), non_null_values.max()
        else:
            col_min, col_max = self.df[col].min(), self.df[col].max()
        
        # float32 범위: 약 ±3.4e38 with 7 significant digits
        # 대부분의 경우 이 범위 내에 있음
        float32_min, float32_max = np.finfo(np.float32).min, np.finfo(np.float32).max
        
        if col_min > float32_min and col_max < float32_max:
            # 높은 정밀도가 필요한지 확인 (소수점 이하 7자리 이상의 정밀도가 필요하면 float64 유지)
            # 간단하게 문자열 표현을 통해 확인
            sample = self.df[col].dropna().sample(min(100, len(self.df[col].dropna()))).astype(str)
            
            # 소수점 이하 자릿수 확인
            max_decimal_places = 0
            for val in sample:
                if '.' in val:
                    decimal_places = len(val.split('.')[1])
                    max_decimal_places = max(max_decimal_places, decimal_places)
            
            # 정밀도가 7자리 이하면 float32로 변환 (경험적 규칙)
            if max_decimal_places <= 7:
                # float32로 변환
                self.df[col] = self.df[col].astype(np.float32)
    
    def _optimize_object_columns(self) -> None:
        """
        문자열(object) 열의 데이터 타입을 최적화합니다.
        """
        for col in self.df.columns:
            # object 타입 또는 string 타입 열에 대해서만 처리
            if not (pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_string_dtype(self.df[col])):
                continue
                
            # 모든 값이 NaN인 경우 최적화하지 않음
            if self.df[col].isna().all():
                continue
                
            # 고유값 수 확인
            unique_count = self.df[col].nunique()
            
            # 데이터가 없으면 스킵
            if unique_count == 0:
                continue
                
            # 고유값 비율 계산
            unique_ratio = unique_count / len(self.df[col])
            
            # 특수 케이스: 불리언 문자열 확인 ('true', 'false', 'yes', 'no' 등)
            if unique_count <= 2:
                lowercase_values = self.df[col].dropna().astype(str).str.lower()
                bool_values = {'true', 'false', 'yes', 'no', 't', 'f', 'y', 'n', '1', '0'}
                
                # 값의 집합이 bool_values의 부분집합인지 확인
                if set(lowercase_values.unique()).issubset(bool_values):
                    # 불리언으로 변환
                    true_values = {'true', 'yes', 't', 'y', '1'}
                    self.df[col] = lowercase_values.isin(true_values)
                    continue
            
            # 고유값 비율이 낮거나 고유값 개수가 적은 경우 category 타입으로 변환
            if (unique_ratio < 0.5 and unique_count > 0) or (0 < unique_count < 100):
                self.df[col] = self.df[col].astype('category')
            else:
                # pandas 1.0.0 이상에서 지원하는 string 데이터 타입 사용 고려
                try:
                    # 텍스트 길이 확인
                    avg_len = self.df[col].astype(str).str.len().mean()
                    
                    # 짧은 문자열이면 string 타입 사용 시도 (pandas 1.0.0+)
                    if avg_len < 50:
                        self.df[col] = self.df[col].astype('string')
                except:
                    # 'string' 타입이 지원되지 않는 경우 object 타입 유지
                    pass
    
    def _optimize_datetime_columns(self) -> None:
        """
        날짜/시간 열을 최적화합니다.
        """
        for col in self.df.columns:
            # 이미 datetime 타입인 경우
            if pd.api.types.is_datetime64_dtype(self.df[col]):
                continue
                
            # object/string 타입인 경우 날짜/시간으로 변환 시도
            if pd.api.types.is_object_dtype(self.df[col]) or pd.api.types.is_string_dtype(self.df[col]):
                # NA가 아닌 샘플 추출
                sample = self.df[col].dropna().head(100).astype(str)
                
                # 샘플이 없으면 스킵
                if len(sample) == 0:
                    continue
                    
                # 날짜/시간 패턴 확인
                date_count = 0
                for val in sample:
                    try:
                        pd.to_datetime(val)
                        date_count += 1
                    except:
                        pass
                
                # 70% 이상이 날짜/시간으로 변환 가능하면 변환 시도
                if date_count >= len(sample) * 0.7:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                        logger.info(f"열 '{col}': 날짜/시간 타입으로 변환")
                    except:
                        logger.warning(f"열 '{col}': 날짜/시간 타입 변환 실패")
    
    def _detect_column_types(self) -> None:
        """
        데이터프레임의 열 유형을 감지하고 분류합니다.
        """
        if self.df is None or self.df.empty:
            self.column_types = {}
            return
        
        # 열 유형 분류
        numeric_cols = []
        categorical_cols = []
        datetime_cols = []
        text_cols = []
        boolean_cols = []
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            
            # Pandas 1.0.0 이상에서는 dtype.name 대신 dtype을 문자열로 변환
            dtype_str = str(dtype)
            
            if pd.api.types.is_numeric_dtype(dtype):
                numeric_cols.append(col)
            elif isinstance(dtype, pd.CategoricalDtype):
                categorical_cols.append(col)
            elif pd.api.types.is_datetime64_dtype(dtype) or 'datetime' in dtype_str:
                datetime_cols.append(col)
            elif pd.api.types.is_bool_dtype(dtype):
                boolean_cols.append(col)
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
                # 문자열로 변환했을 때 평균 길이를 확인
                if self.df[col].notna().any():  # NA가 아닌 값이 있는 경우만
                    avg_len = self.df[col].astype(str).str.len().mean()
                    if avg_len > 50:  # 평균 길이가 50자 이상이면 텍스트로 분류
                        text_cols.append(col)
                    else:
                        categorical_cols.append(col)
                else:
                    categorical_cols.append(col)
            else:
                # 그 외 타입은 일단 범주형으로 분류
                categorical_cols.append(col)
        
        # 결과 저장
        self.column_types = {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols,
            'text': text_cols,
            'boolean': boolean_cols
        }
        
        # 각 열에 대한 추가 정보 수집
        col_info = {}
        for col in self.df.columns:
            info = {
                'dtype': str(self.df[col].dtype),
                'null_count': self.df[col].isna().sum(),
                'null_percent': round(self.df[col].isna().mean() * 100, 2),
            }
            
            # 데이터 타입별 추가 정보
            if col in numeric_cols and self.df[col].notna().any():
                info.update({
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'mean': self.df[col].mean(),
                    'median': self.df[col].median()
                })
            elif col in categorical_cols:
                info.update({
                    'unique_count': self.df[col].nunique(),
                    'unique_ratio': round(self.df[col].nunique() / len(self.df) * 100, 2) if len(self.df) > 0 else 0
                })
            
            col_info[col] = info
        
        self.column_types['column_info'] = col_info
        
        logger.info(f"열 유형 감지 완료: {len(numeric_cols)} 수치, {len(categorical_cols)} 범주, {len(datetime_cols)} 날짜, {len(text_cols)} 텍스트, {len(boolean_cols)} 불리언")
    
    def get_file_info(self) -> Dict[str, Any]:
        """
        현재 로드된 파일의 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 파일 정보 딕셔너리
        """
        return self.file_info
    
    def get_column_types(self) -> Dict[str, List[str]]:
        """
        현재 로드된 데이터프레임의 열 유형 분류를 반환합니다.
        
        Returns:
            Dict[str, List[str]]: 열 유형 분류 딕셔너리
        """
        return self.column_types
    
    def get_sheet_names(self, file_path: Optional[Union[str, Path]] = None) -> List[str]:
        """
        엑셀 파일의 시트 이름 목록을 반환합니다.
        
        Args:
            file_path (str or Path, optional): 엑셀 파일 경로. None인 경우 현재 로드된 파일 사용
            
        Returns:
            List[str]: 시트 이름 목록
        """
        try:
            # 파일 경로가 None인 경우 현재 로드된 파일 사용
            if file_path is None:
                if self.current_file is None:
                    logger.warning("현재 로드된 파일이 없습니다.")
                    return []
                file_path = self.current_file
            
            # 문자열 경로를 Path 객체로 변환
            if isinstance(file_path, str):
                file_path = Path(file_path)
            
            # 파일 확장자 확인
            extension = file_path.suffix.lower()
            if extension not in ['.xlsx', '.xls']:
                logger.warning(f"엑셀 파일이 아닙니다: {extension}")
                return []
            
            # 엑셀 엔진 선택
            excel_engine = 'openpyxl' if extension == '.xlsx' else 'xlrd'
            
            # 시트 이름 목록 반환
            with pd.ExcelFile(file_path, engine=excel_engine) as xl:
                return xl.sheet_names
                
        except Exception as e:
            logger.error(f"시트 이름 가져오기 오류: {str(e)}")
            return []
    
    def get_preview(self, max_rows: int = 5) -> pd.DataFrame:
        """
        데이터프레임의 미리보기를 반환합니다.
        
        Args:
            max_rows (int, optional): 미리보기 행 수 (기본값: 5)
            
        Returns:
            pd.DataFrame: 미리보기 데이터프레임
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()
            
        return self.df.head(max_rows)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        현재 로드된 데이터의 요약 정보를 반환합니다.
        
        Returns:
            Dict[str, Any]: 데이터 요약 정보
        """
        if self.df is None:
            return {'status': 'no_data'}
            
        try:
            # 기본 정보
            summary = {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'column_names': list(self.df.columns),
                'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
                'memory_usage_mb': round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
                'null_counts': {col: int(count) for col, count in self.df.isna().sum().items()},
                'null_percent': {col: round(count / len(self.df) * 100, 2) for col, count in self.df.isna().sum().items()},
            }
            
            # 열 유형별 분류
            if self.column_types:
                summary['column_types'] = {k: v for k, v in self.column_types.items() if k != 'column_info'}
            
            # 파일 정보
            if self.file_info:
                summary['file_info'] = self.file_info
            
            return summary
            
        except Exception as e:
            logger.error(f"요약 정보 생성 오류: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def restore_original(self) -> pd.DataFrame:
        """
        원본 데이터프레임으로 복원합니다.
        
        Returns:
            pd.DataFrame: 복원된 원본 데이터프레임
        """
        if self.original_df is not None:
            self.df = self.original_df.copy()
            logger.info("원본 데이터프레임으로 복원되었습니다.")
            return self.df
        else:
            logger.warning("복원할 원본 데이터프레임이 없습니다.")
            return self.df if self.df is not None else pd.DataFrame() 