import os
import pandas as pd
import numpy as np
import logging

class DataLoader:
    def __init__(self):
        self.df = None
        self.file_path = None
        self.file_info = {}
        self.logger = logging.getLogger(__name__)
        
    def load_file(self, file_path: str) -> pd.DataFrame:
        """
        파일을 로드하여 DataFrame으로 반환
        
        Args:
            file_path (str): 로드할 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        self.file_info = {}
        self.file_path = file_path
        
        try:
            # 파일 크기 확인
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            self.file_info['file_size'] = file_size
            
            # 대용량 파일일 경우 메모리 효율적인 로딩 사용
            use_efficient_loading = file_size > 100  # 100MB 이상이면 효율적 로딩
            
            extension = os.path.splitext(file_path)[1].lower()
            
            if extension == '.csv':
                # CSV 파일 로드
                if use_efficient_loading:
                    # 먼저 열 유형을 확인하기 위해 일부 행만 로드
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
                
                self.file_info['format'] = 'CSV'
                
            elif extension in ['.xlsx', '.xls']:
                # Excel 파일 로드
                if use_efficient_loading:
                    # 엑셀 엔진에 따라 다른 최적화 적용
                    excel_engine = 'openpyxl' if extension == '.xlsx' else 'xlrd'
                    
                    # 시트 이름 가져오기
                    xl = pd.ExcelFile(file_path, engine=excel_engine)
                    sheet_names = xl.sheet_names
                    
                    if len(sheet_names) > 0:
                        # 첫 번째 시트만 로드
                        self.df = pd.read_excel(
                            file_path, 
                            sheet_name=sheet_names[0],
                            engine=excel_engine
                        )
                        self.file_info['sheets'] = sheet_names
                    else:
                        self.df = pd.DataFrame()
                else:
                    # 일반 로딩
                    self.df = pd.read_excel(file_path)
                
                self.file_info['format'] = 'Excel'
                
            else:
                # 지원하지 않는 파일 형식
                raise ValueError(f"지원하지 않는 파일 형식입니다: {extension}")

            # NaN 값을 찾고 개수 계산
            nan_count = self.df.isna().sum().sum()
            self.file_info['nan_count'] = nan_count
            
            # 데이터프레임 최적화
            self.optimize_dataframe()
            
            # 파일 로드 정보 갱신
            self.file_info['rows'] = len(self.df)
            self.file_info['columns'] = len(self.df.columns)
            self.file_info['column_types'] = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
            
            self.logger.info(f"파일 로드 성공: {file_path} (행: {len(self.df)}, 열: {len(self.df.columns)})")
            return self.df
            
        except Exception as e:
            self.logger.error(f"파일 로드 실패: {file_path} - {str(e)}")
            raise
        
    def optimize_dataframe(self):
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
                unique_ratio = self.df[col].nunique() / len(self.df[col])
                
                if unique_ratio < 0.5:  # 고유 값이 전체 데이터의 50% 미만인 경우
                    self.df[col] = self.df[col].astype('category')
            
        # 메모리 사용량 확인 후
        mem_usage_after = self.df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        memory_saved = mem_usage_before - mem_usage_after
        
        # 메모리 최적화 정보 기록
        self.file_info['memory_before_optimization'] = mem_usage_before
        self.file_info['memory_after_optimization'] = mem_usage_after
        self.file_info['memory_saved'] = memory_saved
        self.file_info['memory_reduction_percent'] = (memory_saved / mem_usage_before) * 100 if mem_usage_before > 0 else 0
        
        self.logger.info(f"데이터프레임 최적화 완료: {mem_usage_before:.2f}MB → {mem_usage_after:.2f}MB ({memory_saved:.2f}MB 절약)") 