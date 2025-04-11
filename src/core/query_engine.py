import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable
import logging
import re

class QueryEngine:
    """
    데이터에 대한 복잡한 질의를 처리하는 엔진 클래스
    SQL 스타일의 쿼리, 필터링, 조건부 집계 등 다양한 질의 기능 제공
    """
    
    def __init__(self):
        """QueryEngine 초기화"""
        self.df = None
        self.query_history = []
        self.current_result = None
        self.logger = logging.getLogger(__name__)
        
    def set_dataframe(self, df: pd.DataFrame) -> None:
        """
        분석할 데이터프레임 설정
        
        Args:
            df (pd.DataFrame): 분석 대상 데이터프레임
        """
        self.df = df
        self.current_result = df.copy()
        self.logger.info(f"데이터프레임 설정 완료 (행: {df.shape[0]}, 열: {df.shape[1]})")
        
    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """현재 데이터프레임 반환"""
        return self.df
        
    def get_current_result(self) -> Optional[pd.DataFrame]:
        """현재 쿼리 결과 반환"""
        return self.current_result
        
    def reset_query(self) -> None:
        """쿼리 상태 초기화"""
        if self.df is not None:
            self.current_result = self.df.copy()
        else:
            self.current_result = None
            
        self.logger.info("쿼리 상태가 초기화되었습니다.")
        
    def query(self, query_str: str) -> pd.DataFrame:
        """
        pandas query 문법을 사용한 데이터 필터링
        
        Args:
            query_str (str): 쿼리 문자열 (예: "age > 30 and category == 'A'")
            
        Returns:
            pd.DataFrame: 쿼리 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        try:
            result = self.current_result.query(query_str)
            self.current_result = result
            self.query_history.append(f"query: {query_str}")
            self.logger.info(f"쿼리 실행 완료: {query_str}, 결과 행 수: {len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"쿼리 실행 중 오류 발생: {str(e)}")
            return self.current_result
            
    def select_columns(self, columns: List[str]) -> pd.DataFrame:
        """
        특정 열만 선택
        
        Args:
            columns (List[str]): 선택할 열 목록
            
        Returns:
            pd.DataFrame: 선택된 열만 포함하는 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        # 유효한 열만 필터링
        valid_columns = [col for col in columns if col in self.current_result.columns]
        
        if not valid_columns:
            self.logger.warning("유효한 열이 지정되지 않았습니다.")
            return self.current_result
            
        try:
            result = self.current_result[valid_columns]
            self.current_result = result
            self.query_history.append(f"select_columns: {valid_columns}")
            self.logger.info(f"열 선택 완료: {valid_columns}")
            return result
        except Exception as e:
            self.logger.error(f"열 선택 중 오류 발생: {str(e)}")
            return self.current_result
            
    def filter_by_value(self, column: str, values: List[Any], 
                       include: bool = True) -> pd.DataFrame:
        """
        특정 값 목록으로 데이터 필터링
        
        Args:
            column (str): 필터링할 열
            values (List[Any]): 필터링 값 목록
            include (bool): True인 경우 포함, False인 경우 제외
            
        Returns:
            pd.DataFrame: 필터링된 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        if column not in self.current_result.columns:
            self.logger.warning(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
            return self.current_result
            
        try:
            if include:
                result = self.current_result[self.current_result[column].isin(values)]
                operation = "포함"
            else:
                result = self.current_result[~self.current_result[column].isin(values)]
                operation = "제외"
                
            self.current_result = result
            self.query_history.append(f"filter_by_value: {column}, {values}, {operation}")
            self.logger.info(f"값 필터링 완료: {column} {operation} {values}, 결과 행 수: {len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"값 필터링 중 오류 발생: {str(e)}")
            return self.current_result
            
    def filter_by_range(self, column: str, min_value: Optional[Any] = None, 
                       max_value: Optional[Any] = None, 
                       inclusive: str = "both") -> pd.DataFrame:
        """
        값 범위로 데이터 필터링
        
        Args:
            column (str): 필터링할 열
            min_value (Any, optional): 최소값
            max_value (Any, optional): 최대값
            inclusive (str): 범위 포함 방식 ('both', 'left', 'right', 'neither')
            
        Returns:
            pd.DataFrame: 필터링된 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        if column not in self.current_result.columns:
            self.logger.warning(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
            return self.current_result
            
        # 최소값과 최대값이 모두 None인 경우
        if min_value is None and max_value is None:
            self.logger.warning("최소값과 최대값이 모두 지정되지 않았습니다.")
            return self.current_result
            
        try:
            # 데이터 타입 확인 및 처리
            col_data = self.current_result[column]
            
            # 필터 마스크 생성
            mask = pd.Series(True, index=col_data.index)
            
            if min_value is not None:
                if inclusive in ["both", "left"]:
                    min_mask = col_data >= min_value
                else:
                    min_mask = col_data > min_value
                mask = mask & min_mask
                
            if max_value is not None:
                if inclusive in ["both", "right"]:
                    max_mask = col_data <= max_value
                else:
                    max_mask = col_data < max_value
                mask = mask & max_mask
                
            # 필터링 적용
            result = self.current_result[mask]
            self.current_result = result
            
            # 쿼리 히스토리 업데이트
            range_str = f"{min_value if min_value is not None else ''} ~ {max_value if max_value is not None else ''}"
            self.query_history.append(f"filter_by_range: {column}, {range_str}, inclusive={inclusive}")
            
            self.logger.info(f"범위 필터링 완료: {column} in {range_str}, 결과 행 수: {len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"범위 필터링 중 오류 발생: {str(e)}")
            return self.current_result
            
    def sort_values(self, columns: List[str], ascending: Union[bool, List[bool]] = True) -> pd.DataFrame:
        """
        지정된 열로 데이터 정렬
        
        Args:
            columns (List[str]): 정렬 기준 열 목록
            ascending (Union[bool, List[bool]]): 오름차순 여부
            
        Returns:
            pd.DataFrame: 정렬된 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        # 유효한 열만 필터링
        valid_columns = [col for col in columns if col in self.current_result.columns]
        
        if not valid_columns:
            self.logger.warning("유효한 정렬 열이 지정되지 않았습니다.")
            return self.current_result
            
        try:
            result = self.current_result.sort_values(by=valid_columns, ascending=ascending)
            self.current_result = result
            
            asc_desc = "오름차순" if ascending else "내림차순"
            if isinstance(ascending, list):
                asc_desc = f"custom({ascending})"
                
            self.query_history.append(f"sort_values: {valid_columns}, {asc_desc}")
            self.logger.info(f"정렬 완료: {valid_columns}, {asc_desc}")
            return result
        except Exception as e:
            self.logger.error(f"정렬 중 오류 발생: {str(e)}")
            return self.current_result
            
    def group_and_aggregate(self, group_by: Union[str, List[str]], 
                           agg_dict: Dict[str, Union[str, List[str]]]) -> pd.DataFrame:
        """
        그룹별 집계 수행
        
        Args:
            group_by (Union[str, List[str]]): 그룹화 기준 열
            agg_dict (Dict[str, Union[str, List[str]]]): 열별 집계 함수 딕셔너리
                                                      예: {'age': 'mean', 'salary': ['sum', 'mean']}
            
        Returns:
            pd.DataFrame: 그룹별 집계 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        # 그룹화 기준 열 확인
        if isinstance(group_by, str):
            group_by = [group_by]
            
        valid_group_cols = [col for col in group_by if col in self.current_result.columns]
        
        if not valid_group_cols:
            self.logger.warning("유효한 그룹화 열이 지정되지 않았습니다.")
            return self.current_result
            
        # 유효한 집계 열 및 함수 확인
        valid_agg_dict = {}
        for col, funcs in agg_dict.items():
            if col in self.current_result.columns:
                valid_agg_dict[col] = funcs
                
        if not valid_agg_dict:
            self.logger.warning("유효한 집계 열이 지정되지 않았습니다.")
            return self.current_result
            
        try:
            # 그룹별 집계 수행
            result = self.current_result.groupby(valid_group_cols).agg(valid_agg_dict)
            
            # 결과가 비어있지 않은 경우 (빈 데이터프레임을 반환하는 대신)
            if not result.empty:
                self.current_result = result
                self.query_history.append(f"group_and_aggregate: {valid_group_cols}, {valid_agg_dict}")
                self.logger.info(f"그룹별 집계 완료: {valid_group_cols}, {valid_agg_dict}")
            else:
                self.logger.warning("그룹별 집계 결과가 비어있습니다.")
                
            return result
        except Exception as e:
            self.logger.error(f"그룹별 집계 중 오류 발생: {str(e)}")
            return self.current_result
            
    def apply_function(self, func: Callable, axis: int = 0) -> pd.DataFrame:
        """
        커스텀 함수 적용
        
        Args:
            func (Callable): 적용할 함수
            axis (int): 적용 축 (0: 행, 1: 열)
            
        Returns:
            pd.DataFrame: 함수 적용 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        try:
            result = self.current_result.apply(func, axis=axis)
            
            # 결과가 Series나 DataFrame인 경우에만 current_result 업데이트
            if isinstance(result, (pd.Series, pd.DataFrame)):
                self.current_result = result
                self.query_history.append(f"apply_function: {func.__name__}, axis={axis}")
                self.logger.info(f"함수 적용 완료: {func.__name__}, axis={axis}")
            else:
                self.logger.warning(f"함수 적용 결과가 Series나 DataFrame이 아닙니다: {type(result)}")
                
            return result
        except Exception as e:
            self.logger.error(f"함수 적용 중 오류 발생: {str(e)}")
            return self.current_result
            
    def pivot_table(self, index: Union[str, List[str]], 
                   columns: Optional[Union[str, List[str]]] = None,
                   values: Optional[Union[str, List[str]]] = None,
                   aggfunc: Union[str, List[str], Dict] = 'mean') -> pd.DataFrame:
        """
        피벗 테이블 생성
        
        Args:
            index (Union[str, List[str]]): 피벗 테이블의 행 인덱스
            columns (Union[str, List[str]], optional): 피벗 테이블의 열 인덱스
            values (Union[str, List[str]], optional): 집계할 값 열
            aggfunc (Union[str, List[str], Dict]): 집계 함수
            
        Returns:
            pd.DataFrame: 피벗 테이블 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        try:
            result = pd.pivot_table(
                self.current_result, 
                index=index,
                columns=columns,
                values=values,
                aggfunc=aggfunc
            )
            
            self.current_result = result
            self.query_history.append(f"pivot_table: index={index}, columns={columns}, values={values}")
            self.logger.info(f"피벗 테이블 생성 완료: index={index}, columns={columns}, values={values}")
            return result
        except Exception as e:
            self.logger.error(f"피벗 테이블 생성 중 오류 발생: {str(e)}")
            return self.current_result
            
    def search_text(self, column: str, pattern: str, 
                   case: bool = False, regex: bool = True) -> pd.DataFrame:
        """
        텍스트 검색 및 필터링
        
        Args:
            column (str): 검색할 열
            pattern (str): 검색 패턴
            case (bool): 대소문자 구분 여부
            regex (bool): 정규식 사용 여부
            
        Returns:
            pd.DataFrame: 검색 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        if column not in self.current_result.columns:
            self.logger.warning(f"열 '{column}'이 데이터프레임에 존재하지 않습니다.")
            return self.current_result
            
        try:
            # 열 데이터 타입 확인
            if not pd.api.types.is_string_dtype(self.current_result[column]):
                # 문자열로 변환
                col_data = self.current_result[column].astype(str)
            else:
                col_data = self.current_result[column]
                
            # 텍스트 검색
            mask = col_data.str.contains(pattern, case=case, regex=regex, na=False)
            result = self.current_result[mask]
            
            self.current_result = result
            self.query_history.append(f"search_text: {column}, '{pattern}', case={case}, regex={regex}")
            self.logger.info(f"텍스트 검색 완료: {column}, '{pattern}', 결과 행 수: {len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"텍스트 검색 중 오류 발생: {str(e)}")
            return self.current_result
            
    def add_calculated_column(self, new_column: str, formula: str) -> pd.DataFrame:
        """
        계산식을 사용하여 새 열 추가
        
        Args:
            new_column (str): 새 열 이름
            formula (str): 계산식 (열 이름과 연산자 포함)
            
        Returns:
            pd.DataFrame: 새 열이 추가된 결과
        """
        if self.current_result is None:
            self.logger.error("데이터프레임이 설정되지 않았습니다.")
            return pd.DataFrame()
            
        try:
            # 열 이름 수집 (정규식 사용)
            col_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)'
            potential_cols = re.findall(col_pattern, formula)
            
            # 실제 데이터프레임에 있는 열만 필터링
            df_cols = {col for col in potential_cols if col in self.current_result.columns}
            
            # 계산식 평가
            try:
                # 로컬 변수로 각 열 데이터 설정
                local_vars = {}
                for col in df_cols:
                    local_vars[col] = self.current_result[col]
                    
                # numpy와 pandas 함수 접근 제공
                local_vars['np'] = np
                
                # 계산식 평가
                result_series = eval(formula, {"__builtins__": {}}, local_vars)
                
                # 새 열 추가
                result = self.current_result.copy()
                result[new_column] = result_series
                
                self.current_result = result
                self.query_history.append(f"add_calculated_column: {new_column} = {formula}")
                self.logger.info(f"계산 열 추가 완료: {new_column} = {formula}")
                return result
            except Exception as e:
                self.logger.error(f"계산식 평가 중 오류 발생: {str(e)}")
                return self.current_result
        except Exception as e:
            self.logger.error(f"계산 열 추가 중 오류 발생: {str(e)}")
            return self.current_result
            
    def get_query_history(self) -> List[str]:
        """쿼리 히스토리 반환"""
        return self.query_history
        
    def execute_sql(self, sql_query: str) -> pd.DataFrame:
        """
        SQL 스타일 쿼리 실행 (pandasql 필요)
        
        Args:
            sql_query (str): SQL 쿼리 문자열
            
        Returns:
            pd.DataFrame: 쿼리 결과
        """
        try:
            # pandasql 임포트 시도
            try:
                from pandasql import sqldf
            except ImportError:
                self.logger.error("pandasql 패키지가 설치되지 않았습니다. 'pip install pandasql'로 설치하세요.")
                return self.current_result
                
            # 현재 데이터프레임을 'df'로 참조할 수 있도록 설정
            result = sqldf(sql_query, locals())
            
            self.current_result = result
            self.query_history.append(f"execute_sql: {sql_query}")
            self.logger.info(f"SQL 쿼리 실행 완료: {sql_query}, 결과 행 수: {len(result)}")
            return result
        except Exception as e:
            self.logger.error(f"SQL 쿼리 실행 중 오류 발생: {str(e)}")
            return self.current_result 