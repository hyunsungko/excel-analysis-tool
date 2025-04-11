import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ExcelAnalyzer:
    """
    Data 폴더에 있는 엑셀 파일을 읽고 분석하는 클래스
    """
    
    def __init__(self, file_path=None):
        """
        초기화 함수
        
        Parameters:
        -----------
        file_path : str, optional
            분석할 엑셀 파일 경로. 기본값은 None으로, 이 경우 Data 폴더의 첫 번째 엑셀 파일을 사용합니다.
        """
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
        
        if file_path is None:
            # Data 폴더에서 첫 번째 엑셀 파일 찾기
            excel_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.xlsx', '.xls'))]
            if not excel_files:
                raise FileNotFoundError("Data 폴더에 엑셀 파일이 없습니다.")
            self.file_path = os.path.join(self.data_dir, excel_files[0])
        else:
            self.file_path = file_path
            
        self.df = None
        self.analysis_results = {}
        print(f"분석할 파일: {os.path.basename(self.file_path)}")
        
    def load_data(self):
        """
        엑셀 파일을 읽어 DataFrame으로 변환
        
        Returns:
        --------
        pandas.DataFrame
            읽어온 데이터
        """
        try:
            # 엑셀 파일 로드
            self.df = pd.read_excel(self.file_path)
            print(f"데이터 로드 완료: {len(self.df)} 행, {len(self.df.columns)} 열")
            
            # 데이터 타입 자동 변환 시도
            print("데이터 타입 자동 변환 시작...")
            
            for col in self.df.columns:
                # 현재 열의 데이터 타입 확인
                orig_type = self.df[col].dtype
                
                # 이미 수치형이면 건너뛰기
                if np.issubdtype(orig_type, np.number):
                    continue
                    
                # 문자열이 숫자로 변환 가능한지 확인
                try:
                    # 빈 값이 아닌 행 가져오기
                    non_empty = self.df[col].dropna()
                    if non_empty.empty:
                        continue
                        
                    # 수치형으로 변환 시도
                    numeric_values = pd.to_numeric(non_empty, errors='coerce')
                    # NaN이 된 값의 비율이 30% 미만이면 수치형으로 간주
                    nan_ratio = numeric_values.isna().mean()
                    
                    if nan_ratio < 0.3:  # 70% 이상이 숫자로 변환 가능하면
                        # 실제로 변환 적용
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                        new_type = self.df[col].dtype
                        print(f"열 '{col}' 타입 변환: {orig_type} → {new_type}")
                except Exception as e:
                    print(f"열 '{col}' 타입 변환 중 오류: {str(e)}")
            
            # 수치형 및 범주형 열 개수 출력
            numeric_cols = self.df.select_dtypes(include=['number']).columns
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
            print(f"데이터 타입 분석: 수치형 열 {len(numeric_cols)}개, 범주형 열 {len(categorical_cols)}개")
            
            return self.df
        except Exception as e:
            print(f"데이터 로드 실패: {str(e)}")
            raise
            
    def get_basic_info(self):
        """
        데이터프레임의 기본 정보 분석
        
        Returns:
        --------
        dict
            기본 정보를 담은 딕셔너리
        """
        if self.df is None:
            self.load_data()
            
        # 기본 정보 수집
        info = {
            '행 수': len(self.df),
            '열 수': len(self.df.columns),
            '열 이름': list(self.df.columns),
            '데이터 타입': self.df.dtypes.to_dict(),
            '결측치 수': self.df.isnull().sum().to_dict(),
            '결측치 비율(%)': (self.df.isnull().sum() / len(self.df) * 100).to_dict()
        }
        
        # 수치형 열에 대한 기술 통계
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            info['기술통계'] = self.df[numeric_cols].describe().to_dict()
            
        # 범주형 열에 대한 고유값 수 
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            info['범주형 열 고유값 수'] = {col: self.df[col].nunique() for col in categorical_cols}
            info['범주형 열 상위 5개 값'] = {col: self.df[col].value_counts().head().to_dict() for col in categorical_cols}
            
        self.analysis_results['basic_info'] = info
        return info
    
    def analyze_columns(self):
        """
        각 열에 대한 상세 분석
        
        Returns:
        --------
        dict
            각 열의 분석 결과를 담은 딕셔너리
        """
        if self.df is None:
            self.load_data()
            
        column_analysis = {}
        
        for col in self.df.columns:
            col_info = {}
            
            # 데이터 타입
            col_info['데이터 타입'] = str(self.df[col].dtype)
            
            # 결측치 정보
            col_info['결측치 수'] = self.df[col].isnull().sum()
            col_info['결측치 비율(%)'] = round(self.df[col].isnull().sum() / len(self.df) * 100, 2)
            
            # 데이터 타입별 추가 분석
            if np.issubdtype(self.df[col].dtype, np.number):
                # 수치형 데이터
                col_info['최소값'] = self.df[col].min()
                col_info['최대값'] = self.df[col].max()
                col_info['평균값'] = self.df[col].mean()
                col_info['중앙값'] = self.df[col].median()
                col_info['표준편차'] = self.df[col].std()
                col_info['1사분위수'] = self.df[col].quantile(0.25)
                col_info['3사분위수'] = self.df[col].quantile(0.75)
                col_info['이상치 수'] = len(self.df[(self.df[col] < self.df[col].quantile(0.25) - 1.5 * (self.df[col].quantile(0.75) - self.df[col].quantile(0.25))) | 
                                   (self.df[col] > self.df[col].quantile(0.75) + 1.5 * (self.df[col].quantile(0.75) - self.df[col].quantile(0.25)))])
            else:
                # 범주형 데이터
                col_info['고유값 수'] = self.df[col].nunique()
                col_info['최빈값'] = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else None
                col_info['최빈값 빈도'] = self.df[col].value_counts().iloc[0] if not self.df[col].value_counts().empty else 0
                col_info['상위 5개 값'] = self.df[col].value_counts().head().to_dict()
                
            column_analysis[col] = col_info
            
        self.analysis_results['column_analysis'] = column_analysis
        return column_analysis
    
    def get_correlation_analysis(self):
        """
        수치형 열 간의 상관관계 분석
        
        Returns:
        --------
        pandas.DataFrame
            상관계수 행렬
        """
        if self.df is None:
            self.load_data()
            
        # 수치형 열만 선택
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) >= 2:
            # 상관계수 계산
            corr_matrix = numeric_df.corr()
            self.analysis_results['correlation'] = corr_matrix.to_dict()
            return corr_matrix
        else:
            print("수치형 열이 2개 미만이어서 상관관계 분석을 수행할 수 없습니다.")
            return None
    
    def generate_summary_report(self):
        """
        분석 결과를 종합한 요약 보고서 생성
        
        Returns:
        --------
        dict
            종합 분석 결과
        """
        # 아직 분석하지 않았다면 분석 실행
        if not self.analysis_results.get('basic_info'):
            self.get_basic_info()
        
        if not self.analysis_results.get('column_analysis'):
            self.analyze_columns()
            
        if not self.analysis_results.get('correlation') and len(self.df.select_dtypes(include=['number']).columns) >= 2:
            self.get_correlation_analysis()
            
        # 파일 정보
        summary = {
            '파일명': os.path.basename(self.file_path),
            '분석 시간': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '데이터 크기': f"{len(self.df)} 행 × {len(self.df.columns)} 열",
        }
        
        # 결측치 요약
        missing_cols = {col: count for col, count in self.df.isnull().sum().items() if count > 0}
        if missing_cols:
            summary['결측치가 있는 열'] = missing_cols
            summary['결측치 비율이 높은 열(>10%)'] = {
                col: round(count/len(self.df)*100, 2) 
                for col, count in missing_cols.items() 
                if count/len(self.df)*100 > 10
            }
        else:
            summary['결측치'] = "없음"
            
        # 수치형 열 요약
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['수치형 열'] = list(numeric_cols)
            
            # 이상치가 많은 열
            outlier_cols = {
                col: self.analysis_results['column_analysis'][col]['이상치 수']
                for col in numeric_cols
                if self.analysis_results['column_analysis'][col]['이상치 수'] > 0
            }
            
            if outlier_cols:
                summary['이상치가 있는 열'] = outlier_cols
                
        # 범주형 열 요약
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['범주형 열'] = list(categorical_cols)
            summary['고유값이 많은 범주형 열(>10개)'] = {
                col: self.df[col].nunique()
                for col in categorical_cols
                if self.df[col].nunique() > 10
            }
            
        # 상관관계 요약
        if 'correlation' in self.analysis_results:
            corr_matrix = pd.DataFrame(self.analysis_results['correlation'])
            # 절대값 0.5 이상의 상관관계 찾기
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) >= 0.5:
                        strong_correlations.append({
                            '열1': corr_matrix.columns[i],
                            '열2': corr_matrix.columns[j],
                            '상관계수': round(corr_matrix.iloc[i, j], 3)
                        })
            
            if strong_correlations:
                summary['강한 상관관계(|r|≥0.5)'] = strong_correlations
                
        self.analysis_results['summary'] = summary
        return summary
    
    def print_report(self):
        """
        분석 보고서를 콘솔에 출력
        """
        summary = self.generate_summary_report()
        
        print("\n" + "="*50)
        print(f"📊 엑셀 파일 분석 보고서: {summary['파일명']}")
        print(f"📅 분석 시간: {summary['분석 시간']}")
        print(f"📋 데이터 크기: {summary['데이터 크기']}")
        print("="*50)
        
        # 기본 정보 출력
        print("\n📌 기본 정보:")
        print(f"행 수: {self.analysis_results['basic_info']['행 수']}")
        print(f"열 수: {self.analysis_results['basic_info']['열 수']}")
        print(f"열 이름: {', '.join(self.analysis_results['basic_info']['열 이름'])}")
        
        # 결측치 정보 출력
        print("\n📌 결측치 정보:")
        missing_info = {col: count for col, count in self.df.isnull().sum().items() if count > 0}
        if missing_info:
            for col, count in missing_info.items():
                percent = round(count/len(self.df)*100, 2)
                print(f"- {col}: {count}개 ({percent}%)")
        else:
            print("결측치 없음")
            
        # 수치형 열 요약
        print("\n📌 수치형 열 요약:")
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            for col in numeric_cols:
                col_info = self.analysis_results['column_analysis'][col]
                print(f"\n- {col}:")
                print(f"  범위: {col_info['최소값']} ~ {col_info['최대값']}")
                
                # Timedelta 타입은 round를 지원하지 않으므로 조건부 처리
                if isinstance(col_info['평균값'], pd.Timedelta):
                    print(f"  평균: {col_info['평균값']}, 중앙값: {col_info['중앙값']}")
                else:
                    try:
                        print(f"  평균: {round(col_info['평균값'], 2)}, 중앙값: {col_info['중앙값']}")
                    except:
                        print(f"  평균: {col_info['평균값']}, 중앙값: {col_info['중앙값']}")
                
                # 표준편차도 마찬가지로 조건부 처리
                if isinstance(col_info['표준편차'], pd.Timedelta):
                    print(f"  표준편차: {col_info['표준편차']}")
                else:
                    try:
                        print(f"  표준편차: {round(col_info['표준편차'], 2)}")
                    except:
                        print(f"  표준편차: {col_info['표준편차']}")
                
                print(f"  이상치 수: {col_info['이상치 수']}")
        else:
            print("수치형 열 없음")
            
        # 범주형 열 요약
        print("\n📌 범주형 열 요약:")
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                col_info = self.analysis_results['column_analysis'][col]
                print(f"\n- {col}:")
                print(f"  고유값 수: {col_info['고유값 수']}")
                print(f"  최빈값: {col_info['최빈값']} (빈도: {col_info['최빈값 빈도']})")
                print("  상위 값:")
                for val, count in list(col_info['상위 5개 값'].items())[:3]:
                    print(f"    {val}: {count}개")
        else:
            print("범주형 열 없음")
            
        # 상관관계 요약
        print("\n📌 상관관계 요약:")
        if 'correlation' in self.analysis_results and summary.get('강한 상관관계(|r|≥0.5)'):
            for corr in summary['강한 상관관계(|r|≥0.5)']:
                print(f"- {corr['열1']} ↔ {corr['열2']}: r = {corr['상관계수']}")
        else:
            print("강한 상관관계 없음 또는 분석 불가")
            
        print("\n" + "="*50)
        print("분석 완료")
        print("="*50)
        
        return self.analysis_results
    
    def visualize_data(self, output_dir=None):
        """
        데이터 시각화 결과 생성
        
        Parameters:
        -----------
        output_dir : str, optional
            시각화 결과를 저장할 디렉토리 경로
        """
        if self.df is None:
            self.load_data()
            
        if output_dir is None:
            # 기본 저장 경로 설정
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(script_dir), 'Output')
            os.makedirs(output_dir, exist_ok=True)
        
        # 시각화 스타일 설정
        plt.style.use('ggplot')
        sns.set(font='Malgun Gothic')  # 한글 폰트 설정
        
        # 1. 결측치 시각화
        plt.figure(figsize=(10, 6))
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            # 결측치가 많은 경우 상위 20개만 표시
            if len(missing) > 20:
                missing = missing.sort_values(ascending=False).head(20)
                plt.title('결측치 수 (상위 20개)')
            else:
                plt.title('결측치 수')
                
            missing.plot(kind='bar')
            plt.ylabel('결측치 수')
            plt.xlabel('열')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '결측치_분석.png'))
            plt.close()
            
        # 2. 수치형 열 분포 시각화
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        for i, col in enumerate(numeric_cols):
            # 타임델타 데이터형은 히스토그램으로 시각화하지 않음
            if pd.api.types.is_timedelta64_dtype(self.df[col].dtype):
                continue
                
            # 고유값이 너무 많거나 적은 경우 다른 방식으로 시각화
            unique_vals = self.df[col].nunique()
            
            plt.figure(figsize=(12, 5))
            
            try:
                # 분포도 (히스토그램)
                plt.subplot(1, 2, 1)
                
                if unique_vals <= 10:  # 고유값이 적은 경우 (이산형)
                    # 값 카운트하여 막대 그래프로 표시
                    value_counts = self.df[col].value_counts().sort_index()
                    plt.bar(value_counts.index, value_counts.values)
                    plt.title(f'{col} 분포')
                    plt.xticks(rotation=45)
                else:  # 고유값이 많은 경우 (연속형)
                    # 데이터 범위를 적절한 수의 빈(bin)으로 나누기
                    max_bins = min(30, unique_vals)  # 빈의 최대 개수를 30으로 제한
                    
                    # 히스토그램 그리기 
                    plt.hist(self.df[col].dropna(), bins=max_bins, alpha=0.7)
                    plt.title(f'{col} 분포')
                
                # 박스플롯
                plt.subplot(1, 2, 2)
                plt.boxplot(self.df[col].dropna())
                plt.title(f'{col} 박스플롯')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{col}_분포.png'))
                plt.close()
            except Exception as e:
                print(f"{col} 열 시각화 실패: {str(e)}")
                plt.close()
                continue
            
        # 3. 범주형 열 분포 시각화
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_vals = self.df[col].nunique()
            
            # 고유값이 20개 이하인 경우만 시각화 
            if unique_vals <= 20:
                try:
                    plt.figure(figsize=(10, 6))
                    value_counts = self.df[col].value_counts().sort_values(ascending=False)
                    
                    # 막대 그래프
                    plt.bar(value_counts.index, value_counts.values)
                    plt.title(f'{col} 분포')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'{col}_분포.png'))
                    plt.close()
                except Exception as e:
                    print(f"{col} 열 시각화 실패: {str(e)}")
                    plt.close()
                    continue
                
        # 4. 상관관계 히트맵
        numeric_df = self.df.select_dtypes(include=['number'])
        
        # 수치형 열이 너무 많은 경우 상관관계가 강한 열들만 선택
        if len(numeric_df.columns) > 15:
            print("수치형 열이 너무 많아 강한 상관관계가 있는 열들만 시각화합니다.")
            
            # 상관관계 계산
            corr_matrix = numeric_df.corr()
            
            # 상관관계가 강한 열 찾기 (절대값 0.5 이상)
            strong_corr_cols = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) >= 0.5:
                        strong_corr_cols.add(corr_matrix.columns[i])
                        strong_corr_cols.add(corr_matrix.columns[j])
            
            # 상관관계가 강한 열들에 대한 상관관계 히트맵 그리기
            if strong_corr_cols:
                plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_df[list(strong_corr_cols)].corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
                plt.title('강한 상관관계가 있는 열들의 상관계수 히트맵')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '상관관계_히트맵_강한관계.png'))
                plt.close()
        
        elif len(numeric_df.columns) >= 2:
            try:
                plt.figure(figsize=(10, 8))
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1)
                plt.title('상관관계 히트맵')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, '상관관계_히트맵.png'))
                plt.close()
            except Exception as e:
                print(f"상관관계 히트맵 시각화 실패: {str(e)}")
                plt.close()
            
        print(f"\n시각화 결과가 {output_dir} 폴더에 저장되었습니다.")


def main():
    """
    메인 함수
    """
    print("엑셀 파일 분석기 실행")
    analyzer = ExcelAnalyzer()
    analyzer.load_data()
    analyzer.print_report()
    
    # 시각화 결과 생성 (옵션)
    visualize = input("\n데이터 시각화 결과를 생성하시겠습니까? (y/n): ")
    if visualize.lower() == 'y':
        analyzer.visualize_data()
    
    return analyzer.analysis_results


if __name__ == "__main__":
    main() 