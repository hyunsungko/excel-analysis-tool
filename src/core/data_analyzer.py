import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import json
import platform
import re

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic' if platform.system() == 'Windows' else 'AppleGothic' if platform.system() == 'Darwin' else 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 안전한 파일명으로 변환하는 함수
def safe_filename(name):
    # 줄바꿈, 탭 등의 공백 문자를 일반 공백으로 변경
    name = re.sub(r'\s+', ' ', name)
    # 파일명으로 사용할 수 없는 문자 처리
    name = name.replace('/', '_').replace('\\', '_').replace(':', '_')
    name = name.replace('*', '_').replace('?', '_').replace('"', '_')
    name = name.replace('<', '_').replace('>', '_').replace('|', '_')
    # 최대 길이 제한 (Windows 파일 시스템 제약)
    if len(name) > 200:
        name = name[:197] + '...'
    return name

class SimpleExcelAnalyzer:
    """
    복잡한 엑셀 파일을 간단하게 분석하는 클래스
    """
    
    def __init__(self, file_path=None):
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Data')
        
        if file_path is None:
            excel_files = [f for f in os.listdir(self.data_dir) if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]
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
        """
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"데이터 로드 완료: {len(self.df)} 행, {len(self.df.columns)} 열")
            return self.df
        except Exception as e:
            print(f"데이터 로드 실패: {str(e)}")
            raise
    
    def analyze_basic_stats(self):
        """
        기본 통계 분석
        """
        if self.df is None:
            self.load_data()
        
        # 기본 정보 수집
        info = {
            '파일명': os.path.basename(self.file_path),
            '행 수': len(self.df),
            '열 수': len(self.df.columns),
            '결측치가 있는 열 수': sum(self.df.isnull().any()),
            '결측치 심각 열 수(50% 이상)': sum(self.df.isnull().mean() > 0.5)
        }
        
        # 데이터 타입별 열 개수
        dtype_counts = self.df.dtypes.value_counts().to_dict()
        dtype_counts = {str(k): v for k, v in dtype_counts.items()}
        info['데이터 타입별 열 개수'] = dtype_counts
        
        # 결측치 상위 10개 열
        missing_cols = self.df.isnull().sum().sort_values(ascending=False)
        missing_cols = missing_cols[missing_cols > 0]
        if not missing_cols.empty:
            top_missing = missing_cols.head(10)
            info['결측치 상위 10개 열'] = {col: {'개수': count, '비율': f"{count/len(self.df)*100:.1f}%"} 
                                   for col, count in top_missing.items()}
        
        self.analysis_results['basic_info'] = info
        return info
    
    def analyze_categorical_cols(self, max_cols=15):
        """
        범주형 열 분석
        
        Parameters:
        -----------
        max_cols : int
            분석할 최대 열 개수
        """
        if self.df is None:
            self.load_data()
        
        # 범주형 열 선택
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        # 너무 많은 열이 있는 경우 일부만 선택
        if len(categorical_cols) > max_cols:
            # 고유값이 적은 열을 우선 선택
            nunique_values = {col: self.df[col].nunique() for col in categorical_cols}
            sorted_cols = sorted(nunique_values.items(), key=lambda x: x[1])
            categorical_cols = [col for col, _ in sorted_cols[:max_cols]]
            
            print(f"범주형 열이 너무 많아 고유값이 적은 {max_cols}개 열만 분석합니다.")
        
        categorical_analysis = {}
        
        for col in categorical_cols:
            col_info = {}
            
            # 결측치 정보
            missing_count = self.df[col].isnull().sum()
            col_info['결측치 수'] = missing_count
            col_info['결측치 비율'] = f"{missing_count/len(self.df)*100:.1f}%"
            
            # 고유값 정보
            unique_count = self.df[col].nunique()
            col_info['고유값 수'] = unique_count
            
            # 빈도 분석 (상위 5개)
            if unique_count <= 30:  # 고유값이 많지 않은 경우만
                value_counts = self.df[col].value_counts().head(5).to_dict()
                col_info['상위 5개 값'] = value_counts
            
            categorical_analysis[col] = col_info
        
        self.analysis_results['categorical_analysis'] = categorical_analysis
        return categorical_analysis
    
    def analyze_numeric_cols(self, max_cols=15):
        """
        수치형 열 분석
        
        Parameters:
        -----------
        max_cols : int
            분석할 최대 열 개수
        """
        if self.df is None:
            self.load_data()
        
        # 수치형 열 선택
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        # 너무 많은 열이 있는 경우 일부만 선택
        if len(numeric_cols) > max_cols:
            # 이상치 수가 많은 열 위주로 선택 (더 주목할 가치가 있을 수 있음)
            outlier_counts = {}
            for col in numeric_cols:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_counts[col] = len(self.df[(self.df[col] < Q1 - 1.5 * IQR) | 
                                            (self.df[col] > Q3 + 1.5 * IQR)])
            
            sorted_cols = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)
            numeric_cols = [col for col, _ in sorted_cols[:max_cols]]
            
            print(f"수치형 열이 너무 많아 이상치가 많은 {max_cols}개 열만 분석합니다.")
        
        numeric_analysis = {}
        
        for col in numeric_cols:
            try:
                col_info = {}
                
                # 결측치 정보
                missing_count = self.df[col].isnull().sum()
                col_info['결측치 수'] = missing_count
                col_info['결측치 비율'] = f"{missing_count/len(self.df)*100:.1f}%"
                
                # 기본 통계량
                col_info['최소값'] = float(self.df[col].min())
                col_info['최대값'] = float(self.df[col].max())
                col_info['평균'] = float(self.df[col].mean())
                col_info['중앙값'] = float(self.df[col].median())
                col_info['표준편차'] = float(self.df[col].std())
                
                # 히스토그램 빈도 (10개 구간으로 분할)
                values = self.df[col].dropna()
                if len(values) > 0:
                    hist, bin_edges = np.histogram(values, bins=min(10, len(values.unique())))
                    col_info['히스토그램'] = {
                        '빈도': hist.tolist(),
                        '구간': [f"{bin_edges[i]:.2f} ~ {bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
                    }
                
                numeric_analysis[col] = col_info
            except Exception as e:
                print(f"{col} 열 분석 중 오류 발생: {str(e)}")
        
        self.analysis_results['numeric_analysis'] = numeric_analysis
        return numeric_analysis
    
    def analyze_correlations(self, threshold=0.5, max_pairs=50):
        """
        상관관계 분석
        
        Parameters:
        -----------
        threshold : float
            표시할 상관계수의 최소 절대값 (기본값: 0.5)
        max_pairs : int
            최대 표시할 상관관계 쌍의 수 (기본값: 50)
        """
        if self.df is None:
            self.load_data()
        
        # 수치형 열만 선택
        numeric_df = self.df.select_dtypes(include=['number'])
        
        if len(numeric_df.columns) < 2:
            print("수치형 열이 2개 미만이어서 상관관계 분석을 수행할 수 없습니다.")
            return None
        
        # 상관계수 계산
        corr_matrix = numeric_df.corr()
        
        # 강한 상관관계만 추출
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        '열1': corr_matrix.columns[i],
                        '열2': corr_matrix.columns[j],
                        '상관계수': round(corr_value, 3)
                    })
        
        # 상관계수 절대값 기준으로 정렬
        strong_correlations.sort(key=lambda x: abs(x['상관계수']), reverse=True)
        
        # 최대 표시 개수 제한
        if len(strong_correlations) > max_pairs:
            strong_correlations = strong_correlations[:max_pairs]
            print(f"상관관계가 너무 많아 절대값이 큰 {max_pairs}개만 표시합니다.")
        
        self.analysis_results['correlations'] = strong_correlations
        return strong_correlations
    
    def create_basic_visualizations(self, output_dir=None):
        """
        기본적인 시각화 생성
        
        Parameters:
        -----------
        output_dir : str
            결과 저장 디렉토리
        """
        if self.df is None:
            self.load_data()
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(script_dir), 'Output')
            os.makedirs(output_dir, exist_ok=True)
        
        # 그래프 스타일 설정
        plt.style.use('ggplot')
        
        # 1. 결측치 시각화 (상위 15개만)
        plt.figure(figsize=(10, 6))
        missing = self.df.isnull().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        
        if not missing.empty:
            # 결측치가 많은 경우 상위 15개만 표시
            if len(missing) > 15:
                missing = missing.head(15)
            
            ax = missing.plot(kind='bar')
            ax.set_title('결측치 수 (상위 15개)')
            ax.set_ylabel('결측치 수')
            ax.set_xlabel('열')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '결측치_분석.png'))
            plt.close()
        
        # 2. 범주형 변수 시각화 (고유값이 적은 상위 5개만)
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        cols_with_few_unique = [(col, self.df[col].nunique()) for col in categorical_cols 
                               if self.df[col].nunique() <= 10 and self.df[col].nunique() > 1]
        
        if cols_with_few_unique:
            # 고유값이 적은 순으로 정렬
            cols_with_few_unique.sort(key=lambda x: x[1])
            
            # 최대 5개까지만 시각화
            for col, _ in cols_with_few_unique[:5]:
                plt.figure(figsize=(10, 5))
                value_counts = self.df[col].value_counts().sort_values(ascending=False)
                
                if len(value_counts) <= 15:  # 값이 너무 많지 않은 경우
                    ax = value_counts.plot(kind='bar')
                    ax.set_title(f'{col} 값 분포')
                    ax.set_ylabel('빈도')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    
                    # 안전한 파일명 생성
                    safe_col_name = safe_filename(col)
                    
                    plt.savefig(os.path.join(output_dir, f'{safe_col_name}_분포.png'))
                    plt.close()
        
        # 3. 수치형 변수 시각화 (분산이 큰 상위 5개만)
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            # 분산 계산
            variances = [(col, self.df[col].var()) for col in numeric_cols 
                         if not pd.api.types.is_timedelta64_dtype(self.df[col].dtype)]
            
            # 분산이 큰 순으로 정렬
            variances.sort(key=lambda x: x[1], reverse=True)
            
            # 최대 5개까지만 시각화
            for col, _ in variances[:5]:
                try:
                    plt.figure(figsize=(10, 5))
                    
                    # 히스토그램
                    values = self.df[col].dropna()
                    if len(values) > 0:
                        ax = plt.hist(values, bins=min(20, len(values.unique())), alpha=0.7)
                        plt.title(f'{col} 히스토그램')
                        plt.xlabel('값')
                        plt.ylabel('빈도')
                        plt.tight_layout()
                        
                        # 안전한 파일명 생성
                        safe_col_name = safe_filename(col)
                        
                        plt.savefig(os.path.join(output_dir, f'{safe_col_name}_히스토그램.png'))
                        plt.close()
                except Exception as e:
                    print(f"{col} 열 시각화 중 오류 발생: {str(e)}")
                    plt.close()
        
        # 4. 상관관계 히트맵 (절대값 0.7 이상의 상관관계만)
        numeric_df = self.df.select_dtypes(include=['number'])
        
        # 타임델타 열 제외
        cols_to_drop = [col for col in numeric_df.columns if pd.api.types.is_timedelta64_dtype(numeric_df[col].dtype)]
        if cols_to_drop:
            numeric_df = numeric_df.drop(columns=cols_to_drop)
        
        if len(numeric_df.columns) >= 2:
            try:
                # 상관행렬 계산
                corr_matrix = numeric_df.corr()
                
                # 절대값 0.7 이상인 상관관계만 추출
                strong_cols = set()
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) >= 0.7:
                            strong_cols.add(corr_matrix.columns[i])
                            strong_cols.add(corr_matrix.columns[j])
                
                # 강한 상관관계가 있는 경우만 히트맵 생성
                if strong_cols:
                    # 열 이름이 너무 길면 자르기
                    col_names = list(strong_cols)
                    short_names = {col: col[:15] + '...' if len(col) > 15 else col for col in col_names}
                    
                    plt.figure(figsize=(10, 8))
                    strong_corr = corr_matrix.loc[col_names, col_names].copy()
                    
                    # 열 이름 변경
                    strong_corr.index = [short_names[col] for col in strong_corr.index]
                    strong_corr.columns = [short_names[col] for col in strong_corr.columns]
                    
                    im = plt.matshow(strong_corr, fignum=plt.gcf().number, cmap='coolwarm', vmin=-1, vmax=1)
                    plt.colorbar(im)
                    
                    # 상관계수 표시
                    for i in range(len(strong_corr.columns)):
                        for j in range(len(strong_corr.columns)):
                            plt.text(j, i, f"{strong_corr.iloc[i, j]:.2f}", 
                                     ha='center', va='center', 
                                     color='white' if abs(strong_corr.iloc[i, j]) > 0.7 else 'black')
                    
                    plt.xticks(range(len(strong_corr.columns)), strong_corr.columns, rotation=90)
                    plt.yticks(range(len(strong_corr.columns)), strong_corr.columns)
                    plt.title('강한 상관관계 히트맵 (|r| >= 0.7)')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, '강한_상관관계_히트맵.png'))
                    plt.close()
            except Exception as e:
                print(f"상관관계 히트맵 생성 중 오류 발생: {str(e)}")
                plt.close()
        
        print(f"\n시각화 결과가 {output_dir} 폴더에 저장되었습니다.")
    
    def save_analysis_results(self, output_dir=None):
        """
        분석 결과를 JSON 파일로 저장
        
        Parameters:
        -----------
        output_dir : str
            결과 저장 디렉토리
        """
        if not self.analysis_results:
            print("저장할 분석 결과가 없습니다. 먼저 분석을 실행하세요.")
            return
        
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(script_dir), 'Output')
            os.makedirs(output_dir, exist_ok=True)
        
        # 현재 시간을 파일명에 포함
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'분석결과_{timestamp}.json')
        
        # JSON 변환 불가능한 값 처리
        def json_serialize(obj):
            try:
                return str(obj)
            except:
                return None
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_results, f, ensure_ascii=False, indent=2, default=json_serialize)
        
        print(f"분석 결과가 '{output_file}'에 저장되었습니다.")
    
    def print_analysis_summary(self):
        """
        분석 결과 요약 출력
        """
        if not self.analysis_results:
            print("출력할 분석 결과가 없습니다. 먼저 분석을 실행하세요.")
            return
        
        # 기본 정보 출력
        if 'basic_info' in self.analysis_results:
            info = self.analysis_results['basic_info']
            print("\n" + "=" * 50)
            print(f"📊 엑셀 파일 분석 요약: {info['파일명']}")
            print("=" * 50)
            print(f"📋 데이터 크기: {info['행 수']} 행 × {info['열 수']} 열")
            print(f"🔍 결측치가 있는 열: {info['결측치가 있는 열 수']}개")
            print(f"⚠️ 결측치가 50% 이상인 열: {info['결측치 심각 열 수(50% 이상)']}개")
            
            if '결측치 상위 10개 열' in info:
                print("\n📉 결측치가 많은 열 (상위 5개):")
                for i, (col, data) in enumerate(list(info['결측치 상위 10개 열'].items())[:5]):
                    print(f"  {i+1}. {col}: {data['개수']}개 ({data['비율']})")
        
        # 범주형 열 분석 결과 출력
        if 'categorical_analysis' in self.analysis_results:
            cat_analysis = self.analysis_results['categorical_analysis']
            print("\n📊 범주형 데이터 분석:")
            print(f"  총 {len(cat_analysis)}개 열 분석됨")
            
            # 몇 가지 예시 출력
            if cat_analysis:
                print("\n  [범주형 열 예시]")
                for i, (col, data) in enumerate(list(cat_analysis.items())[:3]):
                    print(f"  {i+1}. {col}:")
                    print(f"     - 고유값 수: {data['고유값 수']}")
                    print(f"     - 결측치: {data['결측치 수']}개 ({data['결측치 비율']})")
                    
                    if '상위 5개 값' in data:
                        print("     - 주요 값:", end=" ")
                        for j, (val, count) in enumerate(list(data['상위 5개 값'].items())[:3]):
                            print(f"{val}({count}개)", end=", " if j < 2 else "")
                        print("...")
        
        # 수치형 열 분석 결과 출력
        if 'numeric_analysis' in self.analysis_results:
            num_analysis = self.analysis_results['numeric_analysis']
            print("\n📈 수치형 데이터 분석:")
            print(f"  총 {len(num_analysis)}개 열 분석됨")
            
            # 몇 가지 예시 출력
            if num_analysis:
                print("\n  [수치형 열 예시]")
                for i, (col, data) in enumerate(list(num_analysis.items())[:3]):
                    print(f"  {i+1}. {col}:")
                    print(f"     - 범위: {data.get('최소값', 'N/A')} ~ {data.get('최대값', 'N/A')}")
                    print(f"     - 평균: {data.get('평균', 'N/A')}, 중앙값: {data.get('중앙값', 'N/A')}")
                    print(f"     - 표준편차: {data.get('표준편차', 'N/A')}")
        
        # 상관관계 분석 결과 출력
        if 'correlations' in self.analysis_results:
            correlations = self.analysis_results['correlations']
            print("\n🔗 강한 상관관계:")
            print(f"  총 {len(correlations)}개 발견됨")
            
            # 절대값이 큰 상위 5개 출력
            if correlations:
                print("\n  [상위 상관관계]")
                for i, corr in enumerate(correlations[:5]):
                    print(f"  {i+1}. {corr['열1']} ↔ {corr['열2']}: r = {corr['상관계수']}")
        
        print("\n" + "=" * 50)
        print("분석 완료")
        print("=" * 50)


def main():
    """
    메인 함수
    """
    print("간소화된 엑셀 파일 분석기 실행")
    
    analyzer = SimpleExcelAnalyzer()
    analyzer.load_data()
    
    # 기본 분석 수행
    print("\n기본 정보 분석 중...")
    analyzer.analyze_basic_stats()
    
    print("\n범주형 데이터 분석 중...")
    analyzer.analyze_categorical_cols()
    
    print("\n수치형 데이터 분석 중...")
    analyzer.analyze_numeric_cols()
    
    print("\n상관관계 분석 중...")
    analyzer.analyze_correlations()
    
    # 요약 출력
    analyzer.print_analysis_summary()
    
    # 결과 저장
    analyzer.save_analysis_results()
    
    # 시각화 생성 (옵션)
    visualize = input("\n데이터 시각화 결과를 생성하시겠습니까? (y/n): ")
    if visualize.lower() == 'y':
        print("\n기본 시각화 생성 중...")
        analyzer.create_basic_visualizations()
    
    return analyzer.analysis_results


if __name__ == "__main__":
    main() 