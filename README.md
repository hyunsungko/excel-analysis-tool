# 엑셀 분석기 (Excel Analyzer)

엑셀 파일을 쉽게 분석하고 시각화할 수 있는 도구입니다. 이 프로그램은 CLI(명령줄 인터페이스)와 GUI(그래픽 사용자 인터페이스) 모두를 지원합니다.

## 주요 기능

- 엑셀 파일 로드 및 구조 분석
- 데이터 유형 자동 감지 및 분류(수치형, 범주형, 주관식 등)
- 기초 통계 분석 및 상관관계 분석
- 데이터 시각화 (막대 차트, 히스토그램, 박스플롯, 상관관계 히트맵 등)
- HTML 및 텍스트 형식의 보고서 생성
- 사용자 친화적인 GUI 제공

## 설치 방법

### 요구 사항

- Python 3.7 이상
- 필요한 패키지: pandas, numpy, matplotlib, seaborn, PyQt5

### 설치 과정

1. 프로젝트 클론 또는 다운로드:
```
git clone https://github.com/yourusername/excel-analyzer.git
```

2. 필요한 패키지 설치:
```
pip install pandas numpy matplotlib seaborn PyQt5
```

## 사용 방법

### CLI 모드

CLI 모드에서는 다음과 같은 명령어로 엑셀 파일을 분석할 수 있습니다:

```
python src/cli.py --file "분석할엑셀파일경로.xlsx" --output "결과저장폴더" [--report-format html|text|all] [--no-viz] [--sheet "시트이름"]
```

옵션 설명:
- `--file`, `-f`: 분석할 엑셀 파일 경로 (필수)
- `--output`, `-o`: 결과물 저장 디렉토리 (기본값: ./output)
- `--report-format`: 보고서 형식 (html, text, all 중 선택, 기본값: html)
- `--no-viz`: 시각화 생성 비활성화
- `--sheet`: 분석할 시트 이름 (기본값: 첫번째 시트)

예시:
```
python src/cli.py --file "./data/example.xlsx" --output "./reports" --report-format all
```

### GUI 모드

GUI 모드는 다음 명령어로 실행할 수 있습니다:

```
python src/main.py
```

GUI 모드에서의 사용 단계:

1. **파일 로드**: 
   - "파일" 메뉴에서 "열기"를 선택하거나 툴바의 "열기" 버튼 클릭
   - 분석할 엑셀 파일 선택

2. **데이터 처리**:
   - "데이터" 메뉴에서 "데이터 처리"를 선택
   - 데이터 처리 옵션 선택 (결측치 처리, 이상치 제거 등)

3. **분석 실행**:
   - "분석" 메뉴에서 "분석 실행"을 선택
   - 분석 유형 선택 (기술통계, 상관관계 등)

4. **시각화 생성**:
   - "시각화" 메뉴에서 "시각화 생성"을 선택
   - 차트 유형 선택 (막대, 히스토그램, 박스플롯 등)

5. **보고서 생성**:
   - "보고서" 메뉴에서 "보고서 생성"을 선택
   - 보고서 형식 및 옵션 선택

## 예제

### 기본 분석 예제

```python
from src.utils.data_loader import DataLoader
from src.core.data_processor import DataProcessor
from src.core.analysis_engine import AnalysisEngine
from src.visualization.visualization_engine import VisualizationEngine
from src.reporting.report_engine import ReportEngine

# 데이터 로드
loader = DataLoader()
df = loader.load_file("./data/example.xlsx")

# 데이터 처리
processor = DataProcessor()
processed_data = processor.process(df)

# 데이터 분석
analyzer = AnalysisEngine()
analysis_results = analyzer.analyze(processed_data)

# 시각화 생성
visualizer = VisualizationEngine(output_dir="./output/viz")
viz_results = visualizer.generate_visualizations(processed_data, analysis_results)

# 보고서 생성
reporter = ReportEngine()
report = reporter.generate_report(processed_data, analysis_results, viz_results)
```

## 라이센스

이 프로젝트는 MIT 라이센스를 따릅니다.

## 기여

이슈 제보 및 풀 리퀘스트는 언제나 환영합니다! 