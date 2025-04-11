import os
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape

class ReportEngine:
    """
    엑셀 분석 시스템의 분석 결과를 기반으로 보고서를 생성하는 엔진
    
    다양한 형식(HTML, PDF, 텍스트 등)으로 보고서를 출력할 수 있습니다.
    """
    
    def __init__(self, output_dir: str = "reports", template_dir: str = "templates"):
        """
        ReportEngine 초기화
        
        Args:
            output_dir (str): 보고서 출력 디렉토리
            template_dir (str): 보고서 템플릿 디렉토리
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.data: Dict[str, Any] = {}
        self.visualizations: List[Dict[str, str]] = []
        self.summary_stats: Dict[str, Any] = {}
        self.title = "데이터 분석 보고서"
        self.subtitle = ""
        self.author = ""
        self.creation_date = datetime.now()
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Jinja2 템플릿 환경 설정
        if os.path.exists(template_dir):
            self.jinja_env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=select_autoescape(['html', 'xml'])
            )
        else:
            self.logger.warning(f"템플릿 디렉토리 '{template_dir}'가 존재하지 않습니다.")
            self.jinja_env = None
            
    def generate_report(self, data: pd.DataFrame, analysis_results: Dict[str, Any], 
                      visualizations: Optional[List[Dict[str, Any]]] = None,
                      output_format: str = 'html', 
                      metadata: Optional[Dict[str, str]] = None) -> str:
        """
        분석 결과를 기반으로 보고서 생성
        
        Args:
            data (pd.DataFrame): 분석 데이터
            analysis_results (Dict[str, Any]): 분석 결과
            visualizations (List[Dict[str, Any]], optional): 시각화 결과 목록
            output_format (str): 출력 형식 ('html', 'text')
            metadata (Dict[str, str], optional): 보고서 메타데이터
            
        Returns:
            str: 생성된 보고서 내용
        """
        self.logger.info(f"{output_format} 형식의 보고서 생성 시작")
        
        # 데이터프레임 추가
        self.add_dataframe(data)
        
        # 메타데이터 설정
        if metadata:
            title = metadata.get('title', '데이터 분석 보고서')
            subtitle = metadata.get('subtitle', '')
            author = metadata.get('author', '')
            self.set_title(title, subtitle, author)
        
        # 분석 결과 추가
        if analysis_results:
            self.add_summary_stats(analysis_results)
        
        # 시각화 추가
        if visualizations:
            for viz in visualizations:
                self.add_visualization(
                    file_path=viz.get('path', ''),
                    title=viz.get('title', ''),
                    description=viz.get('description', ''),
                    category=viz.get('type', 'general')
                )
        
        # 보고서 형식에 따라 생성
        content = ""
        if output_format == 'html':
            # HTML 템플릿 기반 보고서 생성
            if self.jinja_env:
                try:
                    template = self.jinja_env.get_template('report_template.html')
                    
                    # 컨텍스트 준비
                    context = {
                        "title": self.title,
                        "subtitle": self.subtitle,
                        "author": self.author,
                        "generation_date": self.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "summary_stats": self.summary_stats,
                        "visualizations": self.visualizations,
                        "toc": [
                            {"id": "summary", "title": "요약 정보"},
                            {"id": "data_overview", "title": "데이터 개요"},
                            {"id": "visualizations", "title": "시각화"},
                            {"id": "analysis", "title": "분석 결과"},
                            {"id": "conclusion", "title": "결론"}
                        ],
                        "content": self._generate_html_content(data, analysis_results)
                    }
                    
                    # 템플릿 렌더링
                    content = template.render(**context)
                except Exception as e:
                    self.logger.error(f"HTML 보고서 생성 중 오류 발생: {str(e)}")
                    content = f"<h1>보고서 생성 오류</h1><p>{str(e)}</p>"
            else:
                content = "<h1>템플릿 오류</h1><p>템플릿 환경이 설정되지 않았습니다.</p>"
        
        elif output_format == 'text':
            # 텍스트 보고서 생성
            try:
                if self.jinja_env:
                    template = self.jinja_env.get_template('text_report_template.txt')
                    
                    # 컨텍스트 준비
                    context = {
                        "title": self.title,
                        "subtitle": self.subtitle,
                        "author": self.author,
                        "generation_date": self.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
                        "toc": [
                            {"id": "1", "title": "요약 정보"},
                            {"id": "2", "title": "데이터 개요"},
                            {"id": "3", "title": "분석 결과"},
                            {"id": "4", "title": "결론"}
                        ],
                        "content": self._generate_text_content(data, analysis_results)
                    }
                    
                    # 템플릿 렌더링
                    content = template.render(**context)
                else:
                    content = self._generate_text_content(data, analysis_results)
            except Exception as e:
                self.logger.error(f"텍스트 보고서 생성 중 오류 발생: {str(e)}")
                content = f"보고서 생성 오류: {str(e)}"
        
        else:
            self.logger.error(f"지원되지 않는 보고서 형식: {output_format}")
            content = f"지원되지 않는 보고서 형식: {output_format}"
        
        self.logger.info(f"{output_format} 보고서 생성 완료")
        return content
    
    def _generate_html_content(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """HTML 보고서 콘텐츠 생성"""
        sections = []
        
        # 요약 정보
        sections.append("""
        <section id="summary" class="section">
            <h2>요약 정보</h2>
            <div class="summary">
                <p>이 보고서는 제공된 데이터셋에 대한 분석 결과를 포함하고 있습니다.</p>
                <p>데이터 크기: {} 행 x {} 열</p>
            </div>
        </section>
        """.format(data.shape[0], data.shape[1]))
        
        # 데이터 개요
        sections.append("""
        <section id="data_overview" class="section">
            <h2>데이터 개요</h2>
            <h3>데이터 미리보기</h3>
            {}
            <h3>데이터 형식</h3>
            <table>
                <thead>
                    <tr>
                        <th>열 이름</th>
                        <th>데이터 유형</th>
                        <th>결측치</th>
                    </tr>
                </thead>
                <tbody>
                    {}
                </tbody>
            </table>
        </section>
        """.format(
            data.head(5).to_html(classes='table table-striped'),
            "".join([
                f"<tr><td>{col}</td><td>{data[col].dtype}</td><td>{data[col].isna().sum()}</td></tr>"
                for col in data.columns
            ])
        ))
        
        # 시각화
        if self.visualizations:
            viz_html = []
            viz_html.append("""
            <section id="visualizations" class="section">
                <h2>시각화</h2>
            """)
            
            for viz in self.visualizations:
                viz_html.append(f"""
                <div class="visualization">
                    <h3>{viz['title']}</h3>
                    <img src="{viz['file_path']}" alt="{viz['title']}">
                    <p>{viz['description']}</p>
                </div>
                """)
                
            viz_html.append("</section>")
            sections.append("".join(viz_html))
        
        # 분석 결과
        sections.append("""
        <section id="analysis" class="section">
            <h2>분석 결과</h2>
        """)
        
        # 기본 통계량
        if 'basic_stats' in analysis_results:
            sections.append("<h3>기본 통계량</h3>")
            for col, stats in analysis_results['basic_stats'].items():
                sections.append(f"""
                <div class="stat-card">
                    <h4>{col}</h4>
                    <ul>
                        <li>최솟값: {stats.get('min', 'N/A')}</li>
                        <li>최댓값: {stats.get('max', 'N/A')}</li>
                        <li>평균값: {stats.get('mean', 'N/A')}</li>
                        <li>중앙값: {stats.get('median', 'N/A')}</li>
                        <li>표준편차: {stats.get('std', 'N/A')}</li>
                    </ul>
                </div>
                """)
        
        # 범주형 통계량
        if 'categorical_stats' in analysis_results:
            sections.append("<h3>범주형 변수 분석</h3>")
            for col, stats in analysis_results['categorical_stats'].items():
                sections.append(f"""
                <div class="stat-card">
                    <h4>{col}</h4>
                    <p>고유값 수: {stats.get('unique_values', 'N/A')}</p>
                    <p>결측치: {stats.get('missing_count', 'N/A')} ({stats.get('missing_percent', 'N/A')}%)</p>
                </div>
                """)
        
        # 상관관계
        if 'correlation' in analysis_results:
            sections.append("<h3>상관관계 분석</h3>")
            sections.append("<p>변수 간 상관관계 매트릭스:</p>")
            corr_df = analysis_results['correlation']
            if isinstance(corr_df, pd.DataFrame):
                sections.append(corr_df.to_html(classes='table table-striped'))
            
        sections.append("</section>")
        
        # 결론
        sections.append("""
        <section id="conclusion" class="section">
            <h2>결론</h2>
            <p>이 보고서는 제공된 데이터에 대한 기본적인 분석 결과를 제공합니다.</p>
            <p>자세한 분석과 해석은 도메인 전문가의 리뷰가 필요합니다.</p>
        </section>
        """)
        
        return "".join(sections)
    
    def _generate_text_content(self, data: pd.DataFrame, analysis_results: Dict[str, Any]) -> str:
        """텍스트 보고서 콘텐츠 생성"""
        sections = []
        
        # 구분선
        hr = "=" * 60 + "\n"
        
        # 요약 정보
        sections.append(hr)
        sections.append("1. 요약 정보\n")
        sections.append(hr)
        sections.append(f"데이터 크기: {data.shape[0]} 행 x {data.shape[1]} 열\n")
        sections.append(f"분석 날짜: {self.creation_date.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 데이터 개요
        sections.append(hr)
        sections.append("2. 데이터 개요\n")
        sections.append(hr)
        sections.append("데이터 열:\n")
        for col in data.columns:
            sections.append(f"- {col} ({data[col].dtype}): 결측치 {data[col].isna().sum()}개\n")
        sections.append("\n")
        
        # 분석 결과
        sections.append(hr)
        sections.append("3. 분석 결과\n")
        sections.append(hr)
        
        # 기본 통계량
        if 'basic_stats' in analysis_results:
            sections.append("기본 통계량:\n")
            for col, stats in analysis_results['basic_stats'].items():
                sections.append(f"\n{col}:\n")
                sections.append(f"  - 최솟값: {stats.get('min', 'N/A')}\n")
                sections.append(f"  - 최댓값: {stats.get('max', 'N/A')}\n")
                sections.append(f"  - 평균값: {stats.get('mean', 'N/A')}\n")
                sections.append(f"  - 중앙값: {stats.get('median', 'N/A')}\n")
                sections.append(f"  - 표준편차: {stats.get('std', 'N/A')}\n")
            sections.append("\n")
        
        # 범주형 통계량
        if 'categorical_stats' in analysis_results:
            sections.append("범주형 변수 분석:\n")
            for col, stats in analysis_results['categorical_stats'].items():
                sections.append(f"\n{col}:\n")
                sections.append(f"  - 고유값 수: {stats.get('unique_values', 'N/A')}\n")
                sections.append(f"  - 결측치: {stats.get('missing_count', 'N/A')} ({stats.get('missing_percent', 'N/A')}%)\n")
                if 'top_values' in stats:
                    sections.append("  - 상위 값:\n")
                    for val, val_stats in stats['top_values'].items():
                        sections.append(f"    * {val}: {val_stats['count']} ({val_stats['percentage']}%)\n")
            sections.append("\n")
        
        # 결론
        sections.append(hr)
        sections.append("4. 결론\n")
        sections.append(hr)
        sections.append("이 보고서는 제공된 데이터에 대한 기본적인 분석 결과를 제공합니다.\n")
        sections.append("자세한 분석과 해석은 도메인 전문가의 리뷰가 필요합니다.\n\n")
        
        return "".join(sections)
    
    def set_title(self, title: str, subtitle: str = "", author: str = ""):
        """
        보고서 제목 설정
        
        Args:
            title (str): 보고서 제목
            subtitle (str, optional): 보고서 부제목
            author (str, optional): 작성자
        """
        self.title = title
        self.subtitle = subtitle
        self.author = author
        
    def add_dataframe(self, df: pd.DataFrame, name: str = "main_data"):
        """
        보고서에 데이터프레임 추가
        
        Args:
            df (pd.DataFrame): 추가할 데이터프레임
            name (str, optional): 데이터프레임 식별자
        """
        if df is None or df.empty:
            self.logger.warning("빈 데이터프레임이 추가되었습니다.")
            return
            
        self.data[name] = df
        
        # 기본 요약 통계 생성
        try:
            self.summary_stats[name] = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "na_counts": df.isna().sum().to_dict(),
                "numeric_summary": df.describe().to_dict() if not df.select_dtypes(include=['number']).empty else {}
            }
        except Exception as e:
            self.logger.error(f"요약 통계 생성 중 오류 발생: {str(e)}")
            
    def add_visualization(self, file_path: str, title: str = "", description: str = "", 
                        category: str = "general"):
        """
        보고서에 시각화 추가
        
        Args:
            file_path (str): 시각화 이미지 파일 경로
            title (str, optional): 시각화 제목
            description (str, optional): 시각화 설명
            category (str, optional): 시각화 카테고리
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"시각화 파일 '{file_path}'이 존재하지 않습니다.")
            return
            
        self.visualizations.append({
            "file_path": file_path,
            "title": title,
            "description": description,
            "category": category,
            "filename": os.path.basename(file_path)
        })
        
    def add_summary_stats(self, stats: Dict[str, Any], name: str = "custom_stats"):
        """
        보고서에 사용자 정의 요약 통계 추가
        
        Args:
            stats (Dict[str, Any]): 추가할 요약 통계
            name (str, optional): 통계 식별자
        """
        if not stats:
            self.logger.warning("빈 요약 통계가 추가되었습니다.")
            return
            
        self.summary_stats[name] = stats
        
    def generate_html_report(self, filename: str = "", template: str = "report_template.html") -> str:
        """
        HTML 형식의 보고서 생성
        
        Args:
            filename (str, optional): 저장할 파일명
            template (str, optional): 사용할 템플릿 파일명
            
        Returns:
            str: 저장된 HTML 파일 경로
        """
        if self.jinja_env is None:
            self.logger.error("템플릿 환경이 설정되지 않았습니다.")
            return ""
            
        try:
            # 템플릿 로드
            template = self.jinja_env.get_template(template)
            
            # 파일명 생성
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}.html"
                
            # 파일 경로 생성
            file_path = os.path.join(self.output_dir, filename)
            
            # 렌더링할 컨텍스트 데이터 준비
            context = {
                "title": self.title,
                "subtitle": self.subtitle,
                "author": self.author,
                "creation_date": self.creation_date.strftime("%Y-%m-%d %H:%M:%S"),
                "summary_stats": self.summary_stats,
                "visualizations": self.visualizations,
                "data_tables": {name: df.head(10).to_html(classes='table table-striped') 
                             for name, df in self.data.items() if not df.empty}
            }
            
            # 템플릿 렌더링
            html_content = template.render(**context)
            
            # 파일 저장
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(html_content)
                
            self.logger.info(f"HTML 보고서가 생성되었습니다: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"HTML 보고서 생성 중 오류 발생: {str(e)}")
            return ""
            
    def generate_text_report(self, filename: str = "") -> str:
        """
        텍스트 형식의 보고서 생성
        
        Args:
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 텍스트 파일 경로
        """
        try:
            # 파일명 생성
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}.txt"
                
            # 파일 경로 생성
            file_path = os.path.join(self.output_dir, filename)
            
            # 텍스트 보고서 작성
            with open(file_path, "w", encoding="utf-8") as f:
                # 타이틀 섹션
                f.write(f"{self.title}\n")
                f.write("=" * len(self.title) + "\n\n")
                
                if self.subtitle:
                    f.write(f"{self.subtitle}\n\n")
                    
                if self.author:
                    f.write(f"작성자: {self.author}\n")
                    
                f.write(f"생성 일시: {self.creation_date.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # 데이터 요약 섹션
                f.write("데이터 요약\n")
                f.write("-" * 40 + "\n\n")
                
                for name, stats in self.summary_stats.items():
                    f.write(f"데이터셋: {name}\n")
                    
                    if "shape" in stats:
                        f.write(f"- 크기: {stats['shape'][0]}행 x {stats['shape'][1]}열\n")
                        
                    if "columns" in stats:
                        f.write(f"- 열 목록: {', '.join(stats['columns'])}\n")
                        
                    if "na_counts" in stats:
                        na_cols = {k: v for k, v in stats["na_counts"].items() if v > 0}
                        if na_cols:
                            f.write(f"- 결측치: {na_cols}\n")
                            
                    f.write("\n")
                    
                # 시각화 섹션
                if self.visualizations:
                    f.write("시각화 목록\n")
                    f.write("-" * 40 + "\n\n")
                    
                    for i, viz in enumerate(self.visualizations, 1):
                        f.write(f"{i}. {viz['title'] or viz['filename']}\n")
                        if viz['description']:
                            f.write(f"   설명: {viz['description']}\n")
                        f.write(f"   파일: {viz['file_path']}\n")
                        f.write(f"   카테고리: {viz['category']}\n\n")
                        
                # 데이터 미리보기 섹션
                for name, df in self.data.items():
                    if not df.empty:
                        f.write(f"{name} 데이터 미리보기\n")
                        f.write("-" * 40 + "\n\n")
                        f.write(str(df.head(5)) + "\n\n")
                
            self.logger.info(f"텍스트 보고서가 생성되었습니다: {file_path}")
            return file_path
        except Exception as e:
            self.logger.error(f"텍스트 보고서 생성 중 오류 발생: {str(e)}")
            return ""
            
    def generate_pdf_report(self, filename: str = "", template: str = "report_template.html") -> str:
        """
        PDF 형식의 보고서 생성
        
        Args:
            filename (str, optional): 저장할 파일명
            template (str, optional): 사용할 템플릿 파일명
            
        Returns:
            str: 저장된 PDF 파일 경로
        """
        try:
            # 파일명 생성
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{timestamp}.pdf"
                
            # 파일 경로 생성
            file_path = os.path.join(self.output_dir, filename)
            
            # HTML 보고서를 먼저 생성
            html_file = self.generate_html_report(filename.replace(".pdf", ".html"), template)
            
            if not html_file:
                self.logger.error("HTML 보고서 생성에 실패하여 PDF 변환을 진행할 수 없습니다.")
                return ""
            
            try:
                # HTML을 PDF로 변환 (wkhtmltopdf 필요)
                import pdfkit
                pdfkit.from_file(html_file, file_path)
                self.logger.info(f"PDF 보고서가 생성되었습니다: {file_path}")
                return file_path
            except ImportError:
                self.logger.warning("pdfkit 모듈이 설치되어 있지 않습니다. pip install pdfkit를 실행하여 설치하세요.")
                return ""
            except Exception as e:
                self.logger.error(f"PDF 변환 중 오류 발생: {str(e)}")
                self.logger.warning("PDF 생성을 위해 wkhtmltopdf가 설치되어 있어야 합니다.")
                return ""
        except Exception as e:
            self.logger.error(f"PDF 보고서 생성 중 오류 발생: {str(e)}")
            return ""
            
    def export_report(self, format: str = "html", filename: str = "") -> str:
        """
        요청한 형식으로 보고서 내보내기
        
        Args:
            format (str): 보고서 형식 ('html', 'pdf', 'text')
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        format = format.lower()
        
        # 파일 이름 확장자 확인 및 수정
        if filename and not filename.lower().endswith(f".{format}"):
            filename = f"{os.path.splitext(filename)[0]}.{format}"
        
        if format == "html":
            return self.generate_html_report(filename=filename, template="test_template.html")
        elif format == "pdf":
            return self.generate_pdf_report(filename=filename)
        elif format == "text":
            return self.generate_text_report(filename=filename)
        else:
            self.logger.error(f"지원되지 않는 보고서 형식: {format}")
            return "" 