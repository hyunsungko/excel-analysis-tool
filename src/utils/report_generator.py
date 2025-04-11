"""
보고서 생성 유틸리티
"""

import os
import pandas as pd
from datetime import datetime
import webbrowser
from jinja2 import Environment, FileSystemLoader
import pdfkit

class ReportGenerator:
    """보고서 생성 클래스"""
    
    def __init__(self, template_dir=None, output_dir="reports"):
        """
        ReportGenerator 생성자
        
        Args:
            template_dir (str): 템플릿 디렉토리 경로
            output_dir (str): 출력 디렉토리 경로
        """
        self.title = "데이터 분석 보고서"
        self.subtitle = f"생성일: {datetime.now().strftime('%Y-%m-%d')}"
        self.dataframe = None
        self.statistics = None
        self.visualizations = []
        
        # 템플릿 디렉토리가 없으면 기본 경로 설정
        if template_dir is None:
            self.template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))), "templates")
        else:
            self.template_dir = template_dir
            
        # 출력 디렉토리가 없으면 생성
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Jinja2 환경 설정
        if os.path.exists(self.template_dir):
            self.env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=True
            )
        else:
            print(f"경고: 템플릿 디렉토리 '{self.template_dir}'가 존재하지 않습니다.")
    
    def set_title(self, title):
        """보고서 제목 설정"""
        self.title = title
        
    def set_subtitle(self, subtitle):
        """보고서 부제목 설정"""
        self.subtitle = subtitle
        
    def set_dataframe(self, dataframe):
        """데이터프레임 설정"""
        self.dataframe = dataframe
        
    def set_statistics(self, statistics):
        """통계 데이터 설정"""
        self.statistics = statistics
        
    def set_visualizations(self, visualizations):
        """시각화 데이터 설정"""
        self.visualizations = visualizations
        
    def generate_html(self, output_path):
        """HTML 보고서 생성"""
        if self.dataframe is None:
            raise ValueError("데이터프레임이 설정되지 않았습니다.")
            
        # 템플릿 로드
        template = self.env.get_template('report_template.html')
        
        # 데이터 요약 정보
        data_info = {
            "rows": len(self.dataframe),
            "columns": len(self.dataframe.columns),
            "missing": self.dataframe.isna().sum().sum(),
            "numeric_cols": len(self.dataframe.select_dtypes(include=["number"]).columns),
            "categorical_cols": len(self.dataframe.select_dtypes(include=["object", "category"]).columns),
            "datetime_cols": len(self.dataframe.select_dtypes(include=["datetime"]).columns)
        }
        
        # 통계 정보 (없으면 기본 통계 생성)
        if self.statistics is None:
            self.statistics = {
                "descriptive": self.dataframe.describe().to_html(classes="table table-striped"),
                "correlation": self.dataframe.select_dtypes(include=["number"]).corr().to_html(classes="table table-striped")
            }
            
        # 시각화 정보 (플롯 경로 목록)
        visualizations = self.visualizations if self.visualizations else []
        
        # 컨텍스트 생성
        context = {
            "title": self.title,
            "subtitle": self.subtitle,
            "data_info": data_info,
            "data_sample": self.dataframe.head(10).to_html(classes="table table-striped"),
            "statistics": self.statistics,
            "visualizations": visualizations,
            "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # HTML 생성
        html_content = template.render(**context)
        
        # 파일 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path
    
    def generate_pdf(self, output_path):
        """PDF 보고서 생성"""
        # 우선 HTML 파일 생성
        html_path = output_path.replace('.pdf', '.html')
        self.generate_html(html_path)
        
        # HTML을 PDF로 변환
        try:
            pdfkit.from_file(html_path, output_path)
            # 임시 HTML 파일 삭제
            os.remove(html_path)
            return output_path
        except Exception as e:
            print(f"PDF 생성 오류: {str(e)}")
            print("HTML 파일로 대체합니다.")
            return html_path 