import unittest
import os
import shutil
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys

# 로깅 비활성화
logging.disable(logging.CRITICAL)

# 모듈 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 모듈 임포트를 시스템 구성요소에 맞게 수정
from src.reporting.report_integrator import ReportIntegrator

# 테스트용 간단한 스텁 클래스 구현
class StubDataLoader:
    def __init__(self):
        self.data = None
    
    def set_data(self, data):
        self.data = data
        
    def get_data(self):
        return self.data
        
    def load_excel(self, file_path):
        return True

class StubDataProcessor:
    def __init__(self):
        self.data = None
        self.processed_data = None
        
    def set_data(self, data):
        self.data = data
        self.processed_data = data.copy()
        
    def get_processed_data(self):
        return self.processed_data
        
    def fill_missing_values(self):
        if self.processed_data is not None:
            self.processed_data = self.processed_data.fillna(0)
        
    def remove_duplicates(self):
        if self.processed_data is not None:
            self.processed_data = self.processed_data.drop_duplicates()

class StubAnalysisEngine:
    def __init__(self):
        self.data = None
        self.results = {}
        
    def set_data(self, data):
        self.data = data
        
    def calculate_descriptive_statistics(self):
        if self.data is not None:
            self.results['descriptive_stats'] = {'mean': self.data.mean(numeric_only=True).to_dict()}
            
    def calculate_correlation_analysis(self):
        if self.data is not None:
            numeric_data = self.data.select_dtypes(include=['number'])
            if not numeric_data.empty:
                self.results['correlation'] = {'pearson': numeric_data.corr().to_dict()}
                
    def get_all_results(self):
        return self.results

class StubVisualizationEngine:
    def __init__(self, output_dir=''):
        self.df = None
        self.output_dir = output_dir
        
    def set_dataframe(self, df):
        self.df = df
        
    def get_dataframe(self):
        return self.df
        
    def export_all_visualizations(self, prefix='test'):
        if self.df is None:
            return ""
            
        # 가상의 시각화 파일 생성
        with open(os.path.join(self.output_dir, f"{prefix}_test.png"), 'w') as f:
            f.write("Test visualization file")
            
        return """다음 시각화가 생성되었습니다:
1. test_test.png"""

class TestReportIntegrator(unittest.TestCase):
    """ReportIntegrator 클래스에 대한 단위 테스트"""
    
    def setUp(self):
        """테스트 데이터 및 임시 출력 디렉토리 설정"""
        # 임시 출력 디렉토리 생성
        self.temp_dir = tempfile.mkdtemp()
        self.temp_template_dir = tempfile.mkdtemp()
        
        # 테스트용 데이터프레임 생성
        self.df = pd.DataFrame({
            'id': range(1, 11),
            'category': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'B', 'A', 'D'],
            'value1': [10, 20, 15, 25, 30, 35, 40, 45, 50, 55],
            'value2': [100, 90, 80, 70, 60, 50, 40, 30, 20, 10],
            'date': pd.date_range(start='2023-01-01', periods=10),
            'duration': [timedelta(minutes=m) for m in range(10, 110, 10)],
            'text': ['abc', 'def', 'ghi', 'jkl', 'mno', 'pqr', 'stu', 'vwx', 'yz1', '234']
        })
        
        # NaN 값 추가
        self.df.loc[0, 'value1'] = np.nan
        self.df.loc[3, 'value2'] = np.nan
        self.df.loc[5, 'category'] = np.nan
        
        # 테스트용 템플릿 파일 생성
        self.template_file = os.path.join(self.temp_template_dir, "test_template.html")
        with open(self.template_file, "w") as f:
            f.write("""<!DOCTYPE html>
            <html>
            <head><title>{{ title }}</title></head>
            <body>
                <h1>{{ title }}</h1>
                {% if subtitle %}<h2>{{ subtitle }}</h2>{% endif %}
                {% if author %}<p>By: {{ author }}</p>{% endif %}
                <p>Date: {{ creation_date }}</p>
                
                <h2>Data Summary</h2>
                {% for name, stats in summary_stats.items() %}
                    <h3>{{ name }}</h3>
                    <p>Shape: {{ stats.shape[0] }} x {{ stats.shape[1] }}</p>
                {% endfor %}
                
                <h2>Visualizations</h2>
                {% for viz in visualizations %}
                    <div>
                        <h3>{{ viz.title }}</h3>
                        <p>{{ viz.description }}</p>
                        <img src="{{ viz.file_path }}" alt="{{ viz.title }}">
                    </div>
                {% endfor %}
                
                <h2>Data Preview</h2>
                {% for name, table in data_tables.items() %}
                    <h3>{{ name }}</h3>
                    {{ table|safe }}
                {% endfor %}
            </body>
            </html>""")
            
        # 시각화 파일 생성
        self.viz_temp_dir = os.path.join(self.temp_dir, "visualizations")
        os.makedirs(self.viz_temp_dir)
        self.viz_file = os.path.join(self.viz_temp_dir, "test_viz.png")
        with open(self.viz_file, "w") as f:
            f.write("테스트 시각화 파일")
            
        # ReportIntegrator 인스턴스 생성
        self.integrator = ReportIntegrator(
            output_dir=self.temp_dir, 
            template_dir=self.temp_template_dir
        )
        
        # 스텁 구성 요소 설정
        self.data_loader = StubDataLoader()
        self.data_loader.set_data(self.df.copy())
        
        self.data_processor = StubDataProcessor()
        self.data_processor.set_data(self.df.copy())
        
        self.analysis_engine = StubAnalysisEngine()
        self.analysis_engine.set_data(self.df.copy())
        self.analysis_engine.calculate_descriptive_statistics()
        
        self.visualization_engine = StubVisualizationEngine(output_dir=self.viz_temp_dir)
        self.visualization_engine.set_dataframe(self.df.copy())
        
        # 통합기에 구성 요소 설정
        self.integrator.set_components(
            data_loader=self.data_loader,
            data_processor=self.data_processor,
            analysis_engine=self.analysis_engine,
            visualization_engine=self.visualization_engine
        )
        
    def tearDown(self):
        """테스트 후 임시 디렉토리 정리"""
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.temp_template_dir)
        
    def test_set_components(self):
        """set_components 메서드 테스트"""
        integrator = ReportIntegrator(output_dir=self.temp_dir)
        
        # 초기에는 구성 요소가 None
        self.assertIsNone(integrator.data_loader)
        self.assertIsNone(integrator.data_processor)
        self.assertIsNone(integrator.analysis_engine)
        self.assertIsNone(integrator.visualization_engine)
        
        # 구성 요소 설정
        integrator.set_components(
            data_loader=self.data_loader,
            data_processor=self.data_processor,
            analysis_engine=self.analysis_engine,
            visualization_engine=self.visualization_engine
        )
        
        # 구성 요소가 설정되었는지 확인
        self.assertEqual(integrator.data_loader, self.data_loader)
        self.assertEqual(integrator.data_processor, self.data_processor)
        self.assertEqual(integrator.analysis_engine, self.analysis_engine)
        self.assertEqual(integrator.visualization_engine, self.visualization_engine)
        
    def test_integrate_data_sources(self):
        """integrate_data_sources 메서드 테스트"""
        # 데이터 통합
        result = self.integrator.integrate_data_sources()
        
        # 통합 성공 여부 확인
        self.assertTrue(result)
        
        # 보고서 엔진에 데이터가 추가되었는지 확인
        self.assertIn("원본 데이터", self.integrator.report_engine.data)
        self.assertIn("가공된 데이터", self.integrator.report_engine.data)
        
    def test_integrate_analysis_results(self):
        """integrate_analysis_results 메서드 테스트"""
        # 분석 결과 통합
        result = self.integrator.integrate_analysis_results()
        
        # 통합 성공 여부 확인
        self.assertTrue(result)
        
        # 보고서 엔진에 분석 결과가 추가되었는지 확인
        self.assertIn("분석 결과", self.integrator.report_engine.summary_stats)
        
    def test_integrate_visualizations(self):
        """integrate_visualizations 메서드 테스트"""
        # 시각화 통합 (직접 시각화 파일 정보 제공)
        viz_info = [{
            "file_path": self.viz_file,
            "title": "테스트 시각화",
            "description": "테스트 설명",
            "category": "test_category"
        }]
        
        result = self.integrator.integrate_visualizations(viz_info)
        
        # 통합 성공 여부 확인
        self.assertTrue(result)
        
        # 보고서 엔진에 시각화가 추가되었는지 확인
        self.assertEqual(len(self.integrator.report_engine.visualizations), 1)
        self.assertEqual(
            self.integrator.report_engine.visualizations[0]["file_path"], 
            self.viz_file
        )
        
    def test_generate_comprehensive_report(self):
        """generate_comprehensive_report 메서드 테스트"""
        # 종합 보고서 생성
        file_path = self.integrator.generate_comprehensive_report(
            title="종합 보고서 테스트",
            format="text",
            filename="comprehensive_test.txt"
        )
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(file_path))
        
        # 파일 내용 확인
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("종합 보고서 테스트", content)
        
if __name__ == '__main__':
    unittest.main() 