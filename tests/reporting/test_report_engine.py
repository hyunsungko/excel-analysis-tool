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

from src.reporting.report_engine import ReportEngine

class TestReportEngine(unittest.TestCase):
    """ReportEngine 클래스에 대한 단위 테스트"""
    
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
        
        # 테스트용 시각화 파일 생성
        self.viz_file = os.path.join(self.temp_dir, "test_viz.png")
        with open(self.viz_file, "w") as f:
            f.write("테스트 시각화 파일")
            
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
            
        # ReportEngine 인스턴스 생성
        self.engine = ReportEngine(output_dir=self.temp_dir, template_dir=self.temp_template_dir)
        
    def tearDown(self):
        """테스트 후 임시 디렉토리 정리"""
        shutil.rmtree(self.temp_dir)
        shutil.rmtree(self.temp_template_dir)
        
    def test_set_title(self):
        """set_title 메서드 테스트"""
        title = "테스트 보고서"
        subtitle = "테스트 부제목"
        author = "테스트 작성자"
        
        self.engine.set_title(title, subtitle, author)
        
        self.assertEqual(self.engine.title, title)
        self.assertEqual(self.engine.subtitle, subtitle)
        self.assertEqual(self.engine.author, author)
        
    def test_add_dataframe(self):
        """add_dataframe 메서드 테스트"""
        name = "test_data"
        self.engine.add_dataframe(self.df, name)
        
        # 데이터프레임이 추가되었는지 확인
        self.assertIn(name, self.engine.data)
        self.assertEqual(self.engine.data[name].shape, self.df.shape)
        
        # 요약 통계가 생성되었는지 확인
        self.assertIn(name, self.engine.summary_stats)
        self.assertEqual(self.engine.summary_stats[name]["shape"], self.df.shape)
        
    def test_add_visualization(self):
        """add_visualization 메서드 테스트"""
        title = "테스트 시각화"
        description = "테스트 설명"
        category = "test_category"
        
        self.engine.add_visualization(self.viz_file, title, description, category)
        
        # 시각화가 추가되었는지 확인
        self.assertEqual(len(self.engine.visualizations), 1)
        viz = self.engine.visualizations[0]
        self.assertEqual(viz["file_path"], self.viz_file)
        self.assertEqual(viz["title"], title)
        self.assertEqual(viz["description"], description)
        self.assertEqual(viz["category"], category)
        
    def test_add_summary_stats(self):
        """add_summary_stats 메서드 테스트"""
        name = "custom_stats"
        stats = {
            "mean": 10.5,
            "median": 9.0,
            "custom_metric": "test_value"
        }
        
        self.engine.add_summary_stats(stats, name)
        
        # 통계가 추가되었는지 확인
        self.assertIn(name, self.engine.summary_stats)
        self.assertEqual(self.engine.summary_stats[name], stats)
        
    def test_generate_text_report(self):
        """generate_text_report 메서드 테스트"""
        # 데이터 추가
        self.engine.set_title("텍스트 보고서 테스트")
        self.engine.add_dataframe(self.df, "main_data")
        self.engine.add_visualization(self.viz_file, "테스트 차트", "차트 설명")
        
        # 텍스트 보고서 생성
        filename = "text_report_test.txt"
        file_path = self.engine.generate_text_report(filename)
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(file_path))
        
        # 파일 내용 확인
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("텍스트 보고서 테스트", content)
        self.assertIn("main_data", content)
        self.assertIn("테스트 차트", content)
        
    def test_generate_html_report(self):
        """generate_html_report 메서드 테스트"""
        # 데이터 추가
        self.engine.set_title("HTML 보고서 테스트")
        self.engine.add_dataframe(self.df, "main_data")
        self.engine.add_visualization(self.viz_file, "테스트 차트", "차트 설명")
        
        # HTML 보고서 생성
        filename = "html_report_test.html"
        file_path = self.engine.generate_html_report(filename, template="test_template.html")
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(file_path))
        
        # 파일 내용 확인
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        self.assertIn("HTML 보고서 테스트", content)
        self.assertIn("main_data", content)
        self.assertIn("테스트 차트", content)
        
    def test_export_report(self):
        """export_report 메서드 테스트"""
        # 데이터 추가
        self.engine.set_title("내보내기 테스트")
        self.engine.add_dataframe(self.df, "main_data")
        
        # HTML 형식으로 내보내기
        html_path = self.engine.export_report(format="html", filename="export_test.html")
        self.assertTrue(os.path.exists(html_path))
        
        # 텍스트 형식으로 내보내기
        text_path = self.engine.export_report(format="text", filename="export_test.txt")
        self.assertTrue(os.path.exists(text_path))
        
        # 지원되지 않는 형식으로 내보내기
        invalid_path = self.engine.export_report(format="invalid", filename="export_test.xyz")
        self.assertEqual(invalid_path, "")
        
if __name__ == '__main__':
    unittest.main() 