import os
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd

# 시스템 구성요소 모듈 임포트를 수정합니다
try:
    # src 모듈에서 직접 임포트 시도
    from src.utils.data_loader import DataLoader
    from src.core.data_processor import DataProcessor
    from src.core.analysis_engine import AnalysisEngine
    from src.visualization.visualization_engine import VisualizationEngine
    from src.reporting.report_engine import ReportEngine
except ImportError:
    # 상대 경로 임포트 시도
    try:
        from ..utils.data_loader import DataLoader
        from ..core.data_processor import DataProcessor
        from ..core.analysis_engine import AnalysisEngine
        from ..visualization.visualization_engine import VisualizationEngine
        from .report_engine import ReportEngine
    except ImportError:
        # 절대 경로로 임포트
        from utils.data_loader import DataLoader
        from core.data_processor import DataProcessor
        from core.analysis_engine import AnalysisEngine
        from visualization.visualization_engine import VisualizationEngine
        from reporting.report_engine import ReportEngine

class ReportIntegrator:
    """
    엑셀 분석 시스템의 다양한 구성 요소와 결과를 통합하여 종합 보고서를 생성하는 클래스
    
    데이터 로더, 프로세서, 분석 엔진, 시각화 엔진의 결과를 하나의 보고서로 통합합니다.
    """
    
    def __init__(self, output_dir: str = "reports", template_dir: str = "templates"):
        """
        ReportIntegrator 초기화
        
        Args:
            output_dir (str): 보고서 출력 디렉토리
            template_dir (str): 보고서 템플릿 디렉토리
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        
        # 각 엔진 초기화
        self.data_loader = None
        self.data_processor = None
        self.analysis_engine = None
        self.visualization_engine = None
        self.report_engine = ReportEngine(output_dir=output_dir, template_dir=template_dir)
        
        # 로거 설정
        self.logger = logging.getLogger(__name__)
        
        # 출력 디렉토리 생성
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
    def set_components(self, 
                     data_loader: Optional[DataLoader] = None, 
                     data_processor: Optional[DataProcessor] = None,
                     analysis_engine: Optional[AnalysisEngine] = None,
                     visualization_engine: Optional[VisualizationEngine] = None):
        """
        보고서 통합에 사용할 구성 요소 설정
        
        Args:
            data_loader (DataLoader, optional): 데이터 로더 인스턴스
            data_processor (DataProcessor, optional): 데이터 프로세서 인스턴스
            analysis_engine (AnalysisEngine, optional): 분석 엔진 인스턴스
            visualization_engine (VisualizationEngine, optional): 시각화 엔진 인스턴스
        """
        if data_loader is not None:
            self.data_loader = data_loader
            
        if data_processor is not None:
            self.data_processor = data_processor
            
        if analysis_engine is not None:
            self.analysis_engine = analysis_engine
            
        if visualization_engine is not None:
            self.visualization_engine = visualization_engine
            
    def integrate_data_sources(self, original_data: Optional[pd.DataFrame] = None,
                             processed_data: Optional[pd.DataFrame] = None) -> bool:
        """
        원본 데이터와 가공된 데이터를 보고서에 통합
        
        Args:
            original_data (pd.DataFrame, optional): 원본 데이터. None인 경우 data_loader에서 로드
            processed_data (pd.DataFrame, optional): 가공된 데이터. None인 경우 data_processor에서 처리
            
        Returns:
            bool: 성공 여부
        """
        # 원본 데이터 확인
        if original_data is None and self.data_loader is not None:
            original_data = self.data_loader.df
            
        if original_data is not None:
            self.report_engine.add_dataframe(original_data, name="원본 데이터")
            
        # 가공된 데이터 확인
        if processed_data is None and self.data_processor is not None and original_data is not None:
            try:
                self.data_processor.set_dataframe(original_data)
                processed_data = self.data_processor.process(original_data)
            except Exception as e:
                self.logger.error(f"데이터 가공 중 오류 발생: {str(e)}")
                
        if processed_data is not None:
            self.report_engine.add_dataframe(processed_data, name="가공된 데이터")
            
        return original_data is not None or processed_data is not None
        
    def integrate_analysis_results(self, analysis_results: Optional[Dict[str, Any]] = None) -> bool:
        """
        분석 결과를 보고서에 통합
        
        Args:
            analysis_results (Dict[str, Any], optional): 분석 결과. None인 경우 analysis_engine에서 가져옴
            
        Returns:
            bool: 성공 여부
        """
        if analysis_results is None and self.analysis_engine is not None:
            try:
                analysis_results = self.analysis_engine.get_all_results()
            except Exception as e:
                self.logger.error(f"분석 결과 가져오기 중 오류 발생: {str(e)}")
                return False
                
        if not analysis_results:
            return False
            
        # 분석 결과를 요약 통계로 추가
        self.report_engine.add_summary_stats(analysis_results, name="분석 결과")
        return True
        
    def integrate_visualizations(self, visualization_files: Optional[List[Dict[str, str]]] = None) -> bool:
        """
        시각화 결과를 보고서에 통합
        
        Args:
            visualization_files (List[Dict[str, str]], optional): 시각화 파일 정보. 
                None인 경우 visualization_engine에서 가져옴
                
        Returns:
            bool: 성공 여부
        """
        if visualization_files is None and self.visualization_engine is not None:
            try:
                # 필요한 기본 시각화 생성
                if self.visualization_engine.get_dataframe() is not None:
                    vis_result = self.visualization_engine.export_all_visualizations()
                    # 시각화 결과를 파싱하여 파일 목록 생성
                    visualization_files = []
                    if vis_result:
                        lines = vis_result.strip().split('\n')
                        if lines[0].startswith("다음 시각화가 생성되었습니다"):
                            for line in lines[1:]:
                                if line.strip():
                                    try:
                                        idx, filename = line.strip().split('. ', 1)
                                        file_path = os.path.join(self.visualization_engine.output_dir, filename)
                                        if os.path.exists(file_path):
                                            category = "일반"
                                            if "hist" in filename:
                                                category = "분포"
                                            elif "bar" in filename:
                                                category = "범주형"
                                            elif "corr" in filename:
                                                category = "상관관계"
                                            elif "scatter" in filename:
                                                category = "산점도"
                                                
                                            visualization_files.append({
                                                "file_path": file_path,
                                                "title": os.path.splitext(filename)[0],
                                                "description": "",
                                                "category": category
                                            })
                                    except Exception as e:
                                        self.logger.warning(f"시각화 파일 파싱 중 오류: {str(e)}")
            except Exception as e:
                self.logger.error(f"시각화 결과 가져오기 중 오류 발생: {str(e)}")
                return False
                
        if not visualization_files:
            return False
            
        # 시각화 파일 추가
        for viz in visualization_files:
            self.report_engine.add_visualization(
                file_path=viz["file_path"],
                title=viz.get("title", ""),
                description=viz.get("description", ""),
                category=viz.get("category", "general")
            )
            
        return True
        
    def generate_comprehensive_report(self, title: str = "엑셀 데이터 분석 보고서", 
                                    subtitle: str = "", 
                                    author: str = "",
                                    format: str = "html",
                                    filename: str = "") -> str:
        """
        모든 구성 요소를 통합하여 종합 보고서 생성
        
        Args:
            title (str): 보고서 제목
            subtitle (str, optional): 보고서 부제목
            author (str, optional): 작성자
            format (str): 보고서 형식 ('html', 'pdf', 'text')
            filename (str, optional): 저장할 파일명
            
        Returns:
            str: 저장된 파일 경로
        """
        # 보고서 제목 설정
        self.report_engine.set_title(title, subtitle, author)
        
        # 데이터 통합
        data_integrated = self.integrate_data_sources()
        
        # 분석 결과 통합
        analysis_integrated = self.integrate_analysis_results()
        
        # 시각화 통합
        viz_integrated = self.integrate_visualizations()
        
        if not (data_integrated or analysis_integrated or viz_integrated):
            self.logger.warning("통합할 데이터, 분석 결과, 시각화가 없습니다.")
            return ""
            
        # 보고서 생성
        return self.report_engine.export_report(format=format, filename=filename)
        
    def quick_report(self, file_path: str, 
                   title: str = "엑셀 데이터 빠른 분석 보고서",
                   format: str = "html") -> str:
        """
        엑셀 파일을 한 번에 분석하고 보고서를 생성하는 간편 메서드
        
        Args:
            file_path (str): 엑셀 파일 경로
            title (str): 보고서 제목
            format (str): 보고서 형식 ('html', 'pdf', 'text')
            
        Returns:
            str: 저장된 보고서 파일 경로
        """
        try:
            # 데이터 로더 초기화
            self.data_loader = DataLoader()
            self.data_loader.load_file(file_path)
                
            # 원본 데이터 가져오기
            original_data = self.data_loader.df
            if original_data is None or original_data.empty:
                self.logger.error("유효한 데이터가 없습니다.")
                return ""
                
            # 데이터 프로세서 초기화
            self.data_processor = DataProcessor()
            self.data_processor.set_dataframe(original_data)
            
            # 기본 전처리
            processed_data = self.data_processor.process(original_data)
            
            # 분석 엔진 초기화
            self.analysis_engine = AnalysisEngine()
            self.analysis_engine.set_dataframe(processed_data)
            
            # 기본 분석 수행
            analysis_results = self.analysis_engine.analyze(processed_data)
            
            # 시각화 엔진 초기화
            viz_output_dir = os.path.join(self.output_dir, "visualizations")
            if not os.path.exists(viz_output_dir):
                os.makedirs(viz_output_dir)
                
            self.visualization_engine = VisualizationEngine(output_dir=viz_output_dir)
            self.visualization_engine.set_dataframe(processed_data)
            
            # 파일 이름에서 보고서 이름 추출
            file_basename = os.path.splitext(os.path.basename(file_path))[0]
            report_filename = f"report_{file_basename}_{format}"
            
            # 보고서 생성
            return self.generate_comprehensive_report(
                title=f"{title} - {file_basename}",
                subtitle=f"파일: {file_path}",
                format=format,
                filename=report_filename
            )
        except Exception as e:
            self.logger.error(f"빠른 보고서 생성 중 오류 발생: {str(e)}")
            return "" 