#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Excel Analyzer의 명령줄 인터페이스
"""

import argparse
import os
import sys
from pathlib import Path

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 모듈 불러오기
from src.utils.data_loader import DataLoader
from src.core.data_processor import DataProcessor
from src.core.analysis_engine import AnalysisEngine
from src.visualization.visualization_engine import VisualizationEngine
from src.reporting.report_engine import ReportEngine


def parse_args():
    """명령줄 인수 파싱"""
    parser = argparse.ArgumentParser(
        description="엑셀 파일 분석 및 보고서 생성 도구"
    )
    
    parser.add_argument(
        '--file', '-f',
        required=True,
        help='분석할 엑셀 파일 경로'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='./output',
        help='결과물 저장 디렉토리 (기본값: ./output)'
    )
    
    parser.add_argument(
        '--report-format',
        choices=['html', 'text', 'all'],
        default='html',
        help='보고서 형식 (기본값: html)'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='시각화 생성 비활성화'
    )
    
    parser.add_argument(
        '--sheet',
        type=str,
        help='분석할 시트 이름 (기본값: 첫번째 시트)'
    )
    
    return parser.parse_args()


def analyze_file(file_path, output_dir, sheet_name=None, include_viz=True, report_format='html'):
    """
    엑셀 파일 분석 및 보고서 생성
    
    Args:
        file_path (str): 엑셀 파일 경로
        output_dir (str): 출력 디렉토리
        sheet_name (str, optional): 시트 이름
        include_viz (bool): 시각화 포함 여부
        report_format (str): 보고서 형식
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    viz_dir = os.path.join(output_dir, 'viz')
    os.makedirs(viz_dir, exist_ok=True)
    
    print("1. 엑셀 파일 로드 중...")
    loader = DataLoader()
    
    # 시트 이름이 지정되지 않았으면 첫 번째 시트 사용
    if sheet_name is None:
        sheet_names = loader.get_sheet_names(file_path)
        if not sheet_names:
            print("오류: 엑셀 파일에 시트가 없습니다")
            return 1
        sheet_name = sheet_names[0]
        print(f"- 첫 번째 시트 사용: {sheet_name}")
    
    # 데이터 로드
    df = loader.load_file(file_path, sheet_name=sheet_name)
    print(f"- 데이터 로드 완료: {len(df)} 행, {len(df.columns)} 열")
    
    print("2. 데이터 처리 중...")
    processor = DataProcessor()
    processed_data = processor.process(df)
    print("- 데이터 처리 완료")
    
    print("3. 데이터 분석 중...")
    analyzer = AnalysisEngine()
    analysis_results = analyzer.analyze(processed_data)
    print("- 데이터 분석 완료")
    
    if include_viz:
        print("4. 시각화 생성 중...")
        visualizer = VisualizationEngine(output_dir=viz_dir)
        viz_results = visualizer.generate_visualizations(processed_data, analysis_results)
        print(f"- 시각화 생성 완료: {len(viz_results)} 개 생성됨")
    else:
        viz_results = []
    
    print("5. 보고서 생성 중...")
    reporter = ReportEngine()
    
    # 보고서 메타데이터
    report_meta = {
        'title': f"{os.path.basename(file_path)} 분석 보고서",
        'subtitle': f"시트: {sheet_name}",
        'author': "Excel 분석 시스템",
    }
    
    # 보고서 형식별 생성
    if report_format in ['html', 'all']:
        html_report = reporter.generate_report(
            data=processed_data,
            analysis_results=analysis_results,
            visualizations=viz_results if include_viz else None,
            output_format='html',
            metadata=report_meta
        )
        html_path = os.path.join(output_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        print(f"- HTML 보고서 생성 완료: {html_path}")
    
    if report_format in ['text', 'all']:
        text_report = reporter.generate_report(
            data=processed_data,
            analysis_results=analysis_results,
            visualizations=None,  # 텍스트 보고서는 시각화 포함 안 함
            output_format='text',
            metadata=report_meta
        )
        text_path = os.path.join(output_dir, 'report.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(text_report)
        print(f"- 텍스트 보고서 생성 완료: {text_path}")
    
    print("\n분석 및 보고서 생성 완료!")
    return 0


def main(args=None):
    """CLI 메인 함수"""
    if args is None:
        args = parse_args()
    else:
        # args가 문자열 리스트로 전달된 경우, 파서를 통해 처리
        parser = argparse.ArgumentParser(
            description="엑셀 파일 분석 및 보고서 생성 도구"
        )
        
        parser.add_argument(
            '--file', '-f',
            required=True,
            help='분석할 엑셀 파일 경로'
        )
        
        parser.add_argument(
            '--output', '-o',
            default='./output',
            help='결과물 저장 디렉토리 (기본값: ./output)'
        )
        
        parser.add_argument(
            '--report-format',
            choices=['html', 'text', 'all'],
            default='html',
            help='보고서 형식 (기본값: html)'
        )
        
        parser.add_argument(
            '--no-viz',
            action='store_true',
            help='시각화 생성 비활성화'
        )
        
        parser.add_argument(
            '--sheet',
            type=str,
            help='분석할 시트 이름 (기본값: 첫번째 시트)'
        )
        
        args = parser.parse_args(args)
    
    # 파일 존재 확인
    if not os.path.exists(args.file):
        print(f"오류: 파일이 존재하지 않습니다: {args.file}")
        return 1
    
    print(f"파일 분석: {args.file}")
    print(f"결과 저장 경로: {args.output}")
    print(f"보고서 형식: {args.report_format}")
    print(f"시각화 생성: {'비활성화' if args.no_viz else '활성화'}")
    
    # 분석 실행
    return analyze_file(
        args.file, 
        args.output, 
        sheet_name=args.sheet,
        include_viz=not args.no_viz, 
        report_format=args.report_format
    )


if __name__ == "__main__":
    sys.exit(main()) 