import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
import os

class TextAnalyzer:
    """
    주관식(텍스트) 데이터 분석을 담당하는 클래스
    단어 빈도 분석, 키워드 추출, 텍스트 요약 등의 기능 제공
    """
    
    def __init__(self):
        """TextAnalyzer 초기화"""
        self.logger = logging.getLogger(__name__)
        
        # 한국어 불용어 (stopwords) 정의
        self.korean_stop_words = {
            '이', '그', '저', '것', '등', '및', '를', '을', '에', '에서', 
            '은', '는', '이다', '있다', '하다', '이런', '저런', '그런',
            '와', '과', '으로', '로', '의', '가', '이', '한', '하는', '할', '하여',
            '한다', '이고', '입니다', '습니다', '었습니다', '았습니다'
        }
        
    def _tokenize_text(self, text: str) -> List[str]:
        """
        텍스트를 단어로 분리
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            List[str]: 분리된 단어 목록
        """
        # 특수문자 및 숫자 제거
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        
        # 공백 기준으로 단어 분리 (한국어는 형태소 분석기가 있으면 더 좋음)
        words = text.split()
        
        # 불용어 및 짧은 단어 제거 (2글자 이상만 유지)
        words = [w for w in words if len(w) > 1 and w not in self.korean_stop_words]
        
        return words
        
    def summarize_text_column(self, df: pd.DataFrame, column_name: str) -> Dict[str, Any]:
        """
        주관식 텍스트 데이터 분석 및 요약
        
        Args:
            df (pd.DataFrame): 분석할 데이터프레임
            column_name (str): 분석할 열 이름
            
        Returns:
            Dict[str, Any]: 텍스트 분석 결과
        """
        if column_name not in df.columns:
            self.logger.warning(f"열 '{column_name}'이(가) 데이터프레임에 존재하지 않습니다.")
            return None
            
        texts = df[column_name].dropna().astype(str).tolist()
        if not texts:
            self.logger.warning(f"열 '{column_name}'에 분석할 텍스트가 없습니다.")
            return None
            
        # 텍스트 개수, 평균 길이 계산
        response_count = len(texts)
        avg_length = sum(len(text) for text in texts) / response_count if response_count else 0
        
        # 단어 빈도 분석
        all_words = []
        for text in texts:
            words = self._tokenize_text(text)
            all_words.extend(words)
            
        word_freq = Counter(all_words).most_common(30)
        
        # 키워드 추출 (상위 10개 단어)
        keywords = [word for word, _ in word_freq[:10]]
        
        # 대표적인 응답 샘플 (가장 긴 5개 응답 선택)
        samples = sorted(texts, key=len, reverse=True)[:5]
        
        return {
            'response_count': response_count,
            'avg_length': avg_length,
            'word_freq': word_freq,
            'keywords': keywords,
            'samples': samples
        }
        
    def create_text_visualizations(self, text_data: Dict[str, Any], column_name: str, 
                                  output_dir: str = 'output') -> List[Dict[str, str]]:
        """
        주관식 텍스트 데이터 시각화
        
        Args:
            text_data (Dict[str, Any]): summarize_text_column 함수의 결과
            column_name (str): 분석한 열 이름
            output_dir (str): 시각화 결과 저장 디렉토리
            
        Returns:
            List[Dict[str, str]]: 생성된 시각화 파일 정보
        """
        if not text_data:
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        visualizations = []
        
        # 1. 상위 키워드 빈도 차트
        try:
            words, counts = zip(*text_data['word_freq'][:15])
            
            plt.figure(figsize=(10, 6))
            plt.barh(list(reversed(words)), list(reversed(counts)), color='skyblue')
            plt.title(f"'{column_name}' 주요 키워드")
            plt.xlabel('빈도수')
            plt.tight_layout()
            
            keywords_path = os.path.join(output_dir, f"keywords_{column_name.replace(' ', '_')}.png")
            plt.savefig(keywords_path, dpi=100)
            plt.close()
            
            visualizations.append({
                'type': 'keywords',
                'title': f"{column_name} 주요 키워드",
                'path': keywords_path,
                'description': f'{column_name} 열의 주요 키워드를 보여주는 차트입니다.'
            })
        except Exception as e:
            self.logger.error(f"키워드 차트 생성 중 오류 발생: {e}")
        
        # 2. 텍스트 요약 보고서
        try:
            plt.figure(figsize=(10, 8))
            plt.axis('off')
            
            report = f"# {column_name} 주관식 응답 분석\n\n"
            report += f"- 총 응답 수: {text_data['response_count']}개\n"
            report += f"- 평균 응답 길이: {text_data['avg_length']:.1f}자\n\n"
            
            report += "## 주요 키워드\n"
            for word, count in text_data['word_freq'][:10]:
                report += f"- {word}: {count}회\n"
            
            report += "\n## 대표 응답 사례\n"
            for i, sample in enumerate(text_data['samples'], 1):
                # 너무 긴 텍스트는 자름
                display_sample = sample[:200] + "..." if len(sample) > 200 else sample
                report += f"{i}. {display_sample}\n\n"
            
            plt.text(0.05, 0.95, report, va='top', ha='left', fontsize=12, 
                    transform=plt.gca().transAxes, family='Malgun Gothic')
            
            summary_path = os.path.join(output_dir, f"summary_{column_name.replace(' ', '_')}.png")
            plt.savefig(summary_path, dpi=100)
            plt.close()
            
            visualizations.append({
                'type': 'summary',
                'title': f"{column_name} 응답 요약",
                'path': summary_path,
                'description': f'{column_name} 열의 주관식 응답을 요약한 보고서입니다.'
            })
        except Exception as e:
            self.logger.error(f"텍스트 요약 보고서 생성 중 오류 발생: {e}")
            
        return visualizations 