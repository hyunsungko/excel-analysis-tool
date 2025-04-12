import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
import logging
import os
import sys
import platform
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple, Union, Optional, Any, Set

# 로깅 설정
logger = logging.getLogger(__name__)

def reset_font_cache():
    """
    matplotlib 폰트 캐시를 초기화합니다.
    이 함수는 새로운 폰트를 등록한 후 캐시를 갱신하는 데 사용됩니다.
    """
    logger.info("폰트 캐시 초기화 시작")
    try:
        # 방법 1: _rebuild 메소드 사용 (matplotlib 버전에 따라 다를 수 있음)
        if hasattr(fm, '_rebuild'):
            logger.info("_rebuild 메소드를 사용하여 폰트 캐시 재구성")
            fm._rebuild()
            return True
        # 방법 2: fontManager 다시 생성
        elif hasattr(fm, 'fontManager'):
            logger.info("fontManager를 다시 생성하여 폰트 캐시 재구성")
            original_font_list_cache = matplotlib.get_cachedir()
            font_cache_path = os.path.join(original_font_list_cache, 'fontlist-v*.json')
            
            # 폰트 캐시 파일 삭제 시도
            try:
                for cache_file in glob.glob(font_cache_path):
                    logger.info(f"폰트 캐시 파일 삭제: {cache_file}")
                    os.remove(cache_file)
            except Exception as e:
                logger.warning(f"폰트 캐시 파일 삭제 중 오류 발생: {e}")
            
            # fontManager 재생성
            fm.fontManager = fm.FontManager()
            logger.info("폰트 매니저 재생성 완료")
            return True
        # 방법 3: findfont 캐시 초기화
        else:
            logger.info("findfont 캐시 초기화")
            fm._get_fontconfig_fonts.cache_clear()
            fm.findfont.cache_clear()
            return True
    except Exception as e:
        logger.error(f"폰트 캐시 초기화 중 오류 발생: {e}")
        return False

def configure_korean_font():
    """
    한글 폰트를 설정합니다.
    
    운영체제에 따라 적절한 한글 폰트를 찾아 matplotlib에 등록합니다.
    한글 폰트가 성공적으로 설정되면 True를 반환합니다.
    
    Returns:
        bool: 한글 폰트 설정 성공 여부
    """
    logger.info("한글 폰트 설정 시작")
    
    # 시스템 확인
    system = platform.system()
    logger.info(f"운영체제: {system}")
    
    # 한글 폰트 키워드 (정규표현식 패턴)
    korean_font_keywords = [
        r'[mM]algun|맑은고딕',      # 맑은 고딕
        r'[nN]anumGothic|나눔고딕',  # 나눔고딕
        r'[nN]anumMyeongjo|나눔명조',  # 나눔명조
        r'[gG]ulim|굴림',          # 굴림
        r'[bB]atang|바탕',         # 바탕
        r'[dD]otum|돋움',          # 돋움
        r'[hH]Y|한양',             # 한양 폰트
        r'[kK]orean|[kK]o',        # Korean 포함 폰트
        r'UnBatang|은바탕',         # 은바탕
        r'UnDotum|은돋움',          # 은돋움
        r'[sS]eoulHangang|서울한강',  # 서울한강체
        r'[sS]eoulNamsan|서울남산',   # 서울남산체
        r'[kK]CC|케이씨씨',          # KCC 폰트
        r'[gG]ungsuh|궁서',         # 궁서
        r'[pP]ilGi|필기',           # 필기체
    ]
    
    # 폰트 경로 목록
    font_paths = []
    
    # 운영체제별 폰트 경로 설정
    if system == 'Windows':
        # Windows 폰트 경로
        windows_font_dirs = [
            r'C:\Windows\Fonts',  # 시스템 폰트 디렉토리
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Microsoft', 'Windows', 'Fonts')  # 사용자 폰트 디렉토리
        ]
        
        for font_dir in windows_font_dirs:
            if os.path.exists(font_dir):
                logger.info(f"Windows 폰트 디렉토리 검색: {font_dir}")
                # ttf, ttc 파일 검색
                font_paths.extend(glob.glob(os.path.join(font_dir, '*.ttf')))
                font_paths.extend(glob.glob(os.path.join(font_dir, '*.ttc')))
    
    elif system == 'Darwin':  # macOS
        # macOS 폰트 경로
        macos_font_dirs = [
            '/Library/Fonts',
            '/System/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
        
        for font_dir in macos_font_dirs:
            if os.path.exists(font_dir):
                logger.info(f"macOS 폰트 디렉토리 검색: {font_dir}")
                font_paths.extend(glob.glob(os.path.join(font_dir, '*.ttf')))
                font_paths.extend(glob.glob(os.path.join(font_dir, '*.ttc')))
                font_paths.extend(glob.glob(os.path.join(font_dir, '*.otf')))
    
    else:  # Linux 및 기타
        # Linux 폰트 경로
        linux_font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts')
        ]
        
        for font_dir in linux_font_dirs:
            if os.path.exists(font_dir):
                logger.info(f"Linux 폰트 디렉토리 검색: {font_dir}")
                # 재귀적으로 모든 폰트 파일 검색
                for ext in ['ttf', 'ttc', 'otf']:
                    pattern = os.path.join(font_dir, '**', f'*.{ext}')
                    font_paths.extend(glob.glob(pattern, recursive=True))
    
    logger.info(f"총 {len(font_paths)}개의 폰트 파일 발견")
    
    # 발견된 폰트 중 한글 폰트만 필터링
    korean_fonts = []
    korean_font_paths = []
    
    for font_path in font_paths:
        font_name = os.path.basename(font_path).lower()
        
        # 한글 폰트 키워드와 일치하는지 확인
        for pattern in korean_font_keywords:
            if re.search(pattern, font_name, re.IGNORECASE):
                logger.info(f"한글 폰트 발견: {font_path}")
                korean_fonts.append(font_name)
                korean_font_paths.append(font_path)
                break
    
    if not korean_font_paths:
        logger.warning("한글 폰트를 찾을 수 없습니다.")
        return False
    
    logger.info(f"한글 폰트 {len(korean_font_paths)}개를 찾았습니다.")
    
    # 사용 가능한 한글 폰트 로드
    loaded_korean_fonts = []
    for font_path in korean_font_paths:
        try:
            font_prop = fm.FontProperties(fname=font_path)
            loaded_korean_fonts.append(font_prop.get_name())
            # 폰트 파일 직접 등록
            fm.fontManager.addfont(font_path)
            logger.info(f"폰트 등록 성공: {font_path} -> {font_prop.get_name()}")
        except Exception as e:
            logger.warning(f"폰트 등록 실패: {font_path} - {e}")
    
    if not loaded_korean_fonts:
        logger.warning("한글 폰트를 등록할 수 없습니다.")
        return False
    
    # 폰트 우선순위 설정 (맑은 고딕, 나눔고딕 등 선호)
    preferred_fonts = [
        "Malgun Gothic", "맑은 고딕", "NanumGothic", "나눔고딕", 
        "Gulim", "굴림", "Dotum", "돋움", "Batang", "바탕", "Gungsuh", "궁서"
    ]
    
    # 로드된 폰트 중에서 선호하는 폰트 찾기
    selected_font = None
    for preferred in preferred_fonts:
        for loaded_font in loaded_korean_fonts:
            if preferred.lower() in loaded_font.lower():
                selected_font = loaded_font
                logger.info(f"선호하는 한글 폰트를 찾았습니다: {selected_font}")
                break
        if selected_font:
            break
    
    # 선호하는 폰트가 없다면 첫 번째 로드된 폰트 사용
    if not selected_font and loaded_korean_fonts:
        selected_font = loaded_korean_fonts[0]
        logger.info(f"첫 번째 로드된 한글 폰트를 사용합니다: {selected_font}")
    
    if not selected_font:
        logger.warning("적합한 한글 폰트를 찾을 수 없습니다.")
        return False
    
    # Matplotlib 폰트 설정
    logger.info(f"기본 한글 폰트 설정: {selected_font}")
    plt.rcParams['font.family'] = 'sans-serif'
    
    # sans-serif 폰트 목록에 한글 폰트 추가
    sans_serif_fonts = ['sans-serif']
    if 'font.sans-serif' in plt.rcParams:
        sans_serif_fonts = plt.rcParams['font.sans-serif']
        if not isinstance(sans_serif_fonts, list):
            sans_serif_fonts = ['sans-serif']
    
    # 한글 폰트를 맨 앞에 추가
    if selected_font not in sans_serif_fonts:
        sans_serif_fonts.insert(0, selected_font)
    plt.rcParams['font.sans-serif'] = sans_serif_fonts
    
    # 마이너스 기호 표시 설정
    plt.rcParams['axes.unicode_minus'] = False
    
    # 직접 특정 폰트 파일 등록 (맑은 고딕을 시도)
    malgun_gothic_paths = [
        r'C:\Windows\Fonts\malgun.ttf',  # Windows
        '/Library/Fonts/AppleGothic.ttf',  # macOS
        '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # Ubuntu
    ]
    
    for font_path in malgun_gothic_paths:
        if os.path.exists(font_path):
            try:
                logger.info(f"폰트 직접 등록: {font_path}")
                font = fm.FontEntry(
                    fname=font_path,
                    name=os.path.splitext(os.path.basename(font_path))[0],
                    style='normal',
                    variant='normal',
                    weight='normal',
                    stretch='normal'
                )
                fm.fontManager.ttflist.insert(0, font)
            except Exception as e:
                logger.warning(f"폰트 직접 등록 실패: {font_path} - {e}")
    
    # 폰트 캐시 갱신
    reset_font_cache()
    
    # 현재 설정된 폰트 확인
    logger.info(f"최종 설정된 font.family: {plt.rcParams['font.family']}")
    logger.info(f"최종 설정된 font.sans-serif: {plt.rcParams['font.sans-serif'][:5]}...")
    
    return True

# 파일 이름 생성 유틸리티
def get_filename(prefix: str, ext: str = 'png') -> str:
    """
    시간 기반 고유한 파일명 생성
    
    Args:
        prefix (str): 파일명 접두사
        ext (str): 파일 확장자
        
    Returns:
        str: 생성된 파일명
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{ext}" 