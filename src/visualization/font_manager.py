import logging
from src.visualization.utility_functions import configure_korean_font, reset_font_cache

# 로깅 설정
logger = logging.getLogger(__name__)

# 모듈 초기화시 폰트 설정 시도
def setup_korean_fonts():
    """
    한글 폰트 설정을 시도합니다.
    """
    try:
        result = configure_korean_font()
        if result:
            logger.info("한글 폰트 설정 성공")
        else:
            logger.warning("한글 폰트 설정 실패")
        return result
    except Exception as e:
        logger.error(f"한글 폰트 설정 중 오류 발생: {str(e)}")
        return False 