"""
운영체제별 한글 폰트 설정 유틸리티
matplotlib에서 한글이 깨지지 않도록 적절한 폰트를 설정합니다.
"""

import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def setup_korean_font():
    """운영체제에 맞는 한글 폰트 설정"""
    system = platform.system()
    
    try:
        if system == "Darwin":  # macOS
            # macOS 한글 폰트 우선순위
            mac_fonts = [
                'AppleGothic', 
                'Apple SD Gothic Neo', 
                'Apple Gothic',
                'Helvetica'
            ]
            
            # 사용 가능한 폰트 찾기
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            selected_font = None
            
            for font in mac_fonts:
                if any(font in af for af in available_fonts):
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.family'] = [selected_font]
                print(f"🍎 macOS 폰트 설정: {selected_font}")
            else:
                plt.rcParams['font.family'] = ['AppleGothic']
                print("🍎 macOS 기본 폰트 설정: AppleGothic")
                
        elif system == "Windows":  # Windows
            # Windows 한글 폰트 우선순위
            win_fonts = [
                'Malgun Gothic',
                'Microsoft YaHei', 
                'Arial Unicode MS',
                'Gulim',
                'Dotum',
                'Arial'
            ]
            
            # 사용 가능한 폰트 찾기
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            selected_font = None
            
            for font in win_fonts:
                if any(font in af for af in available_fonts):
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.family'] = [selected_font]
                print(f"🪟 Windows 폰트 설정: {selected_font}")
            else:
                plt.rcParams['font.family'] = ['Malgun Gothic']
                print("🪟 Windows 기본 폰트 설정: Malgun Gothic")
                
        else:  # Linux 등
            # Linux 한글 폰트 우선순위
            linux_fonts = [
                'Noto Sans CJK KR',
                'Noto Sans Korean',
                'DejaVu Sans',
                'Liberation Sans',
                'Ubuntu',
                'Arial'
            ]
            
            # 사용 가능한 폰트 찾기
            available_fonts = [f.name for f in fm.fontManager.ttflist]
            selected_font = None
            
            for font in linux_fonts:
                if any(font in af for af in available_fonts):
                    selected_font = font
                    break
            
            if selected_font:
                plt.rcParams['font.family'] = [selected_font]
                print(f"🐧 Linux 폰트 설정: {selected_font}")
            else:
                plt.rcParams['font.family'] = ['DejaVu Sans']
                print("🐧 Linux 기본 폰트 설정: DejaVu Sans")
        
        # 마이너스 기호 깨짐 방지
        plt.rcParams['axes.unicode_minus'] = False
        
        # 폰트 캐시 새로고침
        fm._rebuild()
        
        return True
        
    except Exception as e:
        print(f"⚠️ 폰트 설정 중 오류: {e}")
        # 안전한 기본 설정
        plt.rcParams['font.family'] = ['Arial']
        plt.rcParams['axes.unicode_minus'] = False
        print("🔧 기본 폰트로 설정: Arial")
        return False

def get_available_korean_fonts():
    """시스템에서 사용 가능한 한글 폰트 목록 반환"""
    korean_keywords = [
        'Gothic', 'Gulim', 'Dotum', 'Batang', 'Gungsuh',
        'Malgun', 'YaHei', 'Apple', 'Noto', 'CJK', 'Korean'
    ]
    
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    korean_fonts = []
    
    for font in available_fonts:
        if any(keyword in font for keyword in korean_keywords):
            korean_fonts.append(font)
    
    return sorted(list(set(korean_fonts)))

def test_korean_display():
    """한글 표시 테스트"""
    import matplotlib.pyplot as plt
    
    # 테스트 텍스트
    test_text = "한글 폰트 테스트: 가나다라마바사"
    
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, test_text, fontsize=16, ha='center', va='center')
    plt.title("한글 폰트 표시 테스트")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    print(f"현재 설정된 폰트: {plt.rcParams['font.family']}")

if __name__ == "__main__":
    print("🎨 한글 폰트 설정 유틸리티")
    print("=" * 40)
    
    # 폰트 설정
    setup_korean_font()
    
    # 사용 가능한 한글 폰트 출력
    korean_fonts = get_available_korean_fonts()
    print(f"\n📋 사용 가능한 한글 폰트 ({len(korean_fonts)}개):")
    for font in korean_fonts[:10]:  # 상위 10개만 표시
        print(f"  - {font}")
    
    if len(korean_fonts) > 10:
        print(f"  ... 외 {len(korean_fonts) - 10}개")
    
    # 테스트 실행 여부 확인
    test_choice = input("\n한글 표시 테스트를 실행하시겠습니까? (y/n): ").strip().lower()
    if test_choice == 'y':
        test_korean_display() 