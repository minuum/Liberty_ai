import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import font_manager
import matplotlib.patches as patches

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

def load_and_process_data(csv_path):
    """CSV 데이터를 로드하고 처리"""
    df = pd.read_csv(csv_path)
    
    # 필요한 컬럼만 선택
    processed_data = []
    
    for _, row in df.iterrows():
        processed_data.append({
            'doc_id': row['query_id'].split('_')[2],  # doc_0, doc_1 등에서 숫자 추출
            'difficulty': row['query_id'].split('_')[1],  # difficulty_1, difficulty_2 등에서 숫자 추출
            'question_similarity': row['cosine_similarity'],  # GT Q vs Qi
            'answer_similarity': row['cosine_similarity'],    # GT A vs Ai (동일한 값 사용)
            'law_type': row['original_category'],
            'complexity': row['transformed_query_difficulty'],
            'case_number': row['case_no']
        })
    
    return pd.DataFrame(processed_data)

def create_similarity_plot(df, doc_id=None, title_suffix=""):
    """유사성 산점도 생성"""
    
    if doc_id is not None:
        plot_data = df[df['doc_id'] == str(doc_id)].copy()
        if plot_data.empty:
            print(f"문서 {doc_id}에 대한 데이터가 없습니다.")
            return
    else:
        plot_data = df.copy()
    
    # 그래프 설정
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 색상 및 마커 설정
    colors = ['#4472C4', '#E15759', '#70AD47', '#FFC000', '#9A6FB0']
    markers = ['o', 's', '^', 'D', 'v']
    
    # 난이도별로 그룹화하여 플롯
    difficulties = sorted(plot_data['difficulty'].unique())
    
    for i, diff in enumerate(difficulties):
        diff_data = plot_data[plot_data['difficulty'] == diff]
        
        ax.scatter(
            diff_data['question_similarity'], 
            diff_data['answer_similarity'],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=100,
            alpha=0.7,
            label=f'Question {diff}',
            edgecolors='black',
            linewidth=0.5
        )
    
    # 임계값 선 추가 (0.8)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Threshold (0.8)')
    ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # 그래프 스타일링
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel('Question Similarity (GT Q vs Qi)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Answer Similarity (GT A vs Ai)', fontsize=12, fontweight='bold')
    
    # 제목 설정
    if doc_id is not None:
        case_info = plot_data.iloc[0] if not plot_data.empty else None
        if case_info is not None:
            title = f"Case {doc_id} - {case_info['law_type']}\n{case_info['case_number']}"
        else:
            title = f"Case {doc_id}"
    else:
        title = f"전체 문서 유사성 분석{title_suffix}"
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # 격자 추가
    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    # 범례 설정
    legend = ax.legend(
        title='Methods & Questions',
        title_fontsize=12,
        fontsize=10,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        frameon=True,
        fancybox=True,
        shadow=True
    )
    legend.get_title().set_fontweight('bold')
    
    # 레이아웃 조정
    plt.tight_layout()
    
    return fig, ax

def create_multiple_case_plots(df, num_cases=4):
    """여러 케이스를 한 번에 보여주는 서브플롯"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    doc_ids = sorted(df['doc_id'].unique())[:num_cases]
    
    for i, doc_id in enumerate(doc_ids):
        ax = axes[i]
        plot_data = df[df['doc_id'] == doc_id].copy()
        
        # 색상 및 마커 설정
        colors = ['#4472C4', '#E15759', '#70AD47', '#FFC000', '#9A6FB0']
        markers = ['o', 's', '^', 'D', 'v']
        
        difficulties = sorted(plot_data['difficulty'].unique())
        
        for j, diff in enumerate(difficulties):
            diff_data = plot_data[plot_data['difficulty'] == diff]
            
            ax.scatter(
                diff_data['question_similarity'], 
                diff_data['answer_similarity'],
                c=colors[j % len(colors)],
                marker=markers[j % len(markers)],
                s=80,
                alpha=0.7,
                label=f'Q{diff}',
                edgecolors='black',
                linewidth=0.5
            )
        
        # 임계값 선
        ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        ax.axvline(x=0.8, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
        
        # 스타일링
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel('Question Similarity', fontsize=10)
        ax.set_ylabel('Answer Similarity', fontsize=10)
        
        # 제목
        case_info = plot_data.iloc[0] if not plot_data.empty else None
        if case_info is not None:
            title = f"Case {doc_id} - {case_info['law_type']}"
        else:
            title = f"Case {doc_id}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
    
    plt.suptitle('Liberty AI 질문 생성론 성능 분석', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def analyze_performance_stats(df):
    """성능 통계 분석"""
    
    print("=" * 60)
    print("📊 Liberty AI 질문 생성론 성능 분석 결과")
    print("=" * 60)
    
    # 전체 통계
    total_questions = len(df)
    avg_q_similarity = df['question_similarity'].mean()
    avg_a_similarity = df['answer_similarity'].mean()
    
    print(f"\n🔍 전체 통계:")
    print(f"  • 총 질문 수: {total_questions:,}개")
    print(f"  • 평균 질문 유사성: {avg_q_similarity:.3f}")
    print(f"  • 평균 답변 유사성: {avg_a_similarity:.3f}")
    
    # 임계값 이상 비율
    threshold = 0.8
    above_threshold_q = (df['question_similarity'] >= threshold).sum()
    above_threshold_a = (df['answer_similarity'] >= threshold).sum()
    
    print(f"\n🎯 임계값({threshold}) 이상 성능:")
    print(f"  • 질문 유사성: {above_threshold_q}/{total_questions} ({above_threshold_q/total_questions*100:.1f}%)")
    print(f"  • 답변 유사성: {above_threshold_a}/{total_questions} ({above_threshold_a/total_questions*100:.1f}%)")
    
    # 난이도별 분석
    print(f"\n📈 난이도별 성능:")
    for diff in sorted(df['difficulty'].unique()):
        diff_data = df[df['difficulty'] == diff]
        avg_sim = diff_data['question_similarity'].mean()
        count = len(diff_data)
        print(f"  • 난이도 {diff}: {avg_sim:.3f} (n={count})")
    
    # 법률 분야별 분석
    print(f"\n⚖️ 법률 분야별 성능:")
    for law_type in df['law_type'].unique():
        law_data = df[df['law_type'] == law_type]
        avg_sim = law_data['question_similarity'].mean()
        count = len(law_data)
        print(f"  • {law_type}: {avg_sim:.3f} (n={count})")

def main():
    """메인 실행 함수"""
    
    # 데이터 로드
    csv_path = "liberty_agent/rag/experiment_analysis_results/distance_analysis_results.csv"
    df = load_and_process_data(csv_path)
    
    # 성능 통계 출력
    analyze_performance_stats(df)
    
    # 1. 전체 데이터 시각화
    print("\n📊 전체 데이터 시각화 생성 중...")
    fig1, ax1 = create_similarity_plot(df, title_suffix=" - 전체 문서")
    plt.savefig("liberty_agent/rag/experiment_analysis_results/overall_similarity_plot.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 개별 케이스 시각화 (첫 번째 문서)
    print("\n📊 개별 케이스 시각화 생성 중...")
    fig2, ax2 = create_similarity_plot(df, doc_id=0)
    plt.savefig("liberty_agent/rag/experiment_analysis_results/case_0_similarity_plot.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 다중 케이스 비교
    print("\n📊 다중 케이스 비교 시각화 생성 중...")
    fig3 = create_multiple_case_plots(df, num_cases=4)
    plt.savefig("liberty_agent/rag/experiment_analysis_results/multiple_cases_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 모든 시각화가 완료되었습니다!")
    print("📁 저장된 파일:")
    print("  • overall_similarity_plot.png")
    print("  • case_0_similarity_plot.png") 
    print("  • multiple_cases_comparison.png")

if __name__ == "__main__":
    main() 