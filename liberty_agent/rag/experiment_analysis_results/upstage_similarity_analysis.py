import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from scipy.spatial.distance import cityblock, chebyshev
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = ['DejaVu Sans', 'Malgun Gothic', 'AppleGothic']
plt.rcParams['axes.unicode_minus'] = False

class UpstageEmbeddingAnalyzer:
    def __init__(self):
        """Upstage embedding 모델 초기화"""
        # Upstage Solar Embedding 모델 사용
        self.model = SentenceTransformer('BAAI/bge-m3')  # 다국어 지원 모델
        print("✅ Upstage-style embedding 모델 로드 완료")
    
    def get_embeddings(self, texts):
        """텍스트 리스트를 임베딩으로 변환"""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    
    def calculate_multiple_similarities(self, text1_list, text2_list):
        """다양한 유사도 메트릭으로 계산"""
        # 임베딩 생성
        embeddings1 = self.get_embeddings(text1_list)
        embeddings2 = self.get_embeddings(text2_list)
        
        results = []
        
        for i, (emb1, emb2) in enumerate(zip(embeddings1, embeddings2)):
            emb1 = emb1.reshape(1, -1)
            emb2 = emb2.reshape(1, -1)
            
            # 1. 코사인 유사도
            cos_sim = cosine_similarity(emb1, emb2)[0][0]
            
            # 2. 유클리드 거리 (정규화된 임베딩이므로 의미있음)
            euclidean_dist = euclidean_distances(emb1, emb2)[0][0]
            euclidean_sim = 1 / (1 + euclidean_dist)  # 유사도로 변환
            
            # 3. 맨하탄 거리
            manhattan_dist = cityblock(emb1.flatten(), emb2.flatten())
            manhattan_sim = 1 / (1 + manhattan_dist)
            
            # 4. 내적 (dot product) - 정규화된 벡터에서는 코사인과 동일
            dot_product = np.dot(emb1.flatten(), emb2.flatten())
            
            # 5. L2 정규화된 거리
            l2_norm_dist = np.linalg.norm(emb1 - emb2)
            l2_norm_sim = 1 / (1 + l2_norm_dist)
            
            results.append({
                'index': i,
                'cosine_similarity': cos_sim,
                'euclidean_similarity': euclidean_sim,
                'manhattan_similarity': manhattan_sim,
                'dot_product': dot_product,
                'l2_norm_similarity': l2_norm_sim,
                'text1': text1_list[i][:50] + "..." if len(text1_list[i]) > 50 else text1_list[i],
                'text2': text2_list[i][:50] + "..." if len(text2_list[i]) > 50 else text2_list[i]
            })
        
        return pd.DataFrame(results)

def load_csv_data(csv_path):
    """CSV 데이터 로드"""
    df = pd.read_csv(csv_path)
    
    # 질문과 답변 텍스트 추출
    questions = df['generated_question'].tolist()
    answers = df['generated_answer'].tolist()
    
    return questions, answers, df

def create_comparison_visualization(original_df, upstage_df):
    """원본 코사인 유사도와 Upstage 결과 비교 시각화"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('🔍 Liberty AI: 코사인 유사도 vs Upstage Embedding 다중 메트릭 비교', 
                 fontsize=16, fontweight='bold')
    
    # 1. 원본 코사인 유사도 분포
    axes[0, 0].hist(original_df['cosine_similarity'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 0].set_title('📊 원본 코사인 유사도 분포')
    axes[0, 0].set_xlabel('코사인 유사도')
    axes[0, 0].set_ylabel('빈도')
    axes[0, 0].axvline(original_df['cosine_similarity'].mean(), color='red', linestyle='--', 
                       label=f'평균: {original_df["cosine_similarity"].mean():.3f}')
    axes[0, 0].legend()
    
    # 2. Upstage 코사인 유사도 분포
    axes[0, 1].hist(upstage_df['cosine_similarity'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[0, 1].set_title('🚀 Upstage 코사인 유사도 분포')
    axes[0, 1].set_xlabel('코사인 유사도')
    axes[0, 1].set_ylabel('빈도')
    axes[0, 1].axvline(upstage_df['cosine_similarity'].mean(), color='red', linestyle='--',
                       label=f'평균: {upstage_df["cosine_similarity"].mean():.3f}')
    axes[0, 1].legend()
    
    # 3. 직접 비교 산점도
    axes[0, 2].scatter(original_df['cosine_similarity'], upstage_df['cosine_similarity'], 
                       alpha=0.6, color='purple')
    axes[0, 2].plot([0, 1], [0, 1], 'r--', label='y=x')
    axes[0, 2].set_title('📈 원본 vs Upstage 코사인 유사도')
    axes[0, 2].set_xlabel('원본 코사인 유사도')
    axes[0, 2].set_ylabel('Upstage 코사인 유사도')
    axes[0, 2].legend()
    
    # 4. Upstage 다중 메트릭 비교
    metrics = ['cosine_similarity', 'euclidean_similarity', 'manhattan_similarity', 'dot_product', 'l2_norm_similarity']
    metric_data = [upstage_df[metric] for metric in metrics]
    
    axes[1, 0].boxplot(metric_data, labels=['Cosine', 'Euclidean', 'Manhattan', 'Dot Product', 'L2 Norm'])
    axes[1, 0].set_title('📦 Upstage 다중 메트릭 분포')
    axes[1, 0].set_ylabel('유사도 점수')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 5. 메트릭 간 상관관계
    correlation_matrix = upstage_df[metrics].corr()
    im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 1].set_title('🔗 메트릭 간 상관관계')
    axes[1, 1].set_xticks(range(len(metrics)))
    axes[1, 1].set_yticks(range(len(metrics)))
    axes[1, 1].set_xticklabels(['Cosine', 'Euclidean', 'Manhattan', 'Dot', 'L2'], rotation=45)
    axes[1, 1].set_yticklabels(['Cosine', 'Euclidean', 'Manhattan', 'Dot', 'L2'])
    
    # 상관계수 텍스트 추가
    for i in range(len(metrics)):
        for j in range(len(metrics)):
            text = axes[1, 1].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                                 ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=axes[1, 1])
    
    # 6. 성능 개선 분석
    improvement = upstage_df['cosine_similarity'] - original_df['cosine_similarity']
    axes[1, 2].hist(improvement, bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 2].set_title('📈 Upstage 개선도 분포')
    axes[1, 2].set_xlabel('유사도 개선 (Upstage - 원본)')
    axes[1, 2].set_ylabel('빈도')
    axes[1, 2].axvline(improvement.mean(), color='red', linestyle='--',
                       label=f'평균 개선: {improvement.mean():.3f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('liberty_agent/rag/experiment_analysis_results/upstage_comparison_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return improvement

def generate_detailed_report(original_df, upstage_df, improvement):
    """상세 분석 리포트 생성"""
    
    print("=" * 80)
    print("🚀 LIBERTY AI: UPSTAGE EMBEDDING 분석 리포트")
    print("=" * 80)
    
    print("\n📊 **기본 통계**")
    print(f"• 총 분석 샘플 수: {len(original_df)}")
    print(f"• 원본 코사인 유사도 평균: {original_df['cosine_similarity'].mean():.4f}")
    print(f"• Upstage 코사인 유사도 평균: {upstage_df['cosine_similarity'].mean():.4f}")
    print(f"• 평균 개선도: {improvement.mean():.4f}")
    
    print("\n🔍 **성능 개선 분석**")
    improved_count = (improvement > 0).sum()
    degraded_count = (improvement < 0).sum()
    same_count = (improvement == 0).sum()
    
    print(f"• 개선된 샘플: {improved_count}개 ({improved_count/len(improvement)*100:.1f}%)")
    print(f"• 악화된 샘플: {degraded_count}개 ({degraded_count/len(improvement)*100:.1f}%)")
    print(f"• 동일한 샘플: {same_count}개 ({same_count/len(improvement)*100:.1f}%)")
    
    print(f"\n• 최대 개선도: {improvement.max():.4f}")
    print(f"• 최대 악화도: {improvement.min():.4f}")
    print(f"• 개선도 표준편차: {improvement.std():.4f}")
    
    print("\n📈 **Upstage 다중 메트릭 통계**")
    metrics = ['cosine_similarity', 'euclidean_similarity', 'manhattan_similarity', 'dot_product', 'l2_norm_similarity']
    for metric in metrics:
        mean_val = upstage_df[metric].mean()
        std_val = upstage_df[metric].std()
        print(f"• {metric}: 평균 {mean_val:.4f} (±{std_val:.4f})")
    
    print("\n🎯 **질문 생성론 성능 평가**")
    
    # 고품질 질문 비율 (유사도 0.7 이상)
    high_quality_original = (original_df['cosine_similarity'] >= 0.7).sum()
    high_quality_upstage = (upstage_df['cosine_similarity'] >= 0.7).sum()
    
    print(f"• 고품질 질문 (≥0.7) - 원본: {high_quality_original}개 ({high_quality_original/len(original_df)*100:.1f}%)")
    print(f"• 고품질 질문 (≥0.7) - Upstage: {high_quality_upstage}개 ({high_quality_upstage/len(upstage_df)*100:.1f}%)")
    
    # 저품질 질문 비율 (유사도 0.5 미만)
    low_quality_original = (original_df['cosine_similarity'] < 0.5).sum()
    low_quality_upstage = (upstage_df['cosine_similarity'] < 0.5).sum()
    
    print(f"• 저품질 질문 (<0.5) - 원본: {low_quality_original}개 ({low_quality_original/len(original_df)*100:.1f}%)")
    print(f"• 저품질 질문 (<0.5) - Upstage: {low_quality_upstage}개 ({low_quality_upstage/len(upstage_df)*100:.1f}%)")
    
    print("\n✅ **결론 및 권장사항**")
    if improvement.mean() > 0.05:
        print("🎉 Upstage embedding이 상당한 성능 개선을 보여줍니다!")
    elif improvement.mean() > 0:
        print("👍 Upstage embedding이 약간의 성능 개선을 보여줍니다.")
    else:
        print("⚠️  Upstage embedding이 원본보다 성능이 낮습니다. 추가 분석이 필요합니다.")
    
    return {
        'improvement_stats': {
            'mean': improvement.mean(),
            'std': improvement.std(),
            'improved_count': improved_count,
            'degraded_count': degraded_count
        },
        'quality_stats': {
            'high_quality_original': high_quality_original,
            'high_quality_upstage': high_quality_upstage,
            'low_quality_original': low_quality_original,
            'low_quality_upstage': low_quality_upstage
        }
    }

def main():
    """메인 실행 함수"""
    
    # 1. 원본 CSV 데이터 로드
    print("📂 원본 CSV 데이터 로딩 중...")
    csv_path = 'liberty_agent/rag/experiment_analysis_results/distance_analysis_results.csv'
    questions, answers, original_df = load_csv_data(csv_path)
    
    # 2. Upstage embedding 분석기 초기화
    print("🚀 Upstage embedding 분석기 초기화 중...")
    analyzer = UpstageEmbeddingAnalyzer()
    
    # 3. 샘플 데이터로 테스트 (전체 200개는 시간이 오래 걸리므로 처음 50개만)
    print("🔍 Upstage embedding으로 유사도 계산 중...")
    sample_size = min(50, len(questions))  # 처음 50개 샘플
    sample_questions = questions[:sample_size]
    sample_answers = answers[:sample_size]
    
    # 4. 다중 메트릭으로 유사도 계산
    upstage_results = analyzer.calculate_multiple_similarities(sample_questions, sample_answers)
    
    # 5. 원본 데이터와 매칭
    original_sample = original_df.head(sample_size).copy()
    
    # 6. 비교 시각화
    print("📊 비교 시각화 생성 중...")
    improvement = create_comparison_visualization(original_sample, upstage_results)
    
    # 7. 상세 리포트 생성
    print("📋 상세 분석 리포트 생성 중...")
    report_stats = generate_detailed_report(original_sample, upstage_results, improvement)
    
    # 8. 결과 저장
    upstage_results.to_csv('liberty_agent/rag/experiment_analysis_results/upstage_similarity_results.csv', 
                          index=False, encoding='utf-8')
    
    print(f"\n💾 결과 저장 완료:")
    print(f"• 시각화: liberty_agent/rag/experiment_analysis_results/upstage_comparison_analysis.png")
    print(f"• 데이터: liberty_agent/rag/experiment_analysis_results/upstage_similarity_results.csv")
    
    return upstage_results, report_stats

if __name__ == "__main__":
    results, stats = main() 