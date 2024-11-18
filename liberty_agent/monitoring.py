from typing import List, Dict
from datetime import datetime
class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "search_latency": [],
            "answer_quality": [],
            "rewrite_counts": [],
            "relevance_scores": []
        }
    
    def track_search(self, latency: float, results: List[Dict]):
        self.metrics["search_latency"].append(latency)
        self.metrics["relevance_scores"].extend([r.score for r in results])
    
    def track_answer(self, quality_score: float, rewrite_count: int):
        self.metrics["answer_quality"].append(quality_score)
        self.metrics["rewrite_counts"].append(rewrite_count) 