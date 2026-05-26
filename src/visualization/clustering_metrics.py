import subprocess
import sys

NAMES = ["game24", "hle", "hotpotqa", "humaneval", "logiqa", "matharena", "mimic_rrs", "mtsamples_procedures", "pubmed_qa", "scibench", "sonnetwriting"]

for name in NAMES:
    subprocess.run(
        [sys.executable,
        "./src/visualization/clustering_utils/semantic_trajectory_metrics.py",
        f"./logs/raw_calls/repeats/gpt-4.1-nano/{name}",
        "--backend", "tfidf",
        "--output-dir", f"./results/clustering/semantic_metrics/tfidf/{name}"],
        check=True
    )

    print(f"Semantic Trajectory metrics computed successfully. ({name})")

    subprocess.run(
        [sys.executable,
        "./src/visualization/clustering_utils/cluster_trajectory_metrics.py",
        f"./logs/raw_calls/repeats/gpt-4.1-nano/{name}",
        "--backend", "tfidf",
        "--parsed-dir", f"./results/clustering/semantic_metrics/tfidf/{name}",
        "--output-dir", f"./results/clustering/cluster_metrics/tfidf/{name}"],
        check=True
    )

    print(f"Clustering Trajectory metrics computed successfully. ({name})")

    subprocess.run(
        [sys.executable,
        "./src/visualization/clustering_utils/cluster_visualizations.py",
        "--results-dir", f"./results/clustering/cluster_metrics/tfidf/{name}",
        "--proposals-csv", f"./results/clustering/semantic_metrics/tfidf/{name}/proposals.csv",
        "--output-dir", f"./results/clustering/plots/tfidf/{name}"],
        check=True
    )

    print(f"Clustering Trajectory metrics computed successfully. ({name})")

# # 1. Compute semantic trajectory metrics from the raw conversation-log ZIP
# py .\semantic_metrics\semantic_trajectory_metrics.py `
#   .\input_logs\mimic_rrs.zip `
#   --backend tfidf `
#   --output-dir .\semantic_metrics\results\tfidf

# # 2. Compute pooled clustering metrics from the newly parsed proposals
# py .\cluster_metrics\cluster_trajectory_metrics.py `
#   .\input_logs\mimic_rrs.zip `
#   --backend tfidf `
#   --parsed-dir .\semantic_metrics\results\tfidf `
#   --output-dir .\cluster_metrics\results\clustering_tfidf

# # 3. Generate every visualization
# py .\cluster_metrics\cluster_visualizations.py `
#   --results-dir .\cluster_metrics\results\clustering_tfidf `
#   --proposals-csv .\semantic_metrics\results\tfidf\proposals.csv `
#   --output-dir .\cluster_metrics\results\clustering_tfidf\figures