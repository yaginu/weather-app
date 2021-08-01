# Weather app

24時間分の気象データから、翌日の1時間ごとの気温を予測するシステムです。
1日1回のスクレイピングでデータを追加収集し、予測値を出します。
GitHubレポジトリへの新しいtagのpushをトリガーとして、モデルの訓練・評価・デプロイを実行します。

# 目的

継続的な機械学習パイプラインの運用を実現するために、コードの変更からデプロイまでを自動で行えるようにしました。

# パイプラインの詳細

こちらの記事に詳細を記しました。
https://qiita.com/yagi615/items/ea9b150c578b863d0ac8

# 使用技術

- Python 3.7
- TensorFlow 2.5.0
- TensorFlowr Tansform 1.1.0
- TensorFlow Addons 0.13.0
- Kubeflow Pipelines
- GCP
	- AI Platform
	- Cloud Storage
	- BigQuery
	- Dataflow
	- Cloud Build
	- Cloud Functions
	- Cloud Scheduler
	- Pub/Sub



