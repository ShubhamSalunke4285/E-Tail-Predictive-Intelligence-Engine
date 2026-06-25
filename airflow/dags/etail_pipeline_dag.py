"""Apache Airflow DAG for the E-Tail Predictive Intelligence Engine.

This is the production orchestration layer. The same stage functions used by
`python run.py pipeline` are wired up here as Airflow tasks, giving you
scheduling, retries, backfills and a task-level dependency graph:

    extract_transform_load
        ├── train_recommender ─┐
        └── forecast_sales ────┴── build_kpis

Deploy by pointing AIRFLOW_HOME's dags_folder at this directory (or copy this
file into it). Airflow is intentionally NOT a hard dependency of the project;
the pipeline runs standalone without it.
"""
from __future__ import annotations

from datetime import datetime, timedelta

# Make the project importable from within the Airflow worker.
import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from airflow import DAG                       # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

from src import pipeline                      # noqa: E402

default_args = {
    "owner": "data-engineering",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "depends_on_past": False,
}

with DAG(
    dag_id="etail_predictive_intelligence",
    description="ETL + collaborative filtering + ARIMA forecasting pipeline",
    default_args=default_args,
    schedule="0 2 * * *",            # daily at 02:00
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["etl", "ml", "forecasting", "recommendations"],
) as dag:

    recommend = PythonOperator(
        task_id="etl_recommendations",
        python_callable=pipeline.stage_etl_recommendations,
    )
    forecast = PythonOperator(
        task_id="forecast_sales",
        python_callable=pipeline.stage_forecast_sales,
    )
    inventory = PythonOperator(
        task_id="inventory_engine",
        python_callable=pipeline.stage_inventory,
    )
    kpis = PythonOperator(
        task_id="build_kpis",
        python_callable=pipeline.stage_build_kpis,
    )

    [recommend, forecast, inventory] >> kpis
