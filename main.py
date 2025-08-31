import os
import time
import pandas as pd
from crewai import Agent, Task, Crew
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
import json
from collections import defaultdict
import litellm
import math

# Load environment variables
load_dotenv()
mistral_api_key = os.getenv("MISTRAL_API_KEY")
if not mistral_api_key:
    raise ValueError("MISTRAL_API_KEY is not set.")

# Increased the timeout for the LLM connection for long tasks
mistral_llm = ChatMistralAI(model="mistral/mistral-large-latest", max_retries=3, timeout=600)

# --- AGENT DEFINITIONS ---
column_analyzer_agent = Agent(
    role="Data Science Feature Selector",
    goal="Identify valuable columns in a dataset for a business performance dashboard.",
    backstory="You are an expert data scientist skilled in feature selection. You find the best columns for generating insightful business dashboards, ignoring irrelevant IDs.",
    allow_delegation=False, verbose=True, llm=mistral_llm
)
data_type_agent = Agent(
    role="Data Type Classifier",
    goal="Classify a list of column names into 'numerical' and 'categorical' types.",
    backstory="You are a data schema expert. Your only job is to analyze column headers and sample data to determine which columns contain numbers that can be summed, and which contain text categories that can be counted. You output only a clean JSON object.",
    allow_delegation=False, verbose=True, llm=mistral_llm
)
# NEW AGENT for creating the dashboard strategy
dashboard_strategist_agent = Agent(
    role="Dashboard Strategy Specialist",
    goal="Create a strategic plan in JSON format for building a dashboard from aggregated data.",
    backstory="You are a BI consultant. You analyze final aggregated data and decide which metrics should be KPI cards, which column is best for a bar chart, and which is best for a pie chart. Your output is a simple JSON plan for the frontend team.",
    allow_delegation=False, verbose=True, llm=mistral_llm
)
report_writing_agent = Agent(
    role="Senior Business Analyst",
    goal="Write a comprehensive, insightful markdown report based on aggregated sales and customer data.",
    backstory="You are a seasoned analyst known for your ability to distill complex data into clear, actionable business insights. You write detailed reports for executive review.",
    allow_delegation=False, verbose=True, llm=mistral_llm
)


def _run_resilient_crew(crew, task_description, max_retries=3, initial_wait=10):
    """A wrapper function to run a single task with exponential backoff."""
    retries = 0
    while retries < max_retries:
        try:
            result = crew.kickoff()
            return result.raw if hasattr(result, 'raw') else str(result)
        except litellm.RateLimitError:
            retries += 1
            wait_time = initial_wait * (2 ** retries)
            print(f"Warning: Rate limit hit on task '{task_description}'. Retrying in {wait_time} seconds... ({retries}/{max_retries})")
            time.sleep(wait_time)
        except Exception as e:
            print(f"An unexpected error occurred in task '{task_description}': {e}")
            raise e
    raise ValueError(f"Task '{task_description}' failed after {max_retries} retries due to persistent rate limiting.")

# --- THE FIX: A new, smarter function that takes instructions from the AI ---
def _format_data_for_dashboard(aggregation: dict, strategy: dict):
    """
    Takes the aggregated data and an AI-generated strategy to format a clean JSON for the frontend.
    This is 100% reliable because it's guided by the AI's plan.
    """
    dashboard_data = { "kpiCards": [], "barChart": {}, "pieChart": {}, "lineChart": {} }
    
    # 1. Create KPI Cards based on the AI's strategy
    kpi_cols = strategy.get("kpi_columns", [])
    for col in kpi_cols:
        if col in aggregation and 'sum' in aggregation[col]:
            value = aggregation[col]['sum']
            if value > 1_000_000: formatted_value = f"{value/1_000_000:.1f}M"
            elif value > 1_000: formatted_value = f"{value/1_000:.1f}K"
            else: formatted_value = f"{value:.2f}"
            dashboard_data["kpiCards"].append({"title": col.replace('_', ' '), "value": formatted_value})
    
    # 2. Create Bar Chart based on the AI's strategy
    bar_chart_col = strategy.get("bar_chart_column")
    if bar_chart_col and bar_chart_col in aggregation:
        labels = list(aggregation[bar_chart_col].keys())
        data = list(aggregation[bar_chart_col].values())
        dashboard_data["barChart"] = {"labels": labels, "data": data, "title": bar_chart_col.replace('_', ' ')}

    # 3. Create Pie Chart based on the AI's strategy
    pie_chart_col = strategy.get("pie_chart_column")
    if pie_chart_col and pie_chart_col in aggregation:
        labels = list(aggregation[pie_chart_col].keys())
        data = list(aggregation[pie_chart_col].values())
        dashboard_data["pieChart"] = {"labels": labels, "data": data, "title": pie_chart_col.replace('_', ' ')}

    return dashboard_data


def run_crew(file_path: str):
    try:
        # --- Stages 1-3: File Loading, Column & Type Selection ---
        print("Stages 1-3: Loading, selecting, and classifying...")
        df = pd.read_excel(file_path) if file_path.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
        if df.empty: raise ValueError("Input file is empty.")

        headers_str = ", ".join(df.columns.tolist())
        column_selection_task = Task(description=f"Select relevant columns for sales analysis from: {headers_str}.", expected_output="A comma-separated string of column names.", agent=column_analyzer_agent)
        selected_columns_str = _run_resilient_crew(Crew(agents=[column_analyzer_agent], tasks=[column_selection_task]), "Column Selection")
        required_columns = [col.strip() for col in selected_columns_str.split(',') if col.strip()]
        if not required_columns: raise ValueError("AI failed to select columns.")
        print(f"AI selected columns: {required_columns}")
        
        valid_cols = [col for col in required_columns if col in df.columns]
        sample_data_str = df[valid_cols].head().to_string(index=False)
        type_task = Task(description=f"Classify these columns into 'numerical' and 'categorical'. Sample:\n\n{sample_data_str}", expected_output="A JSON object with 'numerical' and 'categorical' keys.", agent=data_type_agent)
        type_result_raw = _run_resilient_crew(Crew(agents=[data_type_agent], tasks=[type_task]), "Data Type Classification")
        
        json_start = type_result_raw.find('{'); json_end = type_result_raw.rfind('}') + 1
        type_json = json.loads(type_result_raw[json_start:json_end])
        numerical_cols = type_json.get('numerical', [])
        categorical_cols = type_json.get('categorical', [])
        print(f"AI classified as Numerical: {numerical_cols}")
        print(f"AI classified as Categorical: {categorical_cols}")

        # --- Stage 4: Deterministic Aggregation ---
        print("Stage 4: Aggregating data...")
        final_aggregation = defaultdict(lambda: defaultdict(float))
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols: df[col] = df[col].astype(str)
        for col in numerical_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numerical_cols, how='all', inplace=True)

        for col, total in df[numerical_cols].sum().to_dict().items():
            if not math.isnan(total): final_aggregation[col]['sum'] = total
        for col in categorical_cols:
            for category, count in df[col].value_counts().nlargest(10).to_dict().items():
                final_aggregation[col][str(category)] = count
        
        if not final_aggregation: raise ValueError("Aggregation failed.")
        print("Aggregation complete.")

        # --- Stage 5: AI Dashboard Strategy & AI Reporting ---
        final_data_str = json.dumps(final_aggregation, indent=2, default=str)
        
        print("Stage 5a: AI is creating the dashboard strategy...")
        strategy_task = Task(
            description=f"Create a plan for a dashboard from this aggregated data. Identify up to 4 numerical columns for KPIs, one categorical column for a bar chart, and a different one for a pie chart. Data:\n\n{final_data_str}",
            expected_output="A JSON object with keys 'kpi_columns' (a list of strings), 'bar_chart_column' (a string), and 'pie_chart_column' (a string).",
            agent=dashboard_strategist_agent
        )
        strategy_result_raw = _run_resilient_crew(Crew(agents=[dashboard_strategist_agent], tasks=[strategy_task]), "Dashboard Strategy")
        
        json_start = strategy_result_raw.find('{'); json_end = strategy_result_raw.rfind('}') + 1
        dashboard_strategy = json.loads(strategy_result_raw[json_start:json_end])
        print(f"AI dashboard strategy: {dashboard_strategy}")

        print("Stage 5b: Generating dashboard data with code...")
        dashboard_data = _format_data_for_dashboard(final_aggregation, dashboard_strategy)

        print("Stage 5c: AI is writing the final text report...")
        report_task = Task(
            description=f"Based on this aggregated data, write a detailed business analysis in markdown. Data:\n\n{final_data_str}",
            expected_output="A comprehensive business report in markdown format.", agent=report_writing_agent
        )
        report_output = _run_resilient_crew(Crew(agents=[report_writing_agent], tasks=[report_task]), "Report Generation", initial_wait=20)
        
        if not report_output or len(report_output.split()) < 20:
            report_output = "## Analysis Complete\nThe AI did not produce a detailed text summary."

        print("Final assembly complete.")
        return {
            "report": report_output,
            "dashboardData": dashboard_data
        }

    except Exception as e:
        print(f"A critical error occurred in run_crew: {e}")
        return {"report": f"## Analysis Failed\n\nA critical error occurred: {e}", "dashboardData": {}}
