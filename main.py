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
    goal="Classify a list of column names into 'numerical' and 'categorical' types based on their names and a sample of the data.",
    backstory="You are a data schema expert. Your only job is to analyze column headers and sample data to determine which columns contain numbers that can be summed, and which contain text categories that can be counted. You output only a clean JSON object.",
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

def _format_data_for_dashboard(aggregation: dict, numerical_cols: list, categorical_cols: list):
    """
    Takes the aggregated data and formats it into a clean JSON structure for the frontend.
    This is 100% reliable and removes the AI from this critical step.
    """
    dashboard_data = { "kpiCards": [], "barChart": {}, "pieChart": {}, "lineChart": {} }
    kpi_candidates = ["Revenue", "Lifetime_Value", "Average_Order_Value", "Purchase_Frequency"]
    
    for col in kpi_candidates:
        if col in aggregation and 'sum' in aggregation[col]:
            value = aggregation[col]['sum']
            if value > 1_000_000: formatted_value = f"{value/1_000_000:.1f}M"
            elif value > 1_000: formatted_value = f"{value/1_000:.1f}K"
            else: formatted_value = f"{value:.2f}"
            dashboard_data["kpiCards"].append({"title": col.replace('_', ' '), "value": formatted_value})
    
    dashboard_data["kpiCards"] = dashboard_data["kpiCards"][:4]

    # Find the best categorical column for the bar chart (most distinct values)
    bar_chart_col = None
    if categorical_cols:
        bar_candidates = [c for c in categorical_cols if c not in ['Season', 'Preferred_Purchase_Times']]
        if bar_candidates:
            bar_chart_col = max(bar_candidates, key=lambda col: len(aggregation.get(col, {})))
        else: 
            bar_chart_col = max(categorical_cols, key=lambda col: len(aggregation.get(col, {})))

        if bar_chart_col in aggregation and len(aggregation[bar_chart_col]) > 1:
            labels = list(aggregation[bar_chart_col].keys())
            data = list(aggregation[bar_chart_col].values())
            dashboard_data["barChart"] = {"labels": labels, "data": data, "title": bar_chart_col.replace('_', ' ')}

    # Find a different categorical column for the pie chart
    pie_chart_col = None
    pie_candidates = ['Season', 'Most_Frequent_Category', 'Region']
    for col in pie_candidates:
        if col in categorical_cols and col != bar_chart_col and col in aggregation and len(aggregation[col]) > 1:
            pie_chart_col = col
            break
    
    if pie_chart_col:
        labels = list(aggregation[pie_chart_col].keys())
        data = list(aggregation[pie_chart_col].values())
        dashboard_data["pieChart"] = {"labels": labels, "data": data, "title": pie_chart_col.replace('_', ' ')}

    return dashboard_data


def run_crew(file_path: str):
    try:
        # --- Stage 1 & 2: File Loading and AI Column Selection ---
        print("Stage 1 & 2: Loading file and selecting columns...")
        df = pd.read_excel(file_path) if file_path.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path, on_bad_lines='skip', low_memory=False)
        if df.empty: raise ValueError("Input file is empty.")

        headers_str = ", ".join(df.columns.tolist())
        column_selection_task = Task(
            description=f"Select relevant columns for sales analysis from: {headers_str}.",
            expected_output="A comma-separated string of column names.", agent=column_analyzer_agent
        )
        selected_columns_str = _run_resilient_crew(Crew(agents=[column_analyzer_agent], tasks=[column_selection_task]), "Column Selection")
        required_columns = [col.strip() for col in selected_columns_str.split(',') if col.strip()]
        if not required_columns: raise ValueError("AI failed to select columns.")
        print(f"AI selected columns: {required_columns}")
        
        # --- Stage 3: AI Data Type Classification ---
        print("Stage 3: Classifying data types...")
        valid_cols = [col for col in required_columns if col in df.columns]
        sample_data_str = df[valid_cols].head().to_string(index=False)
        type_task = Task(
            description=f"Classify these columns into 'numerical' and 'categorical'. Sample:\n\n{sample_data_str}",
            expected_output="A JSON object with 'numerical' and 'categorical' keys.", agent=data_type_agent
        )
        type_result_raw = _run_resilient_crew(Crew(agents=[data_type_agent], tasks=[type_task]), "Data Type Classification")
        
        json_start = type_result_raw.find('{'); json_end = type_result_raw.rfind('}') + 1
        type_json = json.loads(type_result_raw[json_start:json_end])
        numerical_cols = type_json.get('numerical', [])
        categorical_cols = type_json.get('categorical', [])
        print(f"AI classified as Numerical: {numerical_cols}")
        print(f"AI classified as Categorical: {categorical_cols}")

        # --- Stage 4: Deterministic Aggregation with Pandas ---
        print("Stage 4: Aggregating data...")
        final_aggregation = defaultdict(lambda: defaultdict(float))
        numerical_cols = [col for col in numerical_cols if col in df.columns]
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # --- THE FIX: Data Sanitization Step ---
        # Forcefully convert all categorical columns to strings before any analysis.
        for col in categorical_cols:
            df[col] = df[col].astype(str)

        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=numerical_cols, how='all', inplace=True)

        for col, total in df[numerical_cols].sum().to_dict().items():
            if not math.isnan(total): final_aggregation[col]['sum'] = total

        for col in categorical_cols:
            for category, count in df[col].value_counts().nlargest(10).to_dict().items():
                final_aggregation[col][str(category)] = count
        
        if not final_aggregation: raise ValueError("Aggregation failed. No valid data processed.")
        print("Aggregation complete.")

        # --- Stage 5: Code-based Dashboard Generation & AI-powered Reporting ---
        print("Stage 5a: Generating dashboard data with code...")
        dashboard_data = _format_data_for_dashboard(final_aggregation, numerical_cols, categorical_cols)

        print("Stage 5b: Generating final text report with AI...")
        final_data_str = json.dumps(final_aggregation, indent=2, default=str)
        report_task = Task(
            description=f"Based on the following aggregated data, write a detailed business analysis in markdown. The report should start with an executive summary, followed by bullet points highlighting the top 5-7 key findings (e.g., top-selling products, most profitable regions). Conclude with a brief summary. Data:\n\n{final_data_str}",
            expected_output="A comprehensive business report in markdown format.", agent=report_writing_agent
        )
        report_output = _run_resilient_crew(Crew(agents=[report_writing_agent], tasks=[report_task]), "Report Generation", initial_wait=20)
        
        if not report_output or len(report_output.split()) < 10:
            report_output = "## Analysis Complete\nThe AI generated the dashboard visuals but did not produce a detailed text summary for this dataset."

        print("Final assembly complete.")
        return {
            "report": report_output,
            "dashboardData": dashboard_data
        }

    except Exception as e:
        print(f"A critical error occurred in run_crew: {e}")
        return {"report": f"## Analysis Failed\n\nA critical error occurred: {e}", "dashboardData": {}}

