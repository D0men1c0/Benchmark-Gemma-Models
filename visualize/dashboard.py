import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import io

# Try to import the data loading utility
try:
    from data_utils import load_and_process_benchmark_data
except ImportError:
    st.error("Critical Error: Could not import from 'data_utils.py'. Ensure it's in the same directory.")
    st.stop()

# Define paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RESULTS_JSON_PATH = PROJECT_ROOT / "benchmarks_output" / "benchmark_results.json"

# --------------------------- PLOT UTILITIES ---------------------------

def generate_bar_chart(
    df_plot: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "",
    barmode: str = 'group',
    y_label_override: Optional[str] = None,
    x_label_override: Optional[str] = None,
    color_label_override: Optional[str] = None,
    height: Optional[int] = None
) -> Optional[px.bar]:
    """
    Generate a bar chart from a filtered DataFrame.

    :param df_plot: DataFrame containing plotting data
    :param x_col: X-axis column
    :param y_col: Y-axis column
    :param color_col: Optional color grouping
    :param title: Plot title
    :param barmode: Plotly barmode
    :param y_label_override: Optional label for Y-axis
    :param x_label_override: Optional label for X-axis
    :param color_label_override: Optional label for color
    :param height: Optional figure height
    :return: Plotly bar chart figure
    """
    if df_plot.empty or y_col not in df_plot.columns or x_col not in df_plot.columns:
        return None

    df_plot[y_col] = pd.to_numeric(df_plot[y_col], errors='coerce')
    df_plot_valid = df_plot.dropna(subset=[y_col])
    if df_plot_valid.empty:
        return None

    if height is None:
        num_x_categories = len(df_plot_valid[x_col].unique())
        base_height = 400
        height = base_height + num_x_categories * 30
        height = min(height, 600)
        if color_col and color_col in df_plot_valid.columns:
            height += len(df_plot_valid[color_col].unique()) * 10

    labels = {
        y_col: y_label_override or y_col.replace('_', ' ').title(),
        x_col: x_label_override or x_col.replace('_', ' ').title()
    }
    if color_col and color_col in df_plot_valid.columns:
        labels[color_col] = color_label_override or color_col.replace('_', ' ').title()

    try:
        fig = px.bar(
            df_plot_valid,
            x=x_col,
            y=y_col,
            color=color_col if color_col in df_plot_valid.columns else None,
            barmode=barmode,
            title=title,
            labels=labels,
            text_auto='.3f',
            height=height
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(
            title_x=0.5,
            legend_title_text=labels.get(color_col, color_col.replace('_', ' ').title() if color_col else None),
            xaxis={'categoryorder': 'total descending' if y_col == 'score' else 'trace'}
        )
        return fig
    except Exception:
        return None

def generate_heatmap(
    df_pivot: pd.DataFrame,
    title: str = "",
    value_label_override: Optional[str] = "Score",
    x_label_override: Optional[str] = "Task",
    y_label_override: Optional[str] = "Model"
) -> Optional[px.imshow]:
    """
    Generate a heatmap based on a pivoted DataFrame.

    :param df_pivot: Pivoted DataFrame
    :param title: Plot title
    :param value_label_override: Optional label for cell values
    :param x_label_override: Optional X-axis label
    :param y_label_override: Optional Y-axis label
    :return: Plotly heatmap figure
    """
    if df_pivot.empty:
        return None

    try:
        height = max(400, len(df_pivot.index) * 40 + 100)
        width = max(600, len(df_pivot.columns) * 90 + 150)
        fig = px.imshow(
            df_pivot,
            text_auto='.3f',
            aspect="auto",
            color_continuous_scale=px.colors.sequential.YlGnBu,
            title=title,
            labels=dict(
                x=x_label_override or str(df_pivot.columns.name),
                y=y_label_override or str(df_pivot.index.name),
                color=value_label_override or "Score"
            ),
            height=height,
            width=width
        )
        fig.update_xaxes(side="bottom", tickangle=45)
        fig.update_layout(title_x=0.5)
        return fig
    except Exception:
        return None

# --------------------------- STREAMLIT DASHBOARD ---------------------------

st.set_page_config(layout="wide", page_title="LLM Benchmark Dashboard")
st.title("📊 LLM Benchmark Results Dashboard")

# Upload or load default data
uploaded_file = st.sidebar.file_uploader(
    "Upload JSON (Optional)", type="json", help=f"Default: {DEFAULT_RESULTS_JSON_PATH.name}"
)
data_file_path_to_load: Union[Path, io.BytesIO, Any] = uploaded_file if uploaded_file else DEFAULT_RESULTS_JSON_PATH

# Load benchmark data
main_df = None
if isinstance(data_file_path_to_load, Path) and data_file_path_to_load.exists():
    main_df = load_and_process_benchmark_data(data_file_path_to_load)
elif not isinstance(data_file_path_to_load, Path) and data_file_path_to_load is not None:
    main_df = load_and_process_benchmark_data(data_file_path_to_load)
else:
    if not uploaded_file:
        st.sidebar.warning(f"Default data file not found: {DEFAULT_RESULTS_JSON_PATH.name}")

if main_df is None or main_df.empty:
    st.warning("No benchmark data loaded. Dashboard cannot be displayed.")
    st.stop()

# --------------------------- SIDEBAR FILTERS ---------------------------

st.sidebar.header("Global Filters")

available_models = sorted(main_df['model'].unique()) if 'model' in main_df.columns else []
available_tasks = sorted(main_df['task'].unique()) if 'task' in main_df.columns else []
available_metrics = sorted(main_df['full_metric_name'].unique()) if 'full_metric_name' in main_df.columns else []
available_metrics = [m for m in available_metrics if m not in ["processing_error", "error", None, "None_None"] and pd.notna(m)]

selected_models = st.sidebar.multiselect("Models:", options=available_models, default=available_models)
selected_tasks = st.sidebar.multiselect("Tasks:", options=available_tasks, default=available_tasks)
selected_metrics_global = st.sidebar.multiselect("Metrics (for relevant views):", options=available_metrics, default=available_metrics)

# Apply filters to DataFrame
df_to_display = main_df.copy()
if selected_models:
    df_to_display = df_to_display[df_to_display['model'].isin(selected_models)]
if selected_tasks:
    df_to_display = df_to_display[df_to_display['task'].isin(selected_tasks)]

# Update current options after filter
current_models = sorted(df_to_display['model'].unique()) if 'model' in df_to_display.columns else []
current_tasks = sorted(df_to_display['task'].unique()) if 'task' in df_to_display.columns else []
current_metrics = sorted(df_to_display['full_metric_name'].unique()) if 'full_metric_name' in df_to_display.columns else []
current_metrics = [m for m in current_metrics if m not in ["processing_error", "error", None, "None_None"] and pd.notna(m)]

# --------------------------- TABS ---------------------------

tab_titles = ["📊 Model Comparison", "📚 Task Performance", "⚖️ Metric Breakdown", "⚔️ Model vs Model Comparison", "🌡️ Overall Heatmap"]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_titles)

# Tab 1: Model Comparison
with tab1:
    st.subheader("Model Comparison")
    if not current_tasks or not current_metrics:
        st.info("Select task/metric globally.")
    else:
        col1, col2 = st.columns(2)
        task_for_mc = col1.selectbox("Task:", current_tasks, index=0, key="mc_task")
        metric_for_mc = col2.selectbox("Metric:", current_metrics, index=0, key="mc_metric")
        if task_for_mc and metric_for_mc:
            plot_df = df_to_display[(df_to_display['task'] == task_for_mc) & (df_to_display['full_metric_name'] == metric_for_mc)]
            if not plot_df.empty:
                fig = generate_bar_chart(plot_df, 'model', 'score', title=f"Models: '{task_for_mc}' - '{metric_for_mc}'")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"No data for {task_for_mc} / {metric_for_mc}")

# Tab 2: Task Performance
with tab2:
    st.subheader("Task Performance")
    if not current_models or not current_metrics:
        st.info("Select model/metric globally.")
    else:
        col1, col2 = st.columns(2)
        model_for_tp = col1.selectbox("Model:", current_models, index=0, key="tp_model")
        metric_for_tp = col2.selectbox("Metric:", current_metrics, index=0, key="tp_metric")
        if model_for_tp and metric_for_tp:
            plot_df = df_to_display[(df_to_display['model'] == model_for_tp) & (df_to_display['full_metric_name'] == metric_for_tp)]
            if not plot_df.empty:
                fig = generate_bar_chart(plot_df, 'task', 'score', title=f"Tasks: '{model_for_tp}' - '{metric_for_tp}'")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"No data for {model_for_tp} / {metric_for_tp}")

# Tab 3: Metric Breakdown
with tab3:
    st.subheader("Metric Breakdown")
    if not current_models or not current_tasks:
        st.info("Select model/task globally.")
    else:
        col1, col2 = st.columns(2)
        model_for_mb = col1.selectbox("Model:", current_models, index=0, key="mb_model")
        task_for_mb = col2.selectbox("Task:", current_tasks, index=0, key="mb_task")
        if model_for_mb and task_for_mb:
            plot_df = df_to_display[(df_to_display['model'] == model_for_mb) & (df_to_display['task'] == task_for_mb)]
            plot_df = plot_df[plot_df['full_metric_name'].isin(selected_metrics_global)]
            plot_df = plot_df[~plot_df['full_metric_name'].isin(["processing_error", "error", None, "None_None"]) & pd.notna(plot_df['full_metric_name'])]
            if not plot_df.empty:
                fig = generate_bar_chart(plot_df, 'full_metric_name', 'score', title=f"Metrics: '{model_for_mb}' - '{task_for_mb}'", x_label_override="Metric")
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.caption(f"No metric data for {model_for_mb} / {task_for_mb}")


# Tab 4: Model vs. Model Comparison
with tab4:
    st.subheader("⚔️ Model vs Model Comparison")

    if not current_models or len(current_models) < 1: # Need at least one model to select from, ideally 2
        st.info("Not enough models available after global filters for comparison. Please select at least two models globally.")
    elif not current_tasks:
        st.info("No tasks available after global filters for comparison.")
    else:
        col1_select, col2_select = st.columns(2)
        with col1_select:
            model1_choice = st.selectbox(
                "Select Model 1:",
                current_models,
                index=0 if current_models else -1,
                key="mvm_model1_select"
            )
        with col2_select:
            # Ensure Model 2 options don't include Model 1 for a true MvM, or allow same model for sanity check
            model2_options = [m for m in current_models if m != model1_choice] if len(current_models) > 1 and model1_choice else current_models
            if not model2_options and current_models: # If only one model selected globally, or model1 is the only one
                 model2_options = current_models 
            
            model2_choice = st.selectbox(
                "Select Model 2:",
                model2_options,
                index=0 if len(model2_options) > 0 else -1, # Default to first option if available
                key="mvm_model2_select"
            )

        # Select a common task for comparison
        # For simplicity, use all 'current_tasks'. The logic will handle if data exists.
        common_task_choice = st.selectbox(
            "Select Common Task for Comparison:",
            current_tasks,
            index=0 if current_tasks else -1,
            key="mvm_task_select"
        )

        if not model1_choice or not model2_choice or not common_task_choice:
            st.info("Please select two models and a common task.")
        else:
            # Get metrics relevant to the chosen common task for these two models
            metrics_for_task_model1 = sorted(df_to_display[
                (df_to_display['model'] == model1_choice) &
                (df_to_display['task'] == common_task_choice)
            ]['full_metric_name'].unique())
            
            metrics_for_task_model2 = sorted(df_to_display[
                (df_to_display['model'] == model2_choice) &
                (df_to_display['task'] == common_task_choice)
            ]['full_metric_name'].unique())

            # Find common metrics available for both models on that task
            common_metrics_for_selection = sorted(list(set(metrics_for_task_model1) & set(metrics_for_task_model2)))
            common_metrics_for_selection = [m for m in common_metrics_for_selection if m not in ["processing_error", "error", None, "None_None"] and pd.notna(m)]


            metric_options = ["Compare ALL Metrics for this Task"] + common_metrics_for_selection
            
            metric_to_compare = st.selectbox(
                "Select Metric for Comparison (or 'ALL'):",
                metric_options,
                index=0 if metric_options else -1, # Default to "ALL" if available
                key="mvm_metric_select"
            )

            if metric_to_compare:
                st.markdown("---")
                
                if metric_to_compare == "Compare ALL Metrics for this Task":
                    st.markdown(f"##### Comparing '{model1_choice}' vs '{model2_choice}' on ALL Metrics for Task: '{common_task_choice}'")
                    
                    data_m1 = df_to_display[
                        (df_to_display['model'] == model1_choice) &
                        (df_to_display['task'] == common_task_choice) &
                        (df_to_display['full_metric_name'].isin(common_metrics_for_selection)) # Use common available metrics
                    ]
                    data_m2 = df_to_display[
                        (df_to_display['model'] == model2_choice) &
                        (df_to_display['task'] == common_task_choice) &
                        (df_to_display['full_metric_name'].isin(common_metrics_for_selection))
                    ]
                    
                    if data_m1.empty and data_m2.empty:
                        st.warning(f"No data found for either model on task '{common_task_choice}' for the available metrics.")
                    else:
                        comparison_df_all_metrics = pd.concat([data_m1, data_m2])
                        if not comparison_df_all_metrics.empty:
                            fig = generate_bar_chart(
                                df_plot=comparison_df_all_metrics,
                                x_col='full_metric_name',
                                y_col='score',
                                color_col='model',
                                title=f"All Metrics: '{model1_choice}' vs '{model2_choice}' on '{common_task_choice}'",
                                barmode='group',
                                x_label_override="Metric",
                                height = 600 + len(common_metrics_for_selection) * 20 # Adjust height based on number of metrics
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info("Could not generate comparison chart for all metrics.")
                        else:
                            st.info("No common data to compare for all metrics.")

                else: # Specific metric selected
                    st.markdown(f"##### Comparing '{model1_choice}' vs '{model2_choice}' on Metric: '{metric_to_compare}' for Task: '{common_task_choice}'")

                    score_m1_data = df_to_display[
                        (df_to_display['model'] == model1_choice) &
                        (df_to_display['task'] == common_task_choice) &
                        (df_to_display['full_metric_name'] == metric_to_compare)
                    ]
                    score_m2_data = df_to_display[
                        (df_to_display['model'] == model2_choice) &
                        (df_to_display['task'] == common_task_choice) &
                        (df_to_display['full_metric_name'] == metric_to_compare)
                    ]

                    score_m1 = score_m1_data['score'].iloc[0] if not score_m1_data.empty else None
                    score_m2 = score_m2_data['score'].iloc[0] if not score_m2_data.empty else None

                    # Display individual scores using st.metric
                    cols_scores = st.columns(2)
                    with cols_scores[0]:
                        if score_m1 is not None:
                            st.metric(label=f"{model1_choice}: {metric_to_compare}", value=f"{score_m1:.4f}")
                        else:
                            st.caption(f"No score for {model1_choice} on this metric/task.")
                    with cols_scores[1]:
                        if score_m2 is not None:
                            st.metric(label=f"{model2_choice}: {metric_to_compare}", value=f"{score_m2:.4f}")
                        else:
                            st.caption(f"No score for {model2_choice} on this metric/task.")
                    
                    if score_m1 is not None and score_m2 is not None:
                        df_compare_single_metric = pd.DataFrame({
                            'Model': [model1_choice, model2_choice],
                            'Score': [score_m1, score_m2]
                        })
                        fig = generate_bar_chart(
                            df_plot=df_compare_single_metric,
                            x_col='Model',
                            y_col='Score',
                            color_col='Model', # Colors bars by model name
                            title=f"Comparison on '{metric_to_compare}' for task '{common_task_choice}'",
                            barmode='group' # or 'overlay' if preferred and only 2 bars
                        )
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Could not generate comparison chart for the selected metric.")
                    elif score_m1 is None and score_m2 is None:
                        st.warning("No data available for either model on the selected task/metric.")
                    else:
                        st.info("Data available for only one model. Direct comparison chart not shown.")
            else:
                st.info("Please select a metric or 'ALL' to proceed.")

# Tab 5: Heatmap
with tab5:
    st.subheader("Overall Performance Heatmap")
    if not current_metrics:
        st.info("Select metric globally for heatmap.")
    else:
        metric_for_hm = st.selectbox("Metric for Heatmap:", current_metrics, index=0, key="hm_metric")
        if metric_for_hm:
            heatmap_data = df_to_display[df_to_display['full_metric_name'] == metric_for_hm]
            if not heatmap_data.empty and 'model' in heatmap_data and 'task' in heatmap_data:
                try:
                    pivot_df = heatmap_data.pivot_table(index='model', columns='task', values='score', aggfunc='mean')
                    if not pivot_df.empty:
                        fig = generate_heatmap(pivot_df, title=f"Heatmap: {metric_for_hm}")
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.caption(f"Cannot pivot data for {metric_for_hm}")
                except Exception:
                    st.caption(f"Error pivoting for {metric_for_hm}")
            else:
                st.caption(f"No data for {metric_for_hm} heatmap.")

# --------------------------- RAW DATA DISPLAY ---------------------------

if st.sidebar.checkbox("Show Data Table", False, key="show_table"):
    st.subheader("Filtered Data")
    if not df_to_display.empty:
        st.dataframe(df_to_display.sort_values(by=['model', 'task', 'full_metric_name']).reset_index(drop=True))
    else:
        st.caption("No data after filters.")

# --------------------------- SIDEBAR FOOTER ---------------------------

st.sidebar.markdown("---")
data_source_name = "Uploaded" if uploaded_file else f"Local ({DEFAULT_RESULTS_JSON_PATH.name})"
st.sidebar.markdown(f"Source: `{data_source_name}`")
if main_df is not None:
    st.sidebar.caption(f"Total records: {len(main_df)}")
    st.sidebar.caption(f"Displaying: {len(df_to_display)}")