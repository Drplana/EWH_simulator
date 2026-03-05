
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# plot_manager.py
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

class PlotManager:
    def __init__(self):
        st.set_page_config(layout="wide")

    def plot_temperature(self, time_index, temperatures):
        st.subheader("Water Heater Temperature Over Time")
        fig = go.Figure()
        # If temperatures is 2D: shape (n_heaters, n_timesteps)
        if hasattr(temperatures, "shape") and len(temperatures.shape) == 2:
            for i in range(temperatures.shape[0]):
                fig.add_trace(go.Scatter(
                    x=time_index, y=temperatures[i], mode="lines", name=f"Heater {i + 1}"
                ))
        else:
            # Single heater
            fig.add_trace(go.Scatter(
                x=time_index, y=temperatures, mode="lines", name="Temperature"
            ))
        fig.update_layout(xaxis_title="Time", yaxis_title="Temperature (°C)")
        st.plotly_chart(fig, use_container_width=True)

    def plot_status(self, time_index, statuses):
        st.subheader("Heating Element Status")
        fig = go.Figure()
        n_heaters = statuses.shape[0]
        for i in range(n_heaters):
            fig.add_trace(go.Scatter(
                x=time_index,
                y=statuses[i],
                mode="lines",
                name=f"Heater {i + 1}"
            ))
        fig.update_layout(xaxis_title="Time", yaxis_title="Status (On=1, Off=0)")
        st.plotly_chart(fig, use_container_width=True)

    def plot_power_profiles(self, time_index, power_profiles, agg_power, fig1 = True, fig2 = True, ewh_power_consumption=None):
        st.subheader("Power Consumption Profiles")
        chart_height = 350  # Adjust as needed
        chart_width = 600
        if fig1:
            fig1 = go.Figure()
            for i, power in enumerate(power_profiles):
                fig1.add_trace(go.Scatter(x=time_index, y=power, mode="lines", name=f"EWH {i+1}"))

            fig1.update_layout(xaxis_title="Time [min]", yaxis_title="Power [kW]",
                                height=chart_height, width=chart_width)
            st.plotly_chart(fig1, use_container_width=False)
            # Export as SVG
            svg_bytes = fig1.to_image(format="svg")
            st.download_button("Download Chart 1 as SVG", data=svg_bytes, file_name="chart1.svg", mime="image/svg+xml")
        if fig2:
            fig2 = go.Figure()
            if ewh_power_consumption is not None:
                fig2.add_trace(go.Scatter( x=time_index[:len(ewh_power_consumption)],
                y=ewh_power_consumption['cumulated_power_kW'],
                mode="lines",
                name="Power from N.Leqlerq Model",
                line=dict(color="green", dash="dash")
            ))
            fig2.add_trace(go.Scatter(x=time_index, y=agg_power, mode="lines", name="Aggregated Power",
                                        line=dict(color="red", width=3)))
            fig2.update_layout(xaxis_title="Time [min]", yaxis_title="Power [kW]",
                                height=chart_height, width=chart_width)
            st.plotly_chart(fig2, use_container_width=False)
            svg_bytes = fig2.to_image(format="svg")
            st.download_button("Download Chart 2 as SVG", data=svg_bytes, file_name="chart2.svg", mime="image/svg+xml")
    def plot_boxplot(self, df):
        st.subheader("Tank Volume Distribution by Number of Users")
        if "n_users" not in df or df["n_users"].isnull().all():
            st.warning("Number of users is not available in the data. Boxplot cannot be displayed.")
            return
        fig = px.box(
            df, x="n_users", y="volume_liters", points="all",
            labels={"n_users": "Number of Users", "volume_liters": "Tank Volume (L)"}
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_aggregate_power_comparison(self, time_index, agg_power_baseline, agg_power_shedded,
                                        label_baseline="Baseline (no shedding)", label_shedding="With shedding"):
        st.subheader("Aggregate Power Comparison: Baseline vs. Shedding")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_index, y=agg_power_baseline, mode="lines", name=label_baseline, line=dict(color="blue")
        ))
        fig.add_trace(go.Scatter(
            x=time_index, y=agg_power_shedded, mode="lines", name=label_shedding, line=dict(color="red", dash="dash")
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Aggregate Power (kW)",
            legend=dict(x=0, y=1),
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)