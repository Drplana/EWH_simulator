import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from water_consumption.stochastic_profile import StochasticDHWProfile


def generate_belgian_profiles(n_profiles=1000, nday=1, display_progress=True):
    """
    Generate domestic hot water consumption profiles using Belgian demographics.

    Parameters:
    -----------
    n_profiles : int
        Number of profiles to generate
    nday : int
        Number of days to simulate
    display_progress : bool
        Whether to display progress in Streamlit

    Returns:
    --------
    profiles : list
        List of generated profiles as xarray Datasets
    metadata : pd.DataFrame
        Metadata about the profiles
    """
    # Belgian household size distribution (based on Statbel data)
    # Source: https://statbel.fgov.be/en/themes/households/household-size
    household_sizes = [1, 2, 3, 4, 5]  # 1 to 5+ persons
    probabilities = [0.35, 0.32, 0.15, 0.12, 0.06]  # Example distribution

    # Sample household sizes according to distribution
    sampled_sizes = np.random.choice(household_sizes, size=n_profiles, p=probabilities)

    # Initialize progress reporting
    progress_bar = None
    if display_progress:
        try:
            # Try to create a Streamlit progress bar
            progress_bar = st.progress(0.0)
        except:
            # Fall back to print statements if not in Streamlit
            pass

    # Generate profiles
    profiles = []
    metadata_records = []
    for i, n_users in enumerate(sampled_sizes):
        # Generate profile with the given number of users
        profile_generator = StochasticDHWProfile(nday=nday, n_users=n_users)
        df = profile_generator.DHW_load_gen()

        # Add a profile_id coordinate
        ds = xr.Dataset.from_dataframe(df)
        profiles.append(ds)

        stats = profile_generator.validate_statistics()
        metadata_records.append({
            'household_size': n_users,
            'daily_liters': stats['daily_usage_liters'],
            'shower_count': stats['shower_count'],
            'bath_count': stats['bath_count'],
            'profile_id': i
        })

        # Update progress
        if progress_bar is not None:
            try:
                progress_bar.progress((i + 1) / n_profiles)
            except:
                pass
        elif display_progress and i % 50 == 0:
            print(f"Generated {i + 1}/{n_profiles} profiles ({(i + 1) / n_profiles * 100:.1f}%)")

        # Create metadata dataframe
    metadata = pd.DataFrame(metadata_records)

    return profiles, metadata


def visualize_belgian_profiles(profiles, metadata):
    """
    Create visualizations for the generated profiles
    """
    # Create combined dataset
    combined = xr.concat(profiles, dim="profile")

    # Plot histogram of household sizes
    fig_hist = px.histogram(
        metadata, x="household_size",
        labels={"household_size": "Number of occupants"},
        title="Distribution of Belgian Household Sizes"
    )
    fig_hist.update_layout(xaxis=dict(tickmode='linear', dtick=1))

    # Plot aggregate water consumption patterns
    daily_avg = combined.mean(dim="profile")

    fig_consumption = go.Figure()

    # Add stacked area for appliances
    for appliance in ["sinkA", "sinkB", "shower", "bath"]:
        fig_consumption.add_trace(go.Scatter(
            x=daily_avg.time,
            y=daily_avg[appliance],
            stackgroup='one',
            name=appliance.capitalize()
        ))

    # Add total line
    fig_consumption.add_trace(go.Scatter(
        x=daily_avg.time,
        y=daily_avg.total,
        mode='lines',
        name='Total',
        line=dict(color='darkblue', width=2)
    ))

    fig_consumption.update_layout(
        title='Average DHW Profile by Appliance (Belgian Demographics)',
        xaxis_title='Time [h]',
        yaxis_title='Flow [L/s]'
    )

    return fig_hist, fig_consumption


# Example usage:
if __name__ == "__main__":
    # Generate profiles
    profiles, metadata = generate_belgian_profiles(n_profiles=1000)

    # Create visualizations
    fig_hist, fig_consumption = visualize_belgian_profiles(profiles, metadata)

    # Display in Streamlit
    st.plotly_chart(fig_hist)
    st.plotly_chart(fig_consumption)

    # Calculate statistics
    st.write(f"Total number of profiles: {len(profiles)}")
    st.write(f"Mean household size: {metadata['household_size'].mean():.2f}")

    # Save to file if needed
    # combined = xr.concat(profiles, dim="profile")
    # combined.to_netcdf('belgian_profiles.nc')