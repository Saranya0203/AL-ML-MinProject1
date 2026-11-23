import streamlit as st
import pandas as pd
import plotly.express as px

from sqlalchemy import create_engine
from sqlalchemy.engine import URL


st.set_page_config(
    page_title="NYC 311 Dashboard",
    page_icon="📊",
    layout="wide",
)
st.title("NYC 311 Dashboard")
st.caption("Interactive Complaint Analysis Dashboard")


@st.cache_resource
def get_engine():
    db = st.secrets.get("postgres", {})
    required = ("username", "password", "host", "port", "database")
    missing = [key for key in required if not db.get(key)]
    if missing:
        raise KeyError("Missing postgres secrets: " + ", ".join(missing))

    url = URL.create(
        "postgresql+psycopg2",
        username=db["username"],
        password=db["password"],
        host=db["host"],
        port=int(db["port"]),
        database=db["database"],
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=300)
def load_data():
    engine = get_engine()
    return pd.read_sql("SELECT * FROM public.service_requests", engine)


def prepare_data(data):
    data = data.copy()
    for column in ("created_date", "closed_date"):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column], errors="coerce")

    if "resolution_hours" in data.columns:
        data["resolution_hours"] = pd.to_numeric(
            data["resolution_hours"], errors="coerce"
        )
    elif "resolution_days" in data.columns:
        data["resolution_hours"] = (
            pd.to_numeric(data["resolution_days"], errors="coerce") * 24
        )
    else:
        data["resolution_hours"] = pd.NA

    data.loc[data["resolution_hours"] < 0, "resolution_hours"] = pd.NA
    if "created_date" in data.columns and "closed_date" in data.columns:
        calculated_hours = (
            data["closed_date"] - data["created_date"]
        ).dt.total_seconds() / 3600
        calculated_hours = calculated_hours.where(calculated_hours >= 0)
        data["resolution_hours"] = data["resolution_hours"].fillna(
            calculated_hours
        )

    if "is_closed" in data.columns:
        data["is_closed"] = (
            data["is_closed"]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(("true", "1", "yes", "closed"))
        )
    elif "closed_date" in data.columns:
        data["is_closed"] = data["closed_date"].notna()
    elif "status" in data.columns:
        data["is_closed"] = (
            data["status"].astype(str).str.strip().str.lower() == "closed"
        )
    else:
        data["is_closed"] = False

    return data


try:
    df = prepare_data(load_data())

    st.sidebar.header("Dashboard Filters")

    if "created_date" in df.columns and df["created_date"].notna().any():
        min_date = df["created_date"].min().date()
        max_date = df["created_date"].max().date()
        date_range = st.sidebar.date_input(
            "Created date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            df = df[
                df["created_date"].dt.date.between(start_date, end_date)
            ]

    if "borough" in df.columns:
        boroughs = sorted(df["borough"].dropna().unique().tolist())
        selected_borough = st.sidebar.multiselect(
            "Select Borough",
            boroughs,
            default=boroughs,
        )
        df = df[df["borough"].isin(selected_borough)]

    if "complaint_type" in df.columns:
        complaint_types = sorted(
            df["complaint_type"].dropna().unique().tolist()
        )
        selected_types = st.sidebar.multiselect(
            "Select Complaint Type",
            complaint_types,
        )
        if selected_types:
            df = df[df["complaint_type"].isin(selected_types)]

    closed_count = int(df["is_closed"].sum())
    valid_durations = df["resolution_hours"].dropna()
    total_requests = (
        df["unique_key"].nunique()
        if "unique_key" in df.columns
        else len(df)
    )
    closure_rate = closed_count / len(df) * 100 if len(df) else 0
    over_72_rate = (
        valid_durations.gt(72).mean() * 100
        if not valid_durations.empty
        else 0
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Requests", f"{total_requests:,}")
    col2.metric(
        "Recorded Closure",
        f"{closure_rate:.1f}%",
        help="Based on the recorded closure indicator, not citizen satisfaction.",
    )
    col3.metric("Open Requests", f"{len(df) - closed_count:,}")
    col4.metric(
        "Median Resolution",
        f"{valid_durations.median():.1f} hours"
        if not valid_durations.empty
        else "N/A",
    )
    col5.metric(
        "Over 72 Hours",
        f"{over_72_rate:.1f}%",
        help="Share of valid resolution durations over the project-defined 72-hour threshold.",
    )
    st.divider()

    page = st.sidebar.radio(
        "Analyst pages",
        ("Overview", "Demand", "Resolution", "Geography & Records"),
    )

    if df.empty:
        st.info("No requests match the selected filters.")
    elif page == "Overview":
        if "borough" in df.columns:
            borough_data = (
                df.groupby("borough")
                .size()
                .reset_index(name="Total Requests")
            )
            fig = px.bar(
                borough_data,
                x="borough",
                y="Total Requests",
                title="Requests by Borough",
            )
            st.plotly_chart(fig, width="stretch")

        if "complaint_type" in df.columns:
            type_data = (
                df["complaint_type"]
                .value_counts()
                .head(10)
                .rename_axis("Complaint Type")
                .reset_index(name="Requests")
            )
            fig = px.bar(
                type_data,
                x="Complaint Type",
                y="Requests",
                title="Top 10 Complaint Types",
            )
            st.plotly_chart(fig, width="stretch")

        if "status" in df.columns:
            status_data = (
                df["status"]
                .fillna("Unknown")
                .value_counts()
                .rename_axis("Status")
                .reset_index(name="Requests")
            )
            st.plotly_chart(
                px.pie(status_data, names="Status", values="Requests",
                       title="Recorded Request Status"),
                width="stretch",
            )

    elif page == "Demand":
        if "created_date" in df.columns:
            dated = df.dropna(subset=["created_date"]).copy()
            if not dated.empty:
                dated["Month"] = dated["created_date"].dt.to_period("M").astype(str)
                monthly = dated.groupby("Month").size().reset_index(name="Requests")
                st.plotly_chart(
                    px.line(monthly, x="Month", y="Requests",
                            title="Monthly Request Trend", markers=True),
                    width="stretch",
                )
                dated["Weekday"] = dated["created_date"].dt.day_name()
                weekday_order = [
                    "Monday", "Tuesday", "Wednesday", "Thursday",
                    "Friday", "Saturday", "Sunday",
                ]
                weekday = (
                    dated["Weekday"].value_counts()
                    .reindex(weekday_order, fill_value=0)
                    .rename_axis("Weekday")
                    .reset_index(name="Requests")
                )
                col1, col2 = st.columns(2)
                col1.plotly_chart(
                    px.bar(weekday, x="Weekday", y="Requests",
                           title="Demand by Weekday"),
                    width="stretch",
                )
                dated["Hour"] = dated["created_date"].dt.hour
                hourly = dated.groupby("Hour").size().reset_index(name="Requests")
                col2.plotly_chart(
                    px.bar(hourly, x="Hour", y="Requests",
                           title="Demand by Hour"),
                    width="stretch",
                )
            else:
                st.info("No valid creation dates are available for demand analysis.")
        if "agency" in df.columns:
            agency_data = (
                df["agency"].fillna("Unknown").value_counts().head(15)
                .rename_axis("Agency").reset_index(name="Requests")
            )
            st.plotly_chart(
                px.bar(agency_data, x="Agency", y="Requests",
                       title="Agency Workload (Top 15)"),
                width="stretch",
            )

    elif page == "Resolution":
        if valid_durations.empty:
            st.info("No valid resolution durations are available for these filters.")
        else:
            q1 = valid_durations.quantile(0.25)
            q3 = valid_durations.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = valid_durations[
                (valid_durations < lower_bound) |
                (valid_durations > upper_bound)
            ]
            stats = pd.DataFrame(
                {
                    "Measure": [
                        "Valid records", "Mean (hours)", "Median (hours)",
                        "Minimum (hours)", "Q1 (hours)", "Q3 (hours)",
                        "Maximum (hours)", "Standard deviation (hours)",
                        "Skewness", "IQR outlier records",
                    ],
                    "Value": [
                        len(valid_durations), valid_durations.mean(),
                        valid_durations.median(), valid_durations.min(),
                        q1, q3, valid_durations.max(),
                        valid_durations.std(), valid_durations.skew(),
                        len(outliers),
                    ],
                }
            )
            st.dataframe(stats, hide_index=True, width="stretch")
            st.caption(
                "IQR outliers are review flags, not automatic exclusions."
            )
            col1, col2 = st.columns(2)
            col1.plotly_chart(
                px.histogram(df, x="resolution_hours", nbins=40,
                             title="Resolution Duration Distribution (Hours)"),
                width="stretch",
            )
            col2.plotly_chart(
                px.box(df, y="resolution_hours",
                       title="Resolution Duration Outlier Review"),
                width="stretch",
            )

        for field, title in (
            ("complaint_type", "Median Resolution by Complaint Type"),
            ("agency", "Median Resolution by Agency"),
        ):
            if field in df.columns and not valid_durations.empty:
                grouped = (
                    df.dropna(subset=["resolution_hours"])
                    .groupby(field)["resolution_hours"]
                    .median()
                    .sort_values(ascending=False)
                    .head(15)
                    .rename("Median Hours")
                    .reset_index()
                )
                st.plotly_chart(
                    px.bar(grouped, x=field, y="Median Hours", title=title),
                    width="stretch",
                )
        if {"complaint_type", "resolution_hours"}.issubset(df.columns):
            high_volume = df["complaint_type"].value_counts()
            median_by_type = df.groupby("complaint_type")[
                "resolution_hours"
            ].median()
            combined = pd.concat(
                [high_volume.rename("Requests"),
                 median_by_type.rename("Median Hours")],
                axis=1,
            ).dropna()
            if not combined.empty:
                combined = combined.sort_values(
                    ["Requests", "Median Hours"], ascending=False
                ).head(15).reset_index(names="Complaint Type")
                st.subheader("High-volume Categories with Resolution Context")
                st.dataframe(combined, hide_index=True, width="stretch")

    else:
        if "borough" in df.columns:
            borough_data = (
                df.groupby("borough").size()
                .reset_index(name="Requests")
            )
            st.plotly_chart(
                px.bar(borough_data, x="borough", y="Requests",
                       title="Geographic Concentration by Borough"),
                width="stretch",
            )
        if {"latitude", "longitude"}.issubset(df.columns):
            points = df.copy()
            points["latitude"] = pd.to_numeric(points["latitude"], errors="coerce")
            points["longitude"] = pd.to_numeric(points["longitude"], errors="coerce")
            points = points.dropna(subset=["latitude", "longitude"])
            points = points[
                points["latitude"].between(40, 41.5)
                & points["longitude"].between(-74.3, -73.5)
            ]
            if not points.empty:
                if len(points) > 20000:
                    points = points.sample(20000, random_state=0)
                st.plotly_chart(
                    px.scatter_mapbox(
                        points,
                        lat="latitude",
                        lon="longitude",
                        color="borough" if "borough" in points.columns else None,
                        hover_name="complaint_type"
                        if "complaint_type" in points.columns else None,
                        zoom=9,
                        center={"lat": 40.7128, "lon": -74.0060},
                        mapbox_style="open-street-map",
                        title="Request Locations (sampled to 20,000 points)",
                    ),
                    width="stretch",
                )
            else:
                st.info("No usable NYC coordinates are available for the selected filters.")

        st.subheader("Filtered Complaint Records")
        st.dataframe(df.head(100), width="stretch")

except Exception as error:
    st.error(f"Unable to load dashboard: {error}")
