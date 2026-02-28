"""
Simple Streamlit Dashboard for Hotel ML Predictions
"""

import streamlit as st
import requests
import pandas as pd

# Configuration
st.set_page_config(
    page_title="Hotel Booking Predictions", page_icon="🏨", layout="wide"
)

# API URL
API_URL = st.sidebar.text_input(
    "API URL",
    value="https://hotel-demand-forcasting.onrender.com",
    help="Enter your deployed API URL",
)

st.sidebar.title("🏨 Hotel ML Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Select Feature", ["Cancellation Prediction", "Demand Forecast", "API Health"]
)


# Health Check
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "API timeout - the free tier may be sleeping. Wait 1-2 minutes and try again.",
        }
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Cannot connect to API. Check the URL."}
    except Exception as e:
        return {"status": "error", "message": f"Error: {str(e)}"}


# Cancellation Prediction
def predict_cancellation(booking_data):
    try:
        response = requests.post(
            f"{API_URL}/predict/cancellation", json=booking_data, timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "error": "API timeout - the free tier may be sleeping. Wait 1-2 minutes and try again."
        }
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Check if the API is running."}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


# Demand Forecast
def get_demand_forecast(days):
    try:
        response = requests.get(f"{API_URL}/predict/demand?days={days}", timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {
            "error": "API timeout - the free tier may be sleeping. Wait 1-2 minutes and try again."
        }
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API. Check if the API is running."}
    except Exception as e:
        return {"error": f"Error: {str(e)}"}


if page == "Cancellation Prediction":
    st.title("🔮 Booking Cancellation Prediction")
    st.markdown("Enter booking details to predict cancellation probability.")

    with st.form("booking_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("##### 📅 Booking Details")
            lead_time = st.number_input("Lead Time (days)", 0, 730, 45)
            arrival_week = st.number_input("Arrival Week", 1, 53, 27)
            arrival_month = st.number_input("Arrival Month", 1, 12, 7)
            weekend_nights = st.number_input("Weekend Nights", 0, 14, 2)
            week_nights = st.number_input("Week Nights", 0, 30, 5)
            adults = st.number_input("Adults", 1, 10, 2)
            children = st.number_input("Children", 0, 10, 0)
            babies = st.number_input("Babies", 0, 5, 0)

        with col2:
            st.markdown("##### 💰 Pricing & Hotel")
            adr = st.number_input("ADR ($)", 0.0, 1000.0, 120.0)
            special_requests = st.number_input("Special Requests", 0, 5, 1)
            parking = st.number_input("Parking Spaces", 0, 5, 0)
            hotel = st.selectbox(
                "Hotel Type",
                [0, 1],
                format_func=lambda x: "Resort" if x == 0 else "City",
            )
            deposit_type = st.selectbox(
                "Deposit",
                [0, 1, 2],
                format_func=lambda x: ["No Deposit", "Non Refund", "Refundable"][x],
            )
            is_repeated = st.checkbox("Repeated Guest")

        with col3:
            st.markdown("##### 📊 History & Channel")
            prev_cancellations = st.number_input("Previous Cancellations", 0, 20, 0)
            prev_bookings = st.number_input("Previous Bookings", 0, 50, 0)
            booking_changes = st.number_input("Booking Changes", 0, 10, 0)
            waiting_days = st.number_input("Days Waiting", 0, 365, 0)
            market_segment = st.selectbox(
                "Market Segment",
                list(range(7)),
                format_func=lambda x: [
                    "Direct",
                    "Corporate",
                    "Online TA",
                    "Offline TA",
                    "Groups",
                    "Complementary",
                    "Aviation",
                ][x],
            )
            distribution_channel = st.selectbox(
                "Distribution",
                list(range(5)),
                format_func=lambda x: ["Direct", "Corporate", "TA/TO", "GDS", "Other"][
                    x
                ],
            )
            customer_type = st.selectbox(
                "Customer Type",
                list(range(4)),
                format_func=lambda x: [
                    "Transient",
                    "Contract",
                    "Transient-Party",
                    "Group",
                ][x],
            )
            meal = st.selectbox(
                "Meal Plan",
                list(range(5)),
                format_func=lambda x: ["BB", "HB", "FB", "SC", "None"][x],
            )

        submitted = st.form_submit_button(
            "🔮 Predict Cancellation Risk", type="primary", use_container_width=True
        )

    if submitted:
        season_map = {
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
            9: 3,
            10: 3,
            11: 3,
            12: 0,
        }

        booking_data = {
            "lead_time": lead_time,
            "arrival_date_week_number": arrival_week,
            "arrival_month_num": arrival_month,
            "stays_in_weekend_nights": weekend_nights,
            "stays_in_week_nights": week_nights,
            "adults": adults,
            "children": children,
            "babies": babies,
            "is_repeated_guest": 1 if is_repeated else 0,
            "previous_cancellations": prev_cancellations,
            "previous_bookings_not_canceled": prev_bookings,
            "booking_changes": booking_changes,
            "days_in_waiting_list": waiting_days,
            "adr": adr,
            "required_car_parking_spaces": parking,
            "total_of_special_requests": special_requests,
            "hotel": hotel,
            "meal": meal,
            "market_segment": market_segment,
            "distribution_channel": distribution_channel,
            "deposit_type": deposit_type,
            "customer_type": customer_type,
            "season": season_map[arrival_month],
        }

        with st.spinner("Analyzing booking..."):
            result = predict_cancellation(booking_data)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            prob = result["cancellation_probability"]

            st.markdown("---")
            st.subheader("📊 Prediction Results")

            # Main metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                if result["will_cancel"]:
                    st.error("⚠️ **HIGH RISK**")
                    st.markdown("Booking likely to cancel")
                else:
                    st.success("✅ **LOW RISK**")
                    st.markdown("Booking likely to be kept")
            with col2:
                st.metric("Cancellation Probability", f"{prob*100:.1f}%")
            with col3:
                conf_colors = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.metric(
                    "Model Confidence",
                    f"{conf_colors.get(result['confidence'], '')} {result['confidence'].title()}",
                )

            # Visual progress bar
            st.markdown("##### Risk Level")
            st.progress(prob)

            # Risk factors analysis
            st.markdown("---")
            st.subheader("🔍 Risk Factor Analysis")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### Risk Indicators")
                risk_factors = []
                if lead_time > 100:
                    risk_factors.append(
                        ("⚠️ Long lead time", f"{lead_time} days - higher cancel risk")
                    )
                if prev_cancellations > 0:
                    risk_factors.append(
                        (
                            "⚠️ Previous cancellations",
                            f"{prev_cancellations} past cancellations",
                        )
                    )
                if deposit_type == 0:
                    risk_factors.append(("⚠️ No deposit", "No financial commitment"))
                if market_segment == 2:
                    risk_factors.append(
                        ("⚠️ Online TA booking", "OTA bookings cancel more often")
                    )
                if special_requests == 0:
                    risk_factors.append(
                        ("⚠️ No special requests", "Less committed guest")
                    )

                if risk_factors:
                    for factor, desc in risk_factors:
                        st.markdown(f"- {factor}: {desc}")
                else:
                    st.markdown("✅ No major risk factors detected")

            with col2:
                st.markdown("##### Positive Indicators")
                positive_factors = []
                if is_repeated:
                    positive_factors.append(
                        ("✅ Repeat guest", "Loyal customers rarely cancel")
                    )
                if special_requests > 0:
                    positive_factors.append(
                        (
                            "✅ Has special requests",
                            f"{special_requests} requests - committed guest",
                        )
                    )
                if deposit_type == 1:
                    positive_factors.append(
                        ("✅ Non-refundable deposit", "Financial commitment made")
                    )
                if prev_bookings > 0:
                    positive_factors.append(
                        (
                            "✅ Previous bookings kept",
                            f"{prev_bookings} successful stays",
                        )
                    )
                if lead_time < 30:
                    positive_factors.append(
                        (
                            "✅ Short lead time",
                            f"{lead_time} days - less time to cancel",
                        )
                    )

                if positive_factors:
                    for factor, desc in positive_factors:
                        st.markdown(f"- {factor}: {desc}")
                else:
                    st.markdown("No strong positive indicators")

            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            if prob > 0.6:
                st.warning(
                    """
                **High-risk booking detected. Consider:**
                - Request a deposit or non-refundable rate
                - Send personalized confirmation with value highlights
                - Follow up closer to arrival date
                - Prepare for potential overbooking
                """
                )
            elif prob > 0.4:
                st.info(
                    """
                **Moderate risk. Consider:**
                - Send a friendly confirmation reminder
                - Highlight cancellation policy
                - Offer incentive to keep booking (upgrade, breakfast)
                """
                )
            else:
                st.success(
                    """
                **Low-risk booking.** Standard procedures apply.
                - This guest is likely to complete their stay
                - No special intervention needed
                """
                )


elif page == "Demand Forecast":
    st.title("📈 Booking Demand Forecast")
    st.markdown("Forecast future booking demand with confidence intervals.")

    days = st.slider("Forecast Horizon (days)", 7, 90, 30)

    if st.button("📈 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecast..."):
            result = get_demand_forecast(days)

        if "error" in result:
            st.error(f"Error: {result['error']}")
        elif "forecasts" in result:
            df = pd.DataFrame(result["forecasts"])
            df["date"] = pd.to_datetime(df["date"])
            df["day_name"] = df["date"].dt.day_name()
            df["week"] = df["date"].dt.isocalendar().week

            st.success(f"✅ {days}-day forecast generated")

            # Key Metrics
            st.markdown("---")
            st.subheader("📊 Key Metrics")
            col1, col2, col3, col4 = st.columns(4)

            avg_daily = df["predicted_bookings"].mean()
            total = df["predicted_bookings"].sum()
            peak = df["predicted_bookings"].max()
            low = df["predicted_bookings"].min()

            with col1:
                st.metric(
                    "Average Daily",
                    f"{avg_daily:.1f}",
                    help="Average predicted bookings per day",
                )
            with col2:
                st.metric(
                    "Total Bookings",
                    f"{total:.0f}",
                    help="Total predicted bookings in forecast period",
                )
            with col3:
                st.metric(
                    "Peak Day", f"{peak:.1f}", help="Highest predicted daily bookings"
                )
            with col4:
                st.metric(
                    "Lowest Day", f"{low:.1f}", help="Lowest predicted daily bookings"
                )

            # Main forecast chart
            st.markdown("---")
            st.subheader("📈 Demand Forecast with Confidence Interval")

            chart_df = df.set_index("date")[
                ["predicted_bookings", "lower_bound", "upper_bound"]
            ]
            chart_df.columns = ["Predicted", "Lower Bound (95%)", "Upper Bound (95%)"]
            st.line_chart(chart_df)

            # Weekly pattern
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📅 Weekly Pattern")
                day_order = [
                    "Monday",
                    "Tuesday",
                    "Wednesday",
                    "Thursday",
                    "Friday",
                    "Saturday",
                    "Sunday",
                ]
                weekly_avg = (
                    df.groupby("day_name")["predicted_bookings"]
                    .mean()
                    .reindex(day_order)
                )
                st.bar_chart(weekly_avg)

                # Find best/worst days
                best_day = weekly_avg.idxmax()
                worst_day = weekly_avg.idxmin()
                st.markdown(
                    f"**Best day:** {best_day} ({weekly_avg[best_day]:.1f} avg)"
                )
                st.markdown(
                    f"**Slowest day:** {worst_day} ({weekly_avg[worst_day]:.1f} avg)"
                )

            with col2:
                st.subheader("📊 Booking Distribution")
                # Simple stats instead of histogram
                std_dev = df["predicted_bookings"].std()
                cv = (std_dev / avg_daily) * 100

                # Create range summary
                range_data = {
                    "Metric": [
                        "Minimum",
                        "25th Percentile",
                        "Median",
                        "75th Percentile",
                        "Maximum",
                    ],
                    "Bookings": [
                        df["predicted_bookings"].min(),
                        df["predicted_bookings"].quantile(0.25),
                        df["predicted_bookings"].median(),
                        df["predicted_bookings"].quantile(0.75),
                        df["predicted_bookings"].max(),
                    ],
                }
                range_df = pd.DataFrame(range_data)
                range_df["Bookings"] = range_df["Bookings"].round(1)
                st.dataframe(range_df, hide_index=True, use_container_width=True)

                st.markdown(f"**Variability:** {cv:.1f}% coefficient of variation")
                if cv > 20:
                    st.warning("High variability - consider dynamic pricing")
                else:
                    st.success("Stable demand pattern")

            # Insights
            st.markdown("---")
            st.subheader("💡 Business Insights")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("##### 📆 Peak Periods")
                peak_days = df.nlargest(5, "predicted_bookings")[
                    ["date", "predicted_bookings"]
                ]
                peak_days["date"] = peak_days["date"].dt.strftime("%a, %b %d")
                peak_days.columns = ["Date", "Expected Bookings"]
                st.dataframe(peak_days, hide_index=True, use_container_width=True)

            with col2:
                st.markdown("##### 📉 Low Periods")
                low_days = df.nsmallest(5, "predicted_bookings")[
                    ["date", "predicted_bookings"]
                ]
                low_days["date"] = low_days["date"].dt.strftime("%a, %b %d")
                low_days.columns = ["Date", "Expected Bookings"]
                st.dataframe(low_days, hide_index=True, use_container_width=True)

            # Recommendations
            st.markdown("---")
            st.subheader("🎯 Recommendations")

            if peak > avg_daily * 1.3:
                st.info(
                    f"""
                **Peak demand detected ({peak:.0f} bookings on best day)**
                - Consider premium pricing during peak periods
                - Ensure adequate staffing on {best_day}s
                - Limit discounts during high-demand days
                """
                )

            if low < avg_daily * 0.7:
                st.warning(
                    f"""
                **Low demand periods detected ({low:.0f} bookings on slowest day)**
                - Run promotions on {worst_day}s
                - Consider package deals for slow periods
                - Target business travelers for weekday gaps
                """
                )

            # Weekly summary
            if days >= 14:
                st.markdown("---")
                st.subheader("📅 Weekly Summary")
                weekly = (
                    df.groupby("week")
                    .agg({"predicted_bookings": ["sum", "mean"], "date": "first"})
                    .round(1)
                )
                weekly.columns = ["Total Bookings", "Daily Average", "Week Starting"]
                weekly["Week Starting"] = weekly["Week Starting"].dt.strftime("%b %d")
                weekly = weekly[["Week Starting", "Total Bookings", "Daily Average"]]
                st.dataframe(weekly, use_container_width=True)

            # Download
            with st.expander("📥 Download Forecast Data"):
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    csv,
                    "demand_forecast.csv",
                    "text/csv",
                    use_container_width=True,
                )


elif page == "API Health":
    st.title("❤️ API Health Check")

    if st.button("Check Status", type="primary"):
        with st.spinner("Checking..."):
            health = check_api_health()

        if health.get("status") == "healthy":
            st.success("✅ API is healthy!")
            st.json(health)
        else:
            st.error(f"❌ Error: {health.get('message', 'Unknown')}")

    st.markdown(f"**API Docs**: [{API_URL}/docs]({API_URL}/docs)")
