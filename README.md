Commercial Energy Consumption Anomaly Detection (United States) | Python, Pandas, NumPy, Plotly

Built an end-to-end energy analytics pipeline to process multi-year hourly electricity consumption data for commercial buildings and convert raw meter readings into decision-ready operational insights. Designed the solution to compute daily/monthly performance metrics, model expected usage behavior across business vs off-hours, detect anomalous consumption patterns, quantify seasonal load variation, estimate baseline always-on demand, and project next-month usage for planning and budgeting.

Implemented robust time-series preprocessing to standardize timestamps, handle missing hourly records through forward/backward filling, and maintain data quality for consistent downstream analysis. Developed modular analytics components for aggregation, statistical anomaly detection (weekday/weekend-aware z-score logic), seasonality profiling, and baseline estimation from night/weekend operating windows. Added a forecasting layer to estimate future monthly consumption using historical month-aligned daily trends with fallback logic for sparse cases.

Delivered a self-contained interactive HTML reporting output using Plotly, including a 24xN daily heatmap (hour vs date), visual anomaly highlighting, and cost estimation driven by configurable cost_per_kwh. Ensured portability of outputs by embedding visualization assets inline, enabling stakeholders to review reports without external dependencies.

Validated implementation through automated pytest-based checks and structured output contracts, producing reproducible artifacts suitable for operational monitoring, auditability, and future model enhancement.
