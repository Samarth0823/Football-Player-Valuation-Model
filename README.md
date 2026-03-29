# Football-Player-Valuation-Model
This project builds an end-to-end Machine Learning pipeline to predict the market value of football players for the 2025/26 season. By merging historical market data from Transfermarkt with live performance metrics from FBref, the model identifies "Market Anomalies"—players whose on-pitch output exceeds their current valuation.

Project Architecture
1. Data Integration & Engineering
   Multi-Source Merge: Integrated a 37,000+ row historical Transfermarkt dataset with 2025/26 season stats using advanced string cleaning and fuzzy name matching.
   Domain-Specific Features: Engineered custom metrics to capture football-specific value drivers:
   young_producer: Penalizes age while rewarding high Goal/Assist output (The "Prospect" Factor).
   league_rank: A weighted coefficient based on league financial power and competition difficulty.
   GA_per_MP: Normalizing production relative to availability.

2. Machine Learning Pipeline
   Baseline: Linear Regression to establish a performance floor.
   Advanced Models: Implemented Random Forest and XGBoost to capture non-linear relationships (e.g., why a 19-year-old's value spikes faster than a 29-year-old's).
   Optimization: Utilized GridSearchCV for hyperparameter tuning, achieving a ~50% $R^2$ score—a significant improvement over baseline performance.
3. Explainable AI (SHAP)
   The "Why" Factor: Implemented SHAP (SHapley Additive exPlanations) to break the "Black Box.
   "Generated Waterfall Plots to explain individual player valuations (e.g., "Why is Andrej Ilić a bargain?").
   Created Beeswarm Plots to visualize global market drivers, proving that MP and league_rank are the primary engines of value in 2026.
4. Similarity Engine (Recruitment Tool)
   Cosine Similarity: Developed a recommendation engine that identifies "cheaper alternatives" to superstars.
   Uses vectorized performance stats to find players with similar output profiles but significantly lower market costs.

Key Insights
Top Market Driver: Matches Played (MP) and League Rank correlate most strongly with value, suggesting availability is the "best ability."
Market Bargains: Identified several high-performing assets in mid-tier leagues (e.g., Eredivisie, Championship) whose "Fair Value" per the model is 2x-3x higher than their actual price.

Tech Stack
Languages: Python
Libraries: Pandas, NumPy, Scikit-Learn, XGBoost, SHAP, Matplotlib, Seaborn
Tools: Google Colab, GitHub, Streamlit (Deployment)
