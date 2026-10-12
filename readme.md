# FitInsights — Fitness Trends & Predictive Analytics

FitInsights is the **data science side** of the FitSense project. It takes raw fitness-tracker data, cleans it, engineers features, tests hypotheses statistically, and trains the model behind the [FitSense-AI](https://github.com/minierks/FitSense) web app (live at https://fitsense-pi.vercel.app).

Think of it as the lab. Everything produced here (cleaning rules, engineered features, the trained `.joblib` model) is what FitSense's API serves.

---

## 1. Repository Contents

| File | Type | Purpose |
|---|---|---|
| `FitInsights.ipynb` | Notebook | Data cleaning, feature engineering, EDA and hypothesis testing |
| **`ImprovedModel.ipynb`** | Notebook | **Current model:** baseline comparison of the first models, the calorie-rate model, the recommendation logic, and export of the files FitSense uses |
| `PredictWorkout.ipynb` | Notebook | First-version models (workout-type classifier + duration regressor), kept for reference |
| `tracker_data.csv` | Data | **Raw** dataset (input to cleaning) |
| `cleaned_fitness_dataset.csv` | Data | After cleaning (deduplicated, typos fixed, imputed, outliers capped) |
| `featured_fitness_dataset.csv` | Data | After feature engineering |
| `fitinsights_data.csv` | Data | Training data for the models (1,629 rows) |
| **`fitinsight_calorie_rate_model.joblib`** | Model | **Current model:** predicts calories burned per minute (served by FitSense) |
| **`workout_profiles.json`** | Data | Average intensity per workout, typical session length per experience level, longest session; used by the recommendation rules |
| `fitinsight_workout_model.joblib`, `fitinsight_seesion_model.joblib` | Model | First-version models (no longer served) |
| `G-03.FitInsights.docx` | Report | Written project report |

---

## 2. Problem Statement

Wearables and gym trackers record a lot of raw data (heart rate, calories, duration, hydration), but most people can't turn it into a plan. FitInsights turns that data into **clear trends, statistically tested insights, and personalized workout recommendations**.

## 3. Objectives

1. Analyze trends in heart rate, calories burned, workout frequency and other core metrics.
2. Engineer physiologically meaningful features (intensity, efficiency, cardio reserve, hydration ratio).
3. Test relationships between variables with formal hypothesis tests, not just charts.
4. Train, evaluate and export a model that FitSense uses to recommend workouts for a calorie target.

---

## 4. Dataset

- **Source:** [Kaggle — Fitness Tracker Dataset](https://www.kaggle.com/datasets/nadeemajeedch/fitness-tracker-dataset), mirrored as `tracker_data.csv`.
- **Size:** 1,680 raw rows × 15 columns → **1,674 rows** after cleaning. The training file `fitinsights_data.csv` has **1,629 rows**.
- **Workout types (14):** Aerobics, Boxing, CrossFit, Cycling, HIIT, Pilates, Rowing, Running, Soccer, Strength Training, Swimming, Tennis, Walking, Yoga.

| Category | Columns |
|---|---|
| Demographics | `Age`, `Gender` |
| Body | `Weight_kg`, `Height_m`, `BMI`, `Fat_Percentage` |
| Workout | `Workout_Type`, `Session_Duration_hours`, `Workout_Frequency_days_per_week`, `Experience_Level` |
| Heart rate & energy | `Max_BPM`, `Avg_BPM`, `Resting_BPM`, `Calories_Burned` |
| Hydration | `Water_Intake_liters` |

---

## 5. ETL Pipeline (`FitInsights.ipynb`)

```
tracker_data.csv (raw, 1,680 rows)
        │  EXTRACT: pd.read_csv
        ▼
   TRANSFORM (cleaning)
   ├─ Fix category typos ("Runnng" → "Running", "CrosFit" → "CrossFit", "Advnaced" → "Advanced", …)
   ├─ Drop 6 duplicate rows
   ├─ Fill missing values: numeric → median, categorical → most common value
   ├─ Detect outliers: Z-score (|z| > 3, 10 rows) and IQR
   └─ Winsorize: cap numeric columns to the IQR bounds instead of deleting rows
        │  LOAD
        ▼
cleaned_fitness_dataset.csv (1,674 rows)
        │  TRANSFORM (feature engineering)
        ▼
   ├─ Workout_Intensity  = Avg_BPM / Max_BPM          (effort relative to your own max heart rate)
   ├─ Calories_per_Min   = Calories_Burned / (Session_Duration_hours × 60)
   ├─ HRR                = Max_BPM − Resting_BPM       (Heart Rate Reserve, a cardio-fitness marker)
   ├─ BMI_Category       = Underweight < 18.5 ≤ Normal < 25 ≤ Overweight < 30 ≤ Obese
   └─ Hydration_per_Kg   = Water_Intake_liters / Weight_kg
        │  LOAD
        ▼
featured_fitness_dataset.csv ──► EDA, hypothesis tests, model training
```

**Why winsorize instead of deleting outliers:** the dataset is small, and capping keeps every row while limiting the effect of extreme values.

**Why median/mode:** few values were missing, and the median isn't distorted by outliers.

**Known side effect:** winsorizing BMI on its own capped it at exactly 18.5, which hid 26 underweight users. `ImprovedModel.ipynb` recomputes BMI from weight and height to fix this.

---

## 6. Exploratory Data Analysis

Done in three layers in `FitInsights.ipynb`:

- **Univariate:** histograms, KDE plots and boxplots for every numeric column; count plots for categories.
- **Bivariate:** intensity vs calories, BMI vs fat %, calories/min by experience level, HRR by experience level.
- **Multivariate:** grouped bar and violin plots, a correlation heatmap, regression plots faceted by BMI category.

**Key findings:**

| Finding | Numbers |
|---|---|
| Workouts differ strongly in burn rate | CrossFit 12.95 and HIIT 12.79 kcal/min vs Yoga 6.17 and Pilates 6.10 |
| Longest sessions burn the most in total | Swimming 559 and Cycling 556 kcal per session (sessions ~49 min vs ~35 for HIIT/CrossFit) |
| Experience drives session length and total burn | Beginner 37 min / 370 kcal → Advanced 58 min / 513 kcal |
| Beginners burn the most *per minute* | 9.9 vs 8.5 kcal/min for professionals: shorter but harder sessions |
| Heavier bodies burn more per minute | Normal BMI 8.3 → Obese 12.0 kcal/min |
| Data is physiologically realistic | Age vs Max_BPM: r = −0.89 (matches the "220 − age" rule) |
| **People doing different workouts look alike** | Average age 32–35 and BMI 24.5–25.6 for every workout type. This is why workout *choice* can't be predicted from body measurements (see section 8). |

---

## 7. Hypothesis Testing

| # | Hypothesis | Test | Result |
|---|---|---|---|
| 1 | More experienced users burn more calories per session | One-way ANOVA (Beginner / Intermediate / Advanced) | F = 103.6, p ≈ 4.0 × 10⁻⁴³ → **Supported** |
| 2 | Higher workout intensity goes with higher water intake | Pearson correlation | r = 0.03, p = 0.22 → **Not supported** |
| 3 | Advanced users have a higher Heart Rate Reserve than beginners | Welch's t-test | t = 4.62, p ≈ 5.2 × 10⁻⁶ → **Supported** |
| 4 | Calories burned per minute differs by workout type | One-way ANOVA (14 types) | F = 128.0, p ≈ 4.5 × 10⁻²³⁹ → **Supported** |
| 5 | Higher body fat goes with lower hydration per kg | Spearman rank correlation | ρ = −0.21, p ≈ 1.1 × 10⁻¹⁷ → **Supported** |
| 6 | Higher intensity goes with more calories burned | Pearson correlation | r = 0.40, p ≈ 1.3 × 10⁻⁶⁴ → **Supported (moderate)** |

**Test choices:**
- **ANOVA** instead of several t-tests, to avoid inflating false positives.
- **Welch's t-test** because the two groups may have different variances.
- **Spearman** for hypothesis 5 because the relationship need not be linear.

**Caution on hypothesis 5:** `Hydration_per_Kg` depends more on weight (r = −0.74) than on water intake (r = −0.36), so part of this effect may simply be that people with more body fat weigh more.

**Takeaway used for modeling:** hypothesis 4 is the basis of the current model. Burn rate differs strongly by workout type, so it can be predicted per workout.

> The last few cells of `FitInsights.ipynb` (BMI vs hydration plots and correlations around r = −0.8 to −0.99) use **simulated data** generated with `np.random`, not the real dataset. They are illustrations only.

---

## 8. First Model Version (`PredictWorkout.ipynb`), kept for reference

- **Inputs:** `Age`, `Weight_kg`, `BMI`, `Calories_Burned` (scaled) + `Gender`, `Experience_Level`, `BMI_Category` (one-hot encoded).
- **Models:** `RandomForestClassifier` for `Workout_Type` (14 classes), tuned with `GridSearchCV` (best: 200 trees, max depth 10); `RandomForestRegressor` for session minutes.
- **Results** (re-evaluated in `ImprovedModel.ipynb` with 5-fold cross-validation):

| Task | Baseline | Random Forest |
|---|---|---|
| Workout type | 14.5% accuracy (always "Running") | **18.4% accuracy**, macro F1 0.15 |
| Duration | MAE 7.8 min | MAE 4.5 min, R² 0.65 (a linear model does the same) |

The workout-type classifier is barely better than guessing, because body measurements don't explain which workout people chose (section 6). This led to the redesign below.

---

## 9. Current Model (`ImprovedModel.ipynb`)

**Question asked:** *"If this person does workout X, how many calories per minute will they burn?"*

| | |
|---|---|
| Inputs | `Age`, `Weight_kg`, `BMI` (recomputed from weight/height) + `Gender`, `Experience_Level`, `Workout_Type` |
| Target | `Calories_per_Min` |
| Model | `HistGradientBoostingRegressor` in a scikit-learn `Pipeline` with `StandardScaler` + `OneHotEncoder` |
| Tuning | `GridSearchCV`, 5-fold, MAE; best: `max_depth=3`, `learning_rate=0.03`, `max_iter=500` |
| Not used | Heart-rate features: a new user doesn't know them before working out, so using them would be data leakage |

**Results:**

| Comparison (5-fold CV) | MAE (kcal/min) | R² |
|---|---|---|
| Baseline (mean) | 2.41 | 0.00 |
| Workout type only (linear) | 1.66 | 0.49 |
| Body + experience only (linear) | 1.96 | 0.35 |
| All inputs, Ridge (linear) | 0.83 | 0.86 |
| All inputs, Random Forest | 0.77 | 0.86 |
| **All inputs, Gradient Boosting** | **0.75** | **0.87** |

- **Held-out test set:** R² = **0.855**, MAE = **0.80 kcal/min**.
- **Total calories** (predicted rate × real session length): off by **32 kcal (7.9%)** on average.

**Recommendation logic.** The same logic runs in the FitSense API:

1. Predict kcal/min for all 14 workouts; `minutes = target ÷ rate`.
2. Beginners don't get HIIT or CrossFit. This is a safety rule, not learned from the data.
3. Split plans longer than 70.2 min (the longest session in the data) into equal sessions.
4. Rank single-session plans first, closest to the typical session for the experience level (Beginner 36.6, Intermediate 45.0, Advanced 58.2, Professional 60.6 min); split plans follow, least total time first.
5. Return the top 3. Targets are limited to 800 kcal (the data's maximum is ~806).

**Exports:** `fitinsight_calorie_rate_model.joblib` and `workout_profiles.json`.

Full write-up: [FitSense `docs/model.md`](https://github.com/minierks/FitSense/blob/main/docs/model.md).

---

## 10. Tech Stack

- **Python 3.11+**, Jupyter
- `pandas`, `numpy` · `matplotlib`, `seaborn` · `scipy.stats` (ANOVA, t-test, Pearson, Spearman)
- `scikit-learn` **1.9.1** (`Pipeline`, `ColumnTransformer`, `HistGradientBoostingRegressor`, `RandomForest*`, `GridSearchCV`, `cross_validate`)
- `joblib` for model export

---

## 11. How to Run

```bash
cd FitInsights
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib seaborn scipy scikit-learn==1.9.1 joblib jupyter
jupyter notebook
```

Run the notebooks top to bottom, in this order:

1. **`FitInsights.ipynb`**: cleaning, features, EDA, hypothesis tests. Writes `cleaned_fitness_dataset.csv` and `featured_fitness_dataset.csv`.
   - The first Heart Rate Reserve t-test cell uses `df_exp_hrr`, which is defined in a later cell. Run that later cell first (it starts with `df_exp_hrr = df[df['Experience_Level'].isin(...)]`), or skip the earlier one.
2. **`ImprovedModel.ipynb`**: baselines, the calorie-rate model, recommendation examples. Writes `fitinsight_calorie_rate_model.joblib` and `workout_profiles.json`.
3. *(Optional)* `PredictWorkout.ipynb`: the first-version models.

Keep scikit-learn at **1.9.1** to match the version pinned in FitSense's API. Pickled models are only guaranteed to load in the same version.

---

## 12. How This Feeds FitSense

```
FitInsights (this repo)                          FitSense (deployed)
───────────────────────                          ───────────────────
tracker_data.csv
   │ clean + engineer features
   ▼
fitinsights_data.csv
   │ ImprovedModel.ipynb
   ▼
fitinsight_calorie_rate_model.joblib ──copy──►  ml-api/  (Flask on Railway)
workout_profiles.json                ──copy──►     │ POST /predict → top 3 plans
                                                    ▼
                                                 project/ (Next.js on Vercel)
```

To update the live app:
1. Re-run `ImprovedModel.ipynb`.
2. Copy the two files into `FitSense/ml-api/`.
3. Run its tests and redeploy the API.

The raw data never leaves this repo. The API only loads the exported files.

---

## 13. Possible Future Improvements

- Turn the notebooks into scripts (e.g. `train.py`) so retraining is one command.
- Move imputation into the scikit-learn pipeline so it's learned from training data only.
- Recompute BMI after cleaning weight and height in `FitInsights.ipynb` itself.
- Track experiments with MLflow.
- Collect user feedback from FitSense to evaluate and retrain the recommender.
