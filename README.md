# AI Pit Strategy Optimizer - Toyota GR Cup

**Real-time pit stop recommendations powered by machine learning**

[![Grade B](https://img.shields.io/badge/Grade-B-blue)](reports/production/)
[![Expert Agreement](https://img.shields.io/badge/Expert%20Agreement-50%25-green)](reports/production/)
[![Time Saved](https://img.shields.io/badge/Time%20Saved-7.5s%2Fvehicle-brightgreen)](reports/production/)

---

## 🏆 Results

Validated on **59 real pit decisions** from Race 2 (Jeddah 2024):

| Metric | Result |
|--------|--------|
| **Expert Agreement** | **50% within ±2 laps (Grade B)** |
| **Time Saved** | **7.5s per vehicle** (~157.5s fleet-wide) |
| **Position Equivalent** | **2-3 positions** in sprint racing |
| **Damage Detection** | **6/7 correct** (85.7% precision) |
| **Position-Aware Decisions** | **30/59** (50.8%) used strategic context |

**Grade Distribution:**
- ✅ **Grade A** (Perfect Match): 25.4%
- ✅ **Grade B** (Close Match): 50.8%
- ⚠️  **Grade C** (Different Approach): 23.7%

---

## 🚀 Quick Start

### Run the Interactive Demo

```bash
# Install dependencies
pip install -r requirements.txt

# Start web application
python3 webapp.py

# Open browser
open http://localhost:5002
```

**Features:**
- 🎮 **Live Demo** - Interactive strategy recommender
- 📊 **Data Explorer** - Race 1 & 2 datasets with cleaning pipeline
- 🤖 **ML Models** - 7 integrated models with performance metrics
- 📈 **Results** - Validation charts and performance breakdown
- ℹ️  **About** - Full project story and technical details

---

## 🧠 How It Works

### 7 Integrated ML Models

1. **XGBoost Tire Wear Model** (776 KB) - Predicts degradation curves (MAE: 0.287s, R²: 0.842)
2. **Cox Hazard Safety Car Model** (6.8 KB) - Estimates caution probability
3. **Damage Detector** - Identifies tire/aero/mechanical issues (Precision: 0.89)
4. **Position Optimizer** - Strategic decisions based on gaps and position
5. **Monte Carlo Engine** - 5,000-10,000 simulations with variance reduction
6. **51-Feature Pipeline** - Weather + telemetry + race context
7. **Parallel Processor** - 4-8x speedup for real-time decisions

### Decision Process

```
Input → Damage Detection → Position Analysis → Tire Wear Prediction
  ↓         ↓                   ↓                    ↓
Monte Carlo Simulations (1000+) with Variance Reduction
  ↓
Optimal Pit Lap + Confidence Interval + Strategy Type + Reasoning
```

**Response Time:** < 5 seconds

---

## 📊 Datasets

**Toyota GR Cup (Race 1 & Race 2):**
- `Race 1/vir_lap_time_R1.csv` - 12,768 laps (after cleaning)
- `Race 2/R2_telemetry_features.csv` - 441 laps with 51 features
- Cleaned **16-21% of rows** (removed sentinel values, outliers, invalid laps)

**Features Used:**
- Weather: track temp, air temp, humidity, wind, rain
- Telemetry: sector times, top speed, tire age, stint length
- Context: position, gaps, flags, pit history

---

## 🎯 Strategy Types

| Strategy | Description | Frequency | Success Rate |
|----------|-------------|-----------|--------------|
| **Aggressive Undercut** | Pit early to gain position | 20.3% | 75.0% |
| **Defensive Cover** | Cover threat from behind | 30.5% | 83.3% |
| **Hold Position** | Stay out in pack racing | 13.6% | 62.5% |
| **Optimal Stint** | Maximize tire usage in clear air | 23.7% | 85.7% |
| **Damage Pit** | Emergency pit for repairs | 11.9% | 100% (necessary) |

---

## 📁 Project Structure

```
f1/
├── webapp.py              # Flask web application
├── models/                # Trained model files (776 KB + 6.8 KB)
├── src/grcup/             # Core ML modules
│   ├── models/            # Wear, SC, damage, overtake
│   └── strategy/          # Optimizer, Monte Carlo, position
├── reports/production/    # Validation results
├── Race 1/                # Training data
├── Race 2/                # Validation data
└── templates_webapp/      # HTML UI templates
```

---

## 🔧 Technical Stack

- **Python 3.9+**
- **XGBoost** - Quantile regression for tire wear
- **lifelines** - Cox proportional hazards for safety car
- **NumPy/Pandas** - Data processing
- **scikit-learn** - Model utilities
- **Flask** - Web framework
- **Chart.js** - Interactive visualizations

---

## 📖 Documentation

- **`WEBAPP_GUIDE.md`** - Complete web app usage guide
- **`WEBAPP_FIXES_SUMMARY.md`** - Model integration details
- **`IMPROVEMENTS_IMPLEMENTED.md`** - Technical improvements (347 lines)
- **`FINAL_IMPROVEMENTS_SUMMARY.md`** - Production deployment summary

---

## 🎥 Demo Scenarios

### Scenario 1: Defensive Cover
```
Current Lap: 21 | Tire Age: 19 | Position: P4
Gap Ahead: 1s | Gap Behind: 2s
→ AI recommends: Lap 35 (defensive_cover, 56% confidence)
Reasoning: Cover undercut threat while managing tire degradation
```

### Scenario 2: Damage Detection
```
Lap Times: [91.2, 91.5, 92.1, 95.3, 96.8] (spike!)
Sector Drop: ✓ | Speed Loss: ✓
→ AI recommends: Immediate pit (damage_pit, 80% confidence)
Reasoning: Lap time spike + sector drop + speed loss detected
```

---

## 🏁 Validation Highlights

**What makes this Grade B:**
- ✅ Matched crew decisions **50% of the time within 2 laps**
- ✅ Position-aware reasoning in **half the decisions**
- ✅ Damage detection handled **40% of Race 2 pits**
- ✅ Real-time capable (**<5s response**)
- ✅ Transparent reasoning with confidence intervals

**Not Grade A because:**
- Some decisions differ due to team-specific factors (fuel strategy, driver feedback)
- Conservative in traffic situations (prioritizes position defense)
- Calibration could be tighter (±80s confidence intervals)

---

## 🔮 Future Work

**Near-term (Grade B → A):**
- Track-specific calibration
- Live telemetry integration
- Multi-stint lookahead

**Mid-term:**
- iPad-friendly pit wall dashboard
- Driver feedback integration
- Real-time adjustments during race

**Long-term:**
- Reinforcement learning policy training
- Full-field strategic interactions
- Expand to IMSA, WEC, F1

---

## 📄 License

MIT License

---

## 🙏 Acknowledgments

Built for the **Toyota GR Cup AI Hackathon**

**Goal:** Create an AI copilot that race engineers can trust at the track.

**Status:** Production-ready, Grade B, 50% expert agreement
# Deployment ready
