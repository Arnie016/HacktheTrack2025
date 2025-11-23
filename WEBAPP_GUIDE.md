# 🏎️ AI Pit Strategy Optimizer - Web Application Guide

## ✨ Features

A **comprehensive interactive web application** showcasing the entire AI Pit Strategy Optimizer system:

### 📱 Pages

1. **Home** - Project overview with key metrics
   - 50% Expert Agreement (Grade B)
   - 7.5s average time saved per vehicle
   - 7 integrated ML models

2. **Data Explorer** - Interactive dataset exploration
   - Race 1 & Race 2 datasets
   - Visual data cleaning pipeline (before/after stats)
   - Sample data tables with real telemetry
   - Cleaning steps breakdown (16-21% rows cleaned)

3. **ML Models** - Detailed model architecture
   - All 7 ML models with specifications
   - Feature lists for each model
   - Performance metrics (MAE, R², concordance, precision/recall)
   - File sizes and training data info

4. **Live Demo** - Interactive strategy recommender
   - Adjust race parameters (lap, tire age, position, gaps)
   - Set damage indicators
   - Get real-time AI recommendations with:
     - Recommended pit lap
     - Strategy type (undercut, defensive, optimal stint, damage pit)
     - Confidence level
     - Expected time gain
     - AI reasoning explanation

5. **Results** - Validation performance
   - Grade breakdown (A/B/C)
   - Time advantage analysis
   - Strategy type distribution
   - Success rates per strategy

6. **About** - Full project story
   - Inspiration (Formula 1 movie, pit strategy obsession)
   - Technical implementation details
   - Datasets, pipeline, models
   - Challenges faced
   - Accomplishments
   - Lessons learned
   - Future roadmap

---

## 🚀 Quick Start

### Start the Web Application

```bash
cd /Users/hema/Desktop/f1
python3 webapp.py
```

The server will start on **http://localhost:5002**

### Access the App

Open your browser and navigate to:
- **Main App:** http://localhost:5002
- **Data Explorer:** http://localhost:5002/data-explorer
- **ML Models:** http://localhost:5002/ml-models
- **Live Demo:** http://localhost:5002/live-demo
- **Results:** http://localhost:5002/results
- **About:** http://localhost:5002/about

---

## 🎨 UI Features

### Modern Design
- **Gradient backgrounds** (purple-to-blue theme)
- **Smooth animations** (fade-ins, hover effects)
- **Responsive layout** (works on desktop and mobile)
- **Cool navbar** with active page highlighting
- **Interactive sliders** for live demo parameters
- **Real-time API calls** for data and recommendations

### Data Visualization
- **Before/After comparison** for data cleaning
- **Stats grids** with key metrics
- **Sample data tables** from both races
- **Strategy comparison tables** with success rates
- **Grade performance breakdown** (A/B/C)

### Interactive Demo
- **Sliders** for race parameters (lap, tire age, position, gaps)
- **Checkboxes** for damage indicators
- **Lap time input** for recent performance
- **Real-time AI recommendations** with reasoning
- **Visual confidence indicators** and risk levels

---

## 📊 API Endpoints

The webapp exposes several API endpoints:

- `GET /api/dataset_summary/<race>` - Get Race 1 or Race 2 summary
- `GET /api/data_cleaning_stats` - Before/after cleaning statistics
- `GET /api/strategy_comparison` - Compare strategy types
- `GET /api/ml_model_info` - Detailed model specifications
- `POST /api/get_recommendation` - Get AI pit strategy recommendation
- `GET /api/validation_results` - Race 2 validation results

---

## 🧪 Demo Usage

### Try Different Scenarios

**Aggressive Undercut:**
- Current Lap: 20
- Tire Age: 12
- Position: P5
- Gap Ahead: 1.2s
- Gap Behind: 4.5s
- **Expected:** AI recommends pit in ~2 laps for undercut

**Defensive Cover:**
- Current Lap: 25
- Tire Age: 15
- Position: P3
- Gap Ahead: 5.0s
- Gap Behind: 1.8s
- **Expected:** AI recommends immediate pit to cover threat

**Damage Detection:**
- Recent Lap Times: 91.2, 91.5, 92.1, 95.3, 96.8 (sudden spike!)
- Check "Sector 3 time drop"
- Check "Top speed loss"
- **Expected:** AI detects damage, recommends emergency pit

---

## 🔧 Technical Stack

### Backend
- **Flask** - Web framework
- **Python 3.9+** - Core language
- **NumPy/Pandas** - Data processing
- **XGBoost** - Wear prediction model
- **lifelines** - Safety car hazard model

### Frontend
- **Vanilla HTML/CSS/JavaScript** - No framework dependencies
- **Responsive design** - Works on all devices
- **Modern CSS** - Gradients, animations, flexbox, grid
- **Fetch API** - Real-time data loading

---

## 📁 File Structure

```
f1/
├── webapp.py                      # Main Flask application
├── templates_webapp/              # HTML templates
│   ├── base.html                  # Base template with navbar
│   ├── index.html                 # Home page
│   ├── data_explorer.html         # Data exploration
│   ├── ml_models.html             # Model architecture
│   ├── live_demo.html             # Interactive demo
│   ├── results.html               # Validation results
│   └── about.html                 # Project story
├── static/                        # Static assets (future)
│   ├── css/
│   ├── js/
│   └── img/
├── data/                          # Race datasets
├── reports/                       # Validation reports
└── src/                           # ML models and logic
```

---

## 🎯 For Judges / Demo

### Narrative Flow

1. **Start at Home** - Show overview, key metrics (50% agreement, 7.5s saved)
2. **Data Explorer** - Demonstrate data cleaning (16-21% rows cleaned)
3. **ML Models** - Explain 7 integrated models with specs
4. **Live Demo** - Run 2-3 interactive scenarios:
   - Normal strategy (optimal stint)
   - Aggressive undercut scenario
   - Damage detection scenario
5. **Results** - Show validation against 59 real pit decisions
6. **About** - Tell the story (F1 movie inspiration → production system)

### Key Talking Points

✅ **Production-ready system** with comprehensive web interface  
✅ **Real validation** against actual Race 2 pit crew decisions  
✅ **7 integrated ML models** (not just one predictor)  
✅ **Data cleaning pipeline** with full transparency (flags, provenance)  
✅ **Position-aware strategy** (not just lap time optimization)  
✅ **Damage detection** (handles 40% of Race 2 cases)  
✅ **Interactive demo** for judges to try different scenarios  

---

## 🐛 Troubleshooting

### Port Already in Use
If port 5002 is in use:
```python
# Edit webapp.py, line at bottom:
app.run(debug=True, host="0.0.0.0", port=5003)  # Change port
```

### Models Not Loading
The webapp will work even if models aren't loaded (graceful degradation).
Check console output for warnings.

### Data Files Missing
If Race 1/2 CSV files are missing, Data Explorer will show a friendly error.
The rest of the app will still work.

---

## 🚀 Production Deployment

For production use (e.g., Heroku, AWS, DigitalOcean):

1. **Use a production WSGI server:**
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5002 webapp:app
   ```

2. **Set debug=False** in webapp.py

3. **Add environment variables** for sensitive config

4. **Use nginx** as reverse proxy for static files

---

## 📝 License

MIT License - See main project README

---

## 🔗 Links

- **GitHub Repo:** https://github.com/Arnie016/racing-f1-hackthon
- **Category:** Real-Time Analytics
- **Status:** Production Ready, Grade B, 50% Expert Agreement

---

**Built for the Toyota GR Cup AI Hackathon**  
*Matching professional pit crews with machine learning* 🏁

