# 🚀 Vercel Deployment Guide

## Quick Deploy

### Option 1: GitHub Integration (Recommended)

1. **Push to GitHub** (already done ✅)
   ```bash
   git push hackthetrack main
   ```

2. **Import to Vercel**
   - Go to https://vercel.com/new
   - Select "Import Git Repository"
   - Choose: `Arnie016/HacktheTrack2025`
   - Click "Import"

3. **Configure Build Settings**
   - Framework Preset: **Other**
   - Root Directory: `./`
   - Build Command: (leave empty)
   - Output Directory: (leave empty)

4. **Environment Variables** (Optional)
   ```
   FLASK_ENV=production
   MC_BASE_SCENARIOS=500
   USE_VARIANCE_REDUCTION=1
   ```

5. **Deploy**
   - Click "Deploy"
   - Wait ~2-3 minutes
   - Get URL: `https://hack-the-track2025.vercel.app`

---

### Option 2: Vercel CLI

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Deploy
cd /Users/hema/Desktop/f1
vercel

# Follow prompts:
# - Set up and deploy? Y
# - Which scope? (your account)
# - Link to existing project? N
# - Project name? hack-the-track-2025
# - Directory? ./
# - Override settings? N

# Production deployment
vercel --prod
```

---

## ⚠️ Important Notes

### File Size Limits
- Vercel has **50MB serverless function limit**
- Your models:
  - `wear_quantile_xgb.pkl` = 776 KB ✅
  - `cox_hazard.pkl` = 6.8 KB ✅
  - Total well under limit

### Data Files
- CSV files in `data/` will be included
- Total ~8 files, ~1-2 MB ✅

### Cold Start
- First request may take 5-10 seconds (cold start)
- Subsequent requests: <5 seconds
- Monte Carlo reduced to 500 scenarios for speed

---

## 🎯 Expected URL Structure

After deployment, your app will be at:
```
https://hack-the-track-2025.vercel.app/
https://hack-the-track-2025.vercel.app/live-demo
https://hack-the-track-2025.vercel.app/data-explorer
https://hack-the-track-2025.vercel.app/results
```

---

## 🔧 Configuration Files

### `vercel.json`
```json
{
  "version": 2,
  "builds": [
    {
      "src": "webapp.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "webapp.py"
    }
  ],
  "env": {
    "FLASK_ENV": "production",
    "MC_BASE_SCENARIOS": "500"
  }
}
```

### `requirements.txt`
Already exists ✅

---

## 📊 Performance on Vercel

**Expected:**
- Cold start: 5-10s (first request)
- Warm requests: 2-5s
- Monte Carlo: 500 scenarios (reduced from 1000 for speed)
- Memory: ~512 MB (sufficient for models)

---

## 🐛 Troubleshooting

### Build Fails
```bash
# Check logs
vercel logs

# Common issues:
# - Missing dependencies → Check requirements.txt
# - Large files → Check .vercelignore
# - Python version → Vercel uses Python 3.9 by default
```

### Runtime Errors
```bash
# Check function logs
vercel logs --follow

# Common issues:
# - Model not loading → Check file paths (use relative paths)
# - Import errors → Check sys.path in webapp.py
# - Timeout → Reduce MC_BASE_SCENARIOS to 250
```

---

## ✅ Post-Deployment Checklist

- [ ] Homepage loads
- [ ] Data Explorer shows Race 1 & 2 data
- [ ] Live Demo returns recommendations
- [ ] Results page shows charts
- [ ] Model status shows "LOADED"
- [ ] Response time <10s (cold start) or <5s (warm)

---

## 🔗 Custom Domain (Optional)

Once deployed:
1. Go to Vercel Dashboard → Your Project → Settings → Domains
2. Add custom domain: `aipitstrategy.com` (if you own it)
3. Follow DNS configuration steps

---

## 📝 What Gets Deployed

```
Deployed to Vercel:
✅ webapp.py (Flask app)
✅ templates_webapp/ (HTML UI)
✅ src/grcup/ (ML models code)
✅ models/ (trained .pkl files)
✅ data/ (CSV files)
✅ requirements.txt
✅ vercel.json (config)

NOT deployed:
❌ notebooks/ (too large, not needed)
❌ Race 1/ and Race 2/ folders (old structure)
❌ .git/ (source control)
❌ __pycache__/ (compiled Python)
```

---

## 🎉 After Deployment

**Share the link:**
```
🏁 AI Pit Strategy Optimizer
Live Demo: https://hack-the-track-2025.vercel.app
GitHub: https://github.com/Arnie016/HacktheTrack2025

Try the live demo to see real AI pit strategy recommendations!
- 50% expert agreement (Grade B)
- 7.5s average time saved per vehicle
- Real trained models with 1000+ Monte Carlo simulations
```

---

## 🚀 Deploy Now

**Fastest method:**
1. Go to https://vercel.com/new
2. Import `Arnie016/HacktheTrack2025`
3. Click "Deploy"
4. Done! 🎉

**Estimated time:** 2-3 minutes

