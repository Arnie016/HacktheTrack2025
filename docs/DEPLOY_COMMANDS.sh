#!/bin/bash
# Vercel Deployment Commands

echo "🚀 Deploying to Vercel..."
echo ""

# Navigate to project
cd /Users/hema/Desktop/f1

# Deploy to Vercel
# When prompted:
# 1. Select: https://github.com/Arnie016/HacktheTrack2025.git (hackthetrack)
# 2. Confirm project settings (press Enter)
# 3. Wait for deployment

vercel --prod

echo ""
echo "✅ Deployment complete!"
echo "Your app will be live at the URL shown above"

