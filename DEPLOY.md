# One-command deploy (Railway)

```bash
# 1. Install Railway CLI
npm install -g @railway/cli

# 2. Login and deploy
railway login
railway init
railway up

# 3. Set env vars
railway variables set GEMINI_API_KEY=AIzaSy...
railway variables set GROQ_API_KEY=gsk_...

# 4. Open your live URL
railway open
```

Your URL: `https://meta-screening-[hash].up.railway.app`

Share this URL — anyone can open it in Chrome/Edge/Safari and call the agent.
