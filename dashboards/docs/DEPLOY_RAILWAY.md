# 🚂 Deploy to Railway

Your dashboard is configured for zero-config deployment on Railway.

## Option 1: Using GitHub (Recommended)

1. **Push your code** to one of these repositories:
   - [Kshitij3h/crop-faq-dashboard](https://github.com/Kshitij3h/crop-faq-dashboard)
   - [vicharanashala/crop-faq-dashboard](https://github.com/vicharanashala/crop-faq-dashboard)

2. Go to **[Railway Dashboard](https://railway.app)**.
3. Click "New Project" -> "Deploy from GitHub repo".
4. Select the repository you pushed to.
5. Railway will automatically detect the `Dockerfile` and `package.json` and deploy it.

## Option 2: Using Railway CLI

If you have the Railway CLI installed:

1. Login:
   ```bash
   railway login
   ```

2. Initialize and Deploy:
   ```bash
   # Navigate to project root
   cd dashboards
   railway init
   railway up
   ```

## Configuration Details
- **Build Command**: `npm run build` (handled by Dockerfile)
- **Start Command**: `npm start` (serves the `dist` folder on port 3000)
- **Port**: Railway automatically sets the `$PORT` environment variable, which the app listens to.
