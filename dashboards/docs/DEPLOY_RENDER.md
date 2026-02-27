# ☁️ Deploy to Render

Your dashboard is configured for automatic deployment on Render using Infrastructure as Code (IaC).

## Option 1: Blueprints (Recommended)

Render uses `render.yaml` (Blueprint) to configure your service automatically.

1. **Push your code** to GitHub.
   - [Kshitij3h/crop-faq-dashboard](https://github.com/Kshitij3h/crop-faq-dashboard)
   - [vicharanashala/crop-faq-dashboard](https://github.com/vicharanashala/crop-faq-dashboard)

2. Go to **[Render Dashboard](https://dashboard.render.com)**.
3. Click **New +** -> **Blueprint**.
4. Connect your GitHub account and select the repository.
5. Click **Apply**.
   - Render will see `render.yaml` and configure everything automatically.

## Option 2: Manual Web Service

If you prefer manual setup:

1. Click **New +** -> **Web Service**.
2. Connect the repository.
3. Use these settings:
   - **Runtime**: Node
   - **Build Command**: `npm install && npm run build`
   - **Start Command**: `npm start`
