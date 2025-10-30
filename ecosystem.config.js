module.exports = {
  apps: [
    {
      name: "fastapi-app",
      script: "python3",
      args: "-m uvicorn Api_api:app --host 127.0.0.1 --port 8000",
      cwd: "/root/Documents/fastapi",  // <-- change to your project path
      interpreter: "none", // ensures PM2 uses 'script' directly
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        ENV: "production"
      }
    }
  ]
}
