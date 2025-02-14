# bind = "127.0.0.1:8001"
bind = "0.0.0.0:8003"
workers = 4
timeout = 2000

loglevel = 'debug'
pidfile = "log/gunicorn.pid"
accesslog = "log/access.log"
errorlog = "log/debug.log"
preload_app = True
