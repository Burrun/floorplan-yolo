import sys
import time
import subprocess
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("watchdog package is required. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "watchdog", "--break-system-packages"])
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler

class JupytextSyncHandler(FileSystemEventHandler):
    def __init__(self):
        self.last_sync = 0
        self.cooldown = 2 # Prevent infinite sync loops

    def on_modified(self, event):
        if event.is_directory:
            return
        
        path = Path(event.src_path)
        if path.suffix in ['.ipynb', '.py']:
            # Ignore hidden files or directories
            if any(part.startswith('.') for part in path.parts):
                return
            
            current_time = time.time()
            if current_time - self.last_sync > self.cooldown:
                print(f"[{time.strftime('%H:%M:%S')}] Detected change in {path.name}. Syncing...")
                try:
                    subprocess.run(["jupytext", "--sync", str(path)], check=True)
                    self.last_sync = time.time()
                    print(f"[{time.strftime('%H:%M:%S')}] Sync complete.")
                except subprocess.CalledProcessError as e:
                    print(f"Sync failed: {e}")

if __name__ == "__main__":
    path = "." if len(sys.argv) < 2 else sys.argv[1]
    event_handler = JupytextSyncHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"Jupytext watcher started. Monitoring {path} for .ipynb/.py changes...")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
