import time
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

DATA_DIR = "dataset/train"

class Handler(FileSystemEventHandler):
    def on_created(self, event):
        if event.is_directory:
            return

        print(f"[+] New file detected: {event.src_path}")

        # add + commit + push
        subprocess.run(["git", "add", "dataset/"])
        subprocess.run(["git", "commit", "-m", "auto: new data"])
        subprocess.run(["git", "push"])

event_handler = Handler()
observer = Observer()
observer.schedule(event_handler, DATA_DIR, recursive=True)

observer.start()
print("Watching for new data...")

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    observer.stop()

observer.join()