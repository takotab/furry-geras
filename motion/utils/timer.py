import time


class Timer:
    def __init__(self):
        self.clock = {}

    def start(self, key="default"):
        self.clock[key] = time.time()
        return self

    def end(self, key="default", msg=None):
        interval = self.lap(key=key, msg=msg)
        del self.clock[key]
        return interval

    def lap(self, key="default", msg=None):
        if key not in self.clock:
            raise Exception(f"{key} is not in the clock.")
        interval = time.time() - self.clock[key]
        if msg:
            print(f"Elapsed time {msg}: {str(interval)}")
        return interval
