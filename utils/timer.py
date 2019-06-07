import time
import torch


class Timer:
    """Context manager to time a piece of code, including GPU synchronization."""
    def __init__(self, as_ms=False):
        self.start, self.end = None, None
        self.scale = 1000 if as_ms else 1
        self.is_gpu = torch.cuda.is_available()

    def __enter__(self):
        if self.is_gpu:
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.is_gpu:
            torch.cuda.synchronize()
        self.end = time.perf_counter()

    @property
    def elapsed(self):
        return self.scale * (self.end - self.start) if self.end else None


def main():
    with Timer() as t:
        time.sleep(2)
    print(f'{t.elapsed} secs')

    with Timer(as_ms=True) as t:
        time.sleep(0.002)
    print(f'{t.elapsed} ms')


if __name__ == '__main__':
    main()
