import threading
import time

from .internal_utils import exporter
export, __all__ = exporter()


class LockWrapper:
    def __init__(self, lock_type):
        self.lock = lock_type()
        self._locked = 0

    def acquire(self, blocking=True, timeout=-1):
        acquired = self.lock.acquire(blocking=blocking, timeout=timeout)
        if acquired:
            self._locked += 1
        return Locked(self, acquired)

    def release(self):
        self._locked -= 1
        return self.lock.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        return self.release()

    @property
    def locked(self):
        return self._locked


@export
class Lock(LockWrapper):
    def __init__(self):
        super().__init__(threading.Lock)


@export
class RLock(LockWrapper):
    def __init__(self):
        super().__init__(threading.RLock)


class Locked:
    def __init__(self, lock, acquired):
        self.lock = lock
        self.acquired = acquired

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.acquired:
            return self.lock.release()

    def __getattr__(self, attr):
        return getattr(self.lock, attr)

    def __bool__(self):
        return self.acquired


@export
class Timer:
    def __init__(self, func, interval=None, *, args=None, kwargs=None):
        self.func = func
        self.interval = interval
        self.args = args or []
        self.kwargs = kwargs or {}
        self.last_args = None
        self.last_kwargs = None
        self.ret = None

        self.rlock = RLock()
        self.timer = None
        self.start_time = None
        self.end_time = None
        self._expired = threading.Event()
        self._canceled = threading.Event()

    @property
    def remaining(self):
        with self.rlock:
            if self.end_time is not None:
                return max(self.end_time - time.time(), 0)

    def callback(self, *args, **kwargs):
        with self.rlock:
            self._expired.set()
            self.timer = None
            self.ret = self.func(*args, **kwargs)

    def start(self, interval=None, *, args=None, kwargs=None):
        with self.rlock:
            if self.timer:
                return False

            self.last_args = args or self.args
            self.last_kwargs = kwargs or self.kwargs

            self._expired.clear()
            self._canceled.clear()
            self.timer = threading.Timer(
                    interval or self.interval, self.callback,
                    self.last_args, self.last_kwargs)
            self.start_time = time.time()
            self.end_time = self.start_time + self.timer.interval
            self.timer.start()
            return True

    def cancel(self):
        with self.rlock:
            if not self.timer:
                return False

            self.timer.cancel()
            self.timer = None
            self._canceled.set()
            return True

    def join(self):
        return self.timer.join()

    @property
    def active(self):
        with self.rlock:
            return self.timer is not None

    @property
    def expired(self):
        with self.rlock:
            return self._expired.is_set()

    @property
    def idle(self):
        with self.rlock:
            return not self.active or self.expired

    @property
    def canceled(self):
        with self.rlock:
            return self._canceled.is_set()


class Throttler:
    def __init__(self, func, interval):
        if not callable(func):
            raise TypeError('func must be a callable')

        self.func = func
        self.interval = interval

        self.timestamp = None
        self.timer = Timer(self.lopri)

        self.trtl_lock = Lock()
        self.main_lock = Lock()

    def callback(self, *args, **kwargs):
        ret = self.func(*args, **kwargs)
        self.timestamp = time.time()
        return ret

    def lopri(self, *args, **kwargs):
        # throttling: block simultaneous callers
        with self.trtl_lock.acquire(blocking=False) as tl:
            if not tl:
                return None

            # throttling: drop fast caller that is waiting for make up
            if self.timer.active:
                self.timer.cancel()

            # throttling: defer fast callers
            if self.timestamp is None:
                delta = self.interval
            else:
                delta = time.time() - self.timestamp

            if delta < self.interval:
                self.timer.start(self.interval - delta, args=args, kwargs=kwargs)
                return self.timer

            with self.main_lock.acquire(blocking=False) as ml:
                if not ml:
                    return None

                return self.callback(*args, **kwargs)

    def hipri(self, *args, **kwargs):
        with self.main_lock:
            self.timer.cancel()
            return self.callback(*args, **kwargs)

    def __call__(self, *, blocking=False, args=[], kwargs={}):
        if blocking:
            return self.hipri(*args, **kwargs)
        else:
            return self.lopri(*args, **kwargs)
