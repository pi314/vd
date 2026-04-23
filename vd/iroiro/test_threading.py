import threading

from .lib_test_utils import *

import iroiro as iro


class TestLock(TestCase):
    def test_lock(self):
        my_lock = iro.Lock()
        py_lock = threading.Lock()
        self.eq(type(my_lock.lock), type(py_lock))

        self.false(my_lock.locked)

        with my_lock:
            self.true(my_lock.locked)

        self.false(my_lock.locked)

        with my_lock.acquire() as ac:
            self.true(ac)
            self.true(my_lock.locked)

            with my_lock.acquire(blocking=False) as ac2:
                self.false(ac2)
                self.true(ac2.locked)

            self.true(ac)
            self.true(my_lock.locked)

        self.false(my_lock.locked)

    def test_rlock(self):
        my_lock = iro.RLock()
        py_lock = threading.RLock()
        self.eq(type(my_lock.lock), type(py_lock))

        self.eq(my_lock.locked, 0)
        with my_lock:
            self.eq(my_lock.locked, 1)
            with my_lock:
                self.eq(my_lock.locked, 2)
            self.eq(my_lock.locked, 1)
        self.eq(my_lock.locked, 0)


class TestTimer(TestCase):
    def check_status(self, timer, status):
        self.true(getattr(timer, status))
        self.check_consistency(timer)

    def check_expired(self, timer, value):
        self.eq(timer.expired, value)
        self.check_consistency(timer)

    def check_consistency(self, timer):
        if timer.active:
            self.false(timer.idle)
            self.false(timer.expired)
            self.false(timer.canceled)
        else:
            self.true(timer.idle or timer.expired or timer.canceled)

        if timer.expired:
            self.true(timer.idle)
            self.false(timer.active)
            self.false(timer.canceled)

        if timer.canceled:
            self.true(timer.idle)
            self.false(timer.active)
            self.false(timer.expired)

        if timer.idle:
            self.false(timer.active)

        self.true(any([timer.active, timer.idle, timer.expired, timer.canceled]))

    def test_timer_interval_and_default_interval(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()

        def foo():
            checkpoint.set()

        timer = iro.Timer(foo, 10)
        timer.start()

        import time
        time.sleep(10)
        checkpoint.wait()

        checkpoint.clear()
        timer.start(20)
        time.sleep(10)
        checkpoint.check(False)
        time.sleep(10)
        checkpoint.wait()

    def test_timer_args_and_default_args(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)
        import time

        checkpoint = self.checkpoint()

        test_args = None
        test_kwargs = None
        got_args = None
        got_kwargs = None
        def foo(*args, **kwargs):
            nonlocal got_args, got_kwargs
            got_args, got_kwargs = args, kwargs
            checkpoint.set()
            return len(got_args) + len(got_kwargs)

        test_args = ('wah',)
        test_kwargs = {'key': 'value'}
        timer = iro.Timer(foo, 10, args=test_args, kwargs=test_kwargs)
        timer.start()
        time.sleep(10)
        checkpoint.wait()
        self.eq(got_args, test_args)
        self.eq(got_kwargs, test_kwargs)
        self.eq(timer.last_args, test_args)
        self.eq(timer.last_kwargs, test_kwargs)
        self.eq(timer.ret, 2)
        checkpoint.clear()

        test_args2 = ('wah', 'wow', 'weee')
        test_kwargs2 = {'key': 'value', 'key2': 'value2'}
        timer.start(args=test_args2, kwargs=test_kwargs2)
        time.sleep(10)
        checkpoint.wait()
        self.eq(timer.args, test_args)
        self.eq(timer.kwargs, test_kwargs)
        self.eq(got_args, test_args2)
        self.eq(got_kwargs, test_kwargs2)
        self.eq(timer.last_args, test_args2)
        self.eq(timer.last_kwargs, test_kwargs2)
        self.eq(timer.ret, 5)

    def test_timer_start_expire(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()

        def foo():
            checkpoint.set()

        timer = iro.Timer(foo, 10)
        self.check_status(timer, 'idle')

        timer.start()
        self.check_status(timer, 'active')
        checkpoint.check(False)

        import time
        time.sleep(10)
        checkpoint.wait()
        self.check_status(timer, 'expired')

    def test_timer_start_cancel(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()

        def foo():
            checkpoint.set()

        timer = iro.Timer(foo, 10)
        self.check_status(timer, 'idle')

        timer.start()
        self.check_status(timer, 'active')
        checkpoint.check(False)

        self.true(timer.cancel())
        checkpoint.check(False)
        self.check_status(timer, 'canceled')

    def test_timer_start_expire_cancel(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()

        def foo():
            checkpoint.set()

        timer = iro.Timer(foo, 10)
        self.check_status(timer, 'idle')

        timer.start()
        self.check_status(timer, 'active')
        checkpoint.check(False)

        import time
        time.sleep(10)
        checkpoint.wait()
        self.check_status(timer, 'expired')

        self.false(timer.cancel())
        self.check_status(timer, 'expired')

    def test_timer_start_twice(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()

        def foo():
            checkpoint.set()

        timer = iro.Timer(foo, 10)
        self.check_status(timer, 'idle')

        timer.start()
        self.check_status(timer, 'active')
        checkpoint.check(False)

        res = timer.start()
        self.false(res)
        self.check_status(timer, 'active')

    def test_timer_idle_cancel(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        def foo(*args, **kwargs):
            pass

        timer = iro.Timer(foo, 10)
        self.check_status(timer, 'idle')

        self.false(timer.cancel())
        self.check_status(timer, 'idle')

    def test_timer_join(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)

        checkpoint = self.checkpoint()

        def foo():
            checkpoint.set()

        timer = iro.Timer(foo, 10)
        self.check_status(timer, 'idle')
        self.false(timer.expired)
        self.true(timer.idle)
        self.false(timer.canceled)

        timer.start()
        self.check_status(timer, 'active')
        checkpoint.check(False)

        e = threading.Event()
        def move_time():
            e.wait()
            import time
            for i in range(10):
                time.sleep(i)

        t = threading.Thread(target=move_time, daemon=True)
        t.start()

        e.set()
        timer.join()
        t.join()
        checkpoint.wait()
        self.check_status(timer, 'expired')

    def test_timer_remaing_time(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)
        import time

        checkpoint = self.checkpoint()
        def foo():
            checkpoint.set()

        timer = iro.Timer(foo, 10)
        self.eq(timer.remaining, None)

        timer.start()
        self.eq(timer.remaining, 10)

        time.sleep(3.14)
        self.eq(timer.remaining, 10 - 3.14)

        time.sleep(7)
        self.eq(timer.remaining, 0)


class TestThrottler(TestCase):
    def test_non_callable(self):
        with self.raises(TypeError):
            iro.threading.Throttler(False, 1)

    def test_blocking_calls(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)
        import time

        record = []
        def foo(*args, **kwargs):
            record.append((time.time(), args, kwargs))
            return args[0] if args else 'foo'

        th = iro.threading.Throttler(foo, interval=1)

        ret = th(blocking=True)
        self.eq(ret, 'foo')

        ret = th(blocking=True, args=['wah'], kwargs={'key': 'value'})
        self.eq(ret, 'wah')

        ret = th(blocking=True, args=['wee'], kwargs={'key2': 'value2'})
        self.eq(ret, 'wee')

        self.eq(record, [
            (0, tuple(), dict()),
            (0, tuple(['wah']), {'key': 'value'}),
            (0, tuple(['wee']), {'key2': 'value2'}),
            ])

    def test_non_blocking_calls(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)
        import time

        record = []
        def foo(*args, **kwargs):
            record.append((time.time(), args, kwargs))
            return args[0] if args else 'foo'

        th = iro.threading.Throttler(foo, interval=1)

        ret = th()
        self.eq(ret, 'foo')

        time.sleep(1)
        ret = th(args=['wah'], kwargs={'key': 'value'})
        self.eq(ret, 'wah')

        time.sleep(0.5)
        ret = th(args=['ignored'], kwargs={'ignored': 'ignored'})
        self.isinstance(ret, iro.Timer)
        self.eq(ret.remaining, 0.5)

        time.sleep(0.4)
        ret = th(args=['wee'], kwargs={'key2': 'value2'})
        self.isinstance(ret, iro.Timer)
        self.almost_eq(ret.remaining, 0.1)

        time.sleep(0.1)

        self.eq(record, [
            (0, tuple(), dict()),
            (1, tuple(['wah']), {'key': 'value'}),
            (2, tuple(['wee']), {'key2': 'value2'}),
            ])

    def test_lopri_calls_compete(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)
        import time
        time.sleep(1)

        barrier = threading.Barrier(2)
        first_caller_set = threading.Event()

        record = []
        def foo(*args, **kwargs):
            record.append((time.time(), args, kwargs))
            first_caller_set.set()
            barrier.wait()

        th = iro.threading.Throttler(foo, interval=1)

        t = threading.Thread(target=lambda:th(args=['first caller']), daemon=True)
        t.start()

        first_caller_set.wait()
        th(args=['fast caller'])
        th(args=['fast caller2'])

        barrier.wait()
        t.join()

        self.eq(record, [
            (1, tuple(['first caller']), dict()),
            ])

    def test_lopri_and_hipri_calls_compete(self):
        fake_time = FakeTime()
        for name, func in fake_time.patch():
            self.patch(name, func)
        import time
        time.sleep(1)

        barrier = threading.Barrier(2)

        record = []
        def foo(*args, **kwargs):
            record.append((time.time(), args, kwargs))
            barrier.wait()

        th = iro.threading.Throttler(foo, interval=1)

        t = threading.Thread(target=lambda:th(blocking=True, args=['first caller']), daemon=True)
        t.start()

        ret = th(args=['fast caller'])
        self.eq(ret, None)

        ret = th(args=['fast caller2'])
        self.eq(ret, None)

        barrier.wait()
        t.join()

        self.eq(record, [
            (1, tuple(['first caller']), dict()),
            ])
