import unittest

from fastapi import HTTPException

from app.api.security import RequestGuard


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class RequestGuardTests(unittest.TestCase):
    def test_api_key_is_optional(self):
        RequestGuard(request_limit=0).check(None, "client")

    def test_api_key_uses_constant_time_validation(self):
        guard = RequestGuard(api_key="secret", request_limit=0)
        with self.assertRaises(HTTPException) as context:
            guard.check("wrong", "client")
        self.assertEqual(context.exception.status_code, 401)
        guard.check("secret", "client")

    def test_rate_limit_is_per_client_and_resets(self):
        clock = FakeClock()
        guard = RequestGuard(request_limit=2, window_seconds=10, clock=clock)
        guard.check(None, "first")
        guard.check(None, "first")
        guard.check(None, "second")

        with self.assertRaises(HTTPException) as context:
            guard.check(None, "first")
        self.assertEqual(context.exception.status_code, 429)
        self.assertEqual(context.exception.headers, {"Retry-After": "10"})

        clock.now = 10
        guard.check(None, "first")

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            RequestGuard(request_limit=-1)
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            RequestGuard(window_seconds=0)


if __name__ == "__main__":
    unittest.main()
