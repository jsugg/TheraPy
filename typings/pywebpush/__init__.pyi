from collections.abc import Mapping

class WebPushResponse:
    status_code: int


def webpush(
    *,
    subscription_info: Mapping[str, object],
    data: str,
    vapid_private_key: str,
    vapid_claims: Mapping[str, str],
    ttl: int,
    timeout: int,
) -> WebPushResponse: ...
