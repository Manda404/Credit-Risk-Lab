"""Send held-out applications to the API at a configurable interval."""

import argparse
import json
import time
from urllib import error, request
import pandas as pd

from credit_risk_lab.config.settings import settings


def post_json(url: str, payload: dict, timeout: float = 10.0) -> tuple[int, dict]:
    """POST JSON using the standard library so the simulator stays lightweight."""
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def main() -> None:
    """Stream raw test rows while keeping labels outside API requests."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/v1/predict")
    parser.add_argument("--interval", type=float, default=2.0, help="Seconds between requests")
    parser.add_argument("--limit", type=int, default=0, help="0 sends every row")
    args = parser.parse_args()
    frame = pd.read_csv(settings.test_path)
    if args.limit > 0:
        frame = frame.head(args.limit)
    for index, row in frame.iterrows():
        expected = int(row[settings.target_column])
        payload = row.drop(labels=[settings.target_column]).to_dict()
        status, response = post_json(args.url, payload)
        print(json.dumps({"row": int(index), "expected": expected, "http": status, "response": response}))
        if args.interval > 0:
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
