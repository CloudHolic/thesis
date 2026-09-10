"""Fetching `.osu` chart files from the community mirrors."""

from __future__ import annotations

import random
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path

# How hard one request tries before the next mirror gets a turn. Fixed: these are
# politeness, not something a run has a reason to vary.
RETRIES = 5
BASE_DELAY = 1.0
MAX_DELAY = 30.0
TIMEOUT = 20.0

_BOM = b"\xef\xbb\xbf"
_HEADER = b"osu file format"


class FetchError(Exception):
	"""A chart could not be retrieved from any mirror."""


@dataclass(frozen=True, slots=True)
class Mirrors:
	"""Where to ask for a chart, and how to identify ourselves while asking."""

	urls: tuple[str, ...]
	user_agent: str


@dataclass(frozen=True, slots=True)
class Fetched:
	"""What one pass retrieved, and why the rest did not arrive."""

	ok: tuple[int, ...]
	failed: dict[int, str]


class _RateLimiter:
	"""Spaces requests across threads so the mirrors see one steady stream."""

	def __init__(self, per_second: float) -> None:
		self.min_interval = 1.0 / per_second if per_second > 0 else 0.0
		self._lock = threading.Lock()
		self._next = 0.0

	def acquire(self) -> None:
		"""Block until this thread's turn to send."""
		if self.min_interval <= 0.0:
			return

		with self._lock:
			at = max(self._next, time.monotonic())
			self._next = at + self.min_interval

		wait = at - time.monotonic()
		if wait > 0:
			time.sleep(wait)


def _backoff(attempt: int) -> float:
	return random.uniform(0.0, min(MAX_DELAY, BASE_DELAY * (2.0**attempt)))


def _retry_after(err: urllib.error.HTTPError) -> float | None:
	value = err.headers.get("Retry-After") if err.headers else None
	if value is None:
		return None

	try:
		return float(value)
	except ValueError:
		return None


def _download(url: str, user_agent: str) -> bytes:
	request = urllib.request.Request(url, headers={"User-Agent": user_agent})
	with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
		body = response.read()

	# A mirror that does not have the map may answer with a placeholder page under a 200.
	if not body.removeprefix(_BOM).lstrip().startswith(_HEADER):
		raise FetchError("response is not a .osu file")

	return bytes(body)


def path_for(directory: Path, beatmap_id: int) -> Path:
	"""Where one chart file lives once it has been fetched."""
	return directory / f"{beatmap_id}.osu"


def fetch_one(
	beatmap_id: int,
	*,
	directory: Path,
	mirrors: Mirrors,
	limiter: _RateLimiter | None = None,
	refresh: bool = False,
) -> Path:
	"""One chart, from the first mirror that serves it. Already-fetched files are kept."""
	path = path_for(directory, beatmap_id)
	if path.exists() and path.stat().st_size > 0 and not refresh:
		return path

	last: Exception | None = None
	for template in mirrors.urls:
		url = template.format(beatmap_id=beatmap_id)

		for attempt in range(RETRIES + 1):
			if limiter is not None:
				limiter.acquire()

			try:
				body = _download(url, mirrors.user_agent)
			except urllib.error.HTTPError as err:
				last = err
				# A 404 means this mirror does not have it. Only rate limits and server
				# faults are worth asking the same mirror again.
				if err.code != 429 and not 500 <= err.code < 600:
					break

				delay = _retry_after(err) if err.code == 429 else None
				if delay is None:
					delay = _backoff(attempt)
			except (urllib.error.URLError, TimeoutError, HTTPException, FetchError) as err:
				last = err
				delay = _backoff(attempt)
			else:
				path.parent.mkdir(parents=True, exist_ok=True)
				# Written aside and moved, so an interrupted run leaves no half file behind.
				partial = path.with_name(f"{path.name}.part")
				partial.write_bytes(body)
				partial.replace(path)

				return path

			if attempt < RETRIES:
				time.sleep(delay)

	raise FetchError(f"{beatmap_id}: every mirror failed ({last})")


def prefetch(
	beatmap_ids: Iterable[int],
	*,
	directory: Path,
	mirrors: Mirrors,
	jobs: int,
	rate: float,
	refresh: bool = False,
) -> Fetched:
	"""Fetch every chart, naming the ones that never arrived rather than dropping them."""
	ids = sorted({int(b) for b in beatmap_ids})
	limiter = _RateLimiter(rate)
	ok: list[int] = []
	failed: dict[int, str] = {}

	with ThreadPoolExecutor(max_workers=jobs) as pool:
		futures = {
			pool.submit(
				fetch_one,
				beatmap_id,
				directory=directory,
				mirrors=mirrors,
				limiter=limiter,
				refresh=refresh,
			): beatmap_id
			for beatmap_id in ids
		}
		for future in as_completed(futures):
			beatmap_id = futures[future]
			try:
				future.result()
			except FetchError as err:
				failed[beatmap_id] = str(err)
			else:
				ok.append(beatmap_id)

	return Fetched(ok=tuple(sorted(ok)), failed=failed)
