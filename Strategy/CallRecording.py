from datetime import datetime, timezone


class CallRecording:
    """
    Mixin for the Aggregate subclasses that make new judge calls for an analysis (RQ2 CrossJudge, RQ1-KJ MenuJudge).
    Put it before Aggregate in the bases. Every record also stores its call's provider model version, UTC time and
    API usage ("call"); the file metadata counts the returned model versions and keeps the first / last call time.
    """
    lastCall = None

    def onResponse(self, response):
        super().onResponse(response)
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.lastCall = {"model_version": response.model_version, "called_at": now,
                         "usage_in": response.usage_in, "usage_out": response.usage_out}
        versions = self.store.metadata.setdefault("model_versions", {})
        versions[response.model_version] = versions.get(response.model_version, 0) + 1
        self.store.metadata.setdefault("calls_utc", {"first": now})["last"] = now

    def aggregateItem(self, item_id, presentation_order) -> dict:
        self.lastCall = None
        record = super().aggregateItem(item_id, presentation_order)
        record["call"] = self.lastCall
        return record
