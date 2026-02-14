import importlib

from flask import Flask

from pdf_compressor import bootstrap
from pdf_compressor.factory import create_app


def test_create_app_registers_expected_routes(monkeypatch):
    monkeypatch.setattr("pdf_compressor.factory.bootstrap.bootstrap_runtime", lambda: None)
    app = create_app()

    rules = {rule.rule for rule in app.url_map.iter_rules()}
    expected = {
        "/",
        "/health",
        "/compress-async",
        "/compress-sync",
        "/status/<job_id>",
        "/download/<filename>",
        "/diagnose/<job_id>",
    }
    assert expected.issubset(rules)


def test_bootstrap_runtime_is_idempotent(monkeypatch):
    calls = {"set_processor": 0, "start_workers": 0, "thread_start": 0}

    class DummyThread:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            calls["thread_start"] += 1

    def _set_processor(*args, **kwargs):
        del args, kwargs
        calls["set_processor"] += 1

    def _start_workers(*args, **kwargs):
        del args, kwargs
        calls["start_workers"] += 1

    monkeypatch.setattr(bootstrap, "_bootstrap_started", False)
    monkeypatch.setattr(bootstrap.threading, "Thread", DummyThread)
    monkeypatch.setattr(bootstrap.compression_service.job_queue, "set_processor", _set_processor)
    monkeypatch.setattr(bootstrap.compression_service.job_queue, "start_workers", _start_workers)

    bootstrap.bootstrap_runtime()
    bootstrap.bootstrap_runtime()

    assert calls["set_processor"] == 1
    assert calls["start_workers"] == 1
    assert calls["thread_start"] == 1


def test_root_app_shim_exposes_gunicorn_app():
    app_module = importlib.import_module("app")
    assert hasattr(app_module, "app")
    assert isinstance(app_module.app, Flask)
