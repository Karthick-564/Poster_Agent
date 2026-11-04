"""The local page: form renders, file serving stays inside runs/, bad requests are refused."""
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from poster_agent import webapp


def _serve():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), webapp.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_form_renders_and_bad_requests_and_path_escapes_are_refused():
    httpd, base = _serve()
    try:
        assert b"Make the poster" in urllib.request.urlopen(base + "/").read()
        for path in ("/file/../pyproject.toml", "/file/x/..%2f..%2fpyproject.toml", "/file/a/b/c"):
            try:
                urllib.request.urlopen(base + path)
                raise AssertionError(f"{path} should not be served")
            except urllib.error.HTTPError as e:
                assert e.code == 404
        req = urllib.request.Request(base + "/make", data=b"prompt=&brand=kct&format=story", method="POST")
        try:
            urllib.request.urlopen(req)
            raise AssertionError("an empty prompt should be refused")
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        httpd.shutdown()
