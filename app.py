from flask import Flask, request, Response, render_template, stream_with_context
from src.run_main import run
from argparse import Namespace
import io
import sys
from rich.console import Console
import time
from queue import Queue
import threading

app = Flask(__name__)

class StreamingStdout:
    def __init__(self, q: Queue):
        self.q = q
        self.buffer = ""

    def write(self, data):
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self.q.put(line + "\n")  # keep the newline

    def flush(self):
        pass

def stream_output(args):
    q = Queue()

    old_stdout = sys.stdout
    sys.stdout = StreamingStdout(q)

    buffer = io.StringIO()
    console = Console() 

    def worker():
        try:
            result, _ = run(args, console)
            q.put(f"\n=== FINAL RESULT ===\n{result}\n")
        finally:
            sys.stdout = old_stdout
            q.put(None) 

    threading.Thread(target=worker).start()

    while True:
        line = q.get()
        if line is None:
            break
        yield line

@app.route("/run_stream", methods=["POST"])
def run_app():
    data = request.get_json()
    args = Namespace(
        dataset=data.get("dataset"),
        result_path=data.get("result_path", "results.json"),
        model_name=data["model_name"],
        local_model=data.get("local_model", False),
        id=data.get("id", "manual"),
        source_paper_title=data.get("source_paper_title"),
        target_paper_title=data.get("target_paper_title"),
        excerpt=data.get("excerpt"),
        year=data.get("year", 2025),
        skip_citations=data.get("skip_citations", ""),
        temperature=data.get("temperature", 0.95),
    )

    return Response(stream_with_context(stream_output(args)),
                    mimetype="text/plain; charset=utf-8")

@app.route("/test")
def test_stream():
    def generate():
        buffer = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buffer  # redirect print to buffer

        try:
            # Example prints
            for i in range(5):
                print(f"Line {i}")
                time.sleep(1)

        finally:
            sys.stdout = old_stdout  # restore stdout

        buffer.seek(0)
        for line in buffer:
            yield line

        yield "\n=== FINAL RESULT ===\n"

    return Response(stream_with_context(generate()), mimetype="text/plain")

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")  # serve a HTML page

if __name__ == "__main__":
    app.run(debug=True)

