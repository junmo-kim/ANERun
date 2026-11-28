"""Loopback HTTP-to-stdio benchmark. Model load and image transfer are included in cold requests."""
import argparse
import base64
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import re
import subprocess
import threading
import time


def memory(pid):
    rows = subprocess.check_output(["zprint", "-t"], text=True).splitlines()
    tags = {r.split()[0]: int(r.split()[-1][:-1])*1024 for r in rows
            if re.match(r"^com\.apple\.driver\.AppleH[0-9]+ANEInterface(?:\.\S+)?\s+\d+\s+", r)
            and re.fullmatch(r"\d+K", r.split()[-1])}
    return dict(epoch=time.time(), ane_driver_bytes=sum(tags.values()) if tags else None,
                rss_kib=int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("bundle", type=Path)
    p.add_argument("fixtures", type=Path)
    p.add_argument("model", choices=["valen-4b", "glide-2b", "glide-9b"])
    p.add_argument("out", type=Path)
    p.add_argument("--all-goldens", action="store_true", help="time every reference case instead of the length sweep")
    args = p.parse_args()
    fixtures = json.loads((args.fixtures/"goldens.json").read_text())
    cases = []
    for fixture in fixtures:
        if fixture["model"] != args.model or (not args.all_goldens and fixture["id"] not in ("refund-en", "image-red-circle")):
            continue
        q = dict(name=fixture["name"], kind=fixture["type"], instructions=fixture["instructions"],
                 candidates=[dict(key=k,description=v) for k,v in fixture["criteria"]])
        body = dict(op="decide", state=fixture["state"], questions=[q])
        if fixture["id"] == "image-red-circle":
            body["image"] = base64.b64encode((args.fixtures/"red-circle.png").read_bytes()).decode()
        cases.append((fixture["id"],body))
    for repeats in ([] if args.all_goldens else [102] if args.model == "valen-4b" else [25,51]):
        cases.append((str(repeats), dict(op="decide", state="This is a neutral archived shipping status note. "*repeats+
                    " The customer requests a refund.", questions=[dict(name="refund",kind="noul",
                    instructions="Does the customer request a refund?",candidates=[])])))
    # A separate invocation of 'prepare' must already have exited on this machine.
    proc = subprocess.Popen([args.executable.resolve(),"worker",args.bundle.resolve()], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE)
    def command(body):
        proc.stdin.write(body+b"\n"); proc.stdin.flush()
        line=proc.stdout.readline()
        if not line:
            raise RuntimeError(f"worker exited {proc.poll()}")
        return line
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):
            pass
        def do_POST(self):
            data=command(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200); self.send_header("Content-Length",str(len(data))); self.end_headers()
            self.wfile.write(data)
    def post(body):
        data=json.dumps(body,separators=(",",":")).encode()
        conn=http.client.HTTPConnection(*server.server_address,timeout=600)
        try:
            start=time.time(); begin=time.monotonic()
            conn.request("POST","/",data)
            response=json.loads(conn.getresponse().read())
            return dict(start=start,end=time.time(),seconds=time.monotonic()-begin,result=response)
        finally:
            conn.close()
    report=dict(model=args.model,runs=[])
    try:
        with HTTPServer(("127.0.0.1",0),Handler) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
            try:
                report["baseline"]=memory(proc.pid)
                for i in range(5):
                    for name,body in cases:
                        assert post(dict(op="status"))["result"] is False
                        row=dict(case=name,cycle=i,samples=[post(body) for _ in range(3)])
                        assert post(dict(op="status"))["result"] is True
                        if i==0:
                            time.sleep(1)
                            row["loaded_idle"]=memory(proc.pid)
                        post(dict(op="unload"))
                        if i==0:
                            time.sleep(2)
                            row["after_unload"]=memory(proc.pid)
                        report["runs"].append(row)
                        args.out.write_text(json.dumps(report,indent=2))
            finally:
                server.shutdown(); thread.join()
    finally:
        proc.stdin.close(); report["worker_exit"]=proc.wait()
        args.out.write_text(json.dumps(report,indent=2))
    if report["worker_exit"]:
        raise RuntimeError("worker failed")


if __name__=="__main__":
    main()
