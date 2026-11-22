"""Serve only SHA-256-verified public image blobs on loopback for local Docker import."""
from pathlib import Path
import hashlib,http.server,json,re
ROOT=Path('/data/by/genie_sim/dockerhub-sidebuild-20261119')
MANIFEST=(ROOT/'verified-image-manifest.raw.json').read_bytes()
ORIGIN=json.loads((ROOT/'verified-image-origin.json').read_text())

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_HEAD(self):self.respond(False)
    def do_GET(self):self.respond(True)
    def respond(self,body):
        if self.path.rstrip('/')=='/v2':
            self.send_response(200);self.send_header('Docker-Distribution-Api-Version','registry/2.0');self.send_header('Content-Length','2');self.end_headers()
            if body:self.wfile.write(b'{}')
            return
        if '/manifests/' in self.path:
            ref=self.path.rsplit('/',1)[-1]
            if ref not in ('geniesim4',ORIGIN['digest']):self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type',ORIGIN['media_type']);self.send_header('Docker-Content-Digest',ORIGIN['digest']);self.send_header('Content-Length',str(len(MANIFEST)));self.end_headers()
            if body:self.wfile.write(MANIFEST)
            return
        if '/blobs/' in self.path:
            digest=self.path.rsplit('/',1)[-1]
            if not re.fullmatch(r'sha256:[0-9a-f]{64}',digest):self.send_error(404);return
            path=ROOT/'verified-blobs'/digest.split(':',1)[1]
            if not path.is_file():self.send_error(404);return
            size=path.stat().st_size;start=0;end=size-1
            if self.headers.get('Range'):
                match=re.fullmatch(r'bytes=(\d+)-(\d*)',self.headers['Range'])
                if not match:self.send_error(416);return
                start=int(match[1]);end=min(int(match[2]) if match[2] else size-1,size-1)
            self.send_response(206 if start or self.headers.get('Range') else 200)
            self.send_header('Content-Type','application/octet-stream');self.send_header('Docker-Content-Digest',digest)
            self.send_header('Content-Length',str(end-start+1));self.send_header('Accept-Ranges','bytes')
            if self.headers.get('Range'):self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
            self.end_headers()
            if body:
                try:
                    with path.open('rb') as f:
                        f.seek(start);remaining=end-start+1
                        while remaining:
                            chunk=f.read(min(1024*1024,remaining))
                            if not chunk:break
                            self.wfile.write(chunk);remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError):pass
            return
        self.send_error(404)

if __name__=='__main__':http.server.ThreadingHTTPServer(('127.0.0.1',5009),Handler).serve_forever()
