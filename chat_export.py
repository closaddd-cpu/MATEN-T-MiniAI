# BY MATEN/T | MATEN/T MiniAI v1.1 | M/ License v1.0
"""Chat with exported safetensors weights in a terminal or local browser."""
from __future__ import annotations
from brand_guard import enforce_attribution
enforce_attribution()
import argparse
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock
import torch
import sentencepiece as spm
from export_model import load_export
from pretrain import device_for
from miniai import infer

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser(description='MATEN/T MiniAI v1.1 — chat from safetensors')
    p.add_argument('--release',default=str(ROOT/'pretrain_checkpoints'/'release'))
    p.add_argument('--device',default='auto',choices=['auto','cpu','cuda','mps'])
    p.add_argument('--serve',action='store_true')
    p.add_argument('--port',type=int,default=8765)
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--temperature',type=float,default=0.8)
    args=p.parse_args()
    device=device_for(args.device)
    model,cfg=load_export(args.release,device)
    tok=spm.SentencePieceProcessor(model_file=str(Path(args.release)/'tokenizer.model'))
    def answer(message):
        return infer(model,cfg,tok,device,message,args.max_new_tokens,args.temperature)
    if not args.serve:
        print('BY MATEN/T | local chat. Type /exit to quit. Model output may be inaccurate.')
        while True:
            try:prompt=input('You> ').strip()
            except (EOFError,KeyboardInterrupt):break
            if prompt in ('/exit','/quit'):break
            if prompt:print('MiniAI>',answer(prompt))
        return
    page=(ROOT/'ui'/'index.html').read_bytes()
    lock=Lock()
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path!='/':self.send_error(404);return
            self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
            self.send_header('Content-Length',str(len(page)));self.end_headers();self.wfile.write(page)
        def do_POST(self):
            if self.path!='/api/chat':self.send_error(404);return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=16000:raise ValueError('Request too large or empty')
                body=json.loads(self.rfile.read(n))
                prompt=body.get('message')
                if not isinstance(prompt,str) or not 0<len(prompt)<=4000:raise ValueError('Invalid message')
                lang=body.get('language','auto')
                from catalog import NATURAL
                if lang!='auto':
                    if lang not in NATURAL:raise ValueError('Unknown language')
                    prompt=f'Reply in {NATURAL[lang][0]}.\n'+prompt
                with lock:response=answer(prompt)
                status=200;payload={'answer':response or '[No text generated]'}
            except (TypeError,ValueError,json.JSONDecodeError) as exc:
                status=400;payload={'error':str(exc)}
            except Exception:
                status=500;payload={'error':'Generation failed; check logs'}
            raw=json.dumps(payload,ensure_ascii=False).encode()
            self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    print(f'BY MATEN/T | Browser: http://127.0.0.1:{args.port}',flush=True)
    ThreadingHTTPServer(('127.0.0.1',args.port),Handler).serve_forever()

if __name__=='__main__':main()
