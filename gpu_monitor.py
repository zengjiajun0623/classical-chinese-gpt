#!/usr/bin/env python3
"""Live GPU monitor for the RTX 3080 (jiajuns-pc), served from the Mac mini.

Polls `nvidia-smi` on pc3080 over SSH every few seconds and serves a live
dashboard at http://localhost:7787 (auto-refreshes; no dependencies).

Run:  python3 ~/llm-lab/gpu_monitor.py
"""
import json, subprocess, threading, time
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = 7787
HOST = "pc3080"            # from ~/.ssh/config
POLL_S = 3
HISTORY = deque(maxlen=1200)   # ~1h at 3s
STATE = {"online": False, "procs": [], "last": None}

QUERY = ("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,"
         "temperature.gpu,power.draw --format=csv,noheader,nounits && "
         "nvidia-smi --query-compute-apps=process_name,used_memory "
         "--format=csv,noheader,nounits")

def poll_loop():
    while True:
        try:
            out = subprocess.run(
                ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes", HOST, QUERY],
                capture_output=True, text=True, timeout=15)
            lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
            if out.returncode != 0 or not lines:
                raise RuntimeError(out.stderr.strip() or "no output")
            util, mem, mem_tot, temp, power = [float(x) for x in lines[0].split(",")]
            procs = []
            for l in lines[1:]:
                parts = l.rsplit(",", 1)
                if len(parts) == 2:
                    name = parts[0].strip().split("\\")[-1]
                    raw = parts[1].strip()
                    try:
                        pmem = float(raw)          # Windows graphics apps report [N/A]
                    except ValueError:
                        pmem = None
                    procs.append({"name": name, "mem": pmem})
            # compute processes (real VRAM number, e.g. python) first, then by name
            procs.sort(key=lambda p: (p["mem"] is None, p["name"].lower()))
            sample = {"t": time.time(), "util": util, "mem": mem,
                      "mem_tot": mem_tot, "temp": temp, "power": power}
            HISTORY.append(sample)
            STATE.update(online=True, procs=procs, last=sample)
        except Exception:
            STATE.update(online=False, procs=[])
        time.sleep(POLL_S)

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>3080 · live</title>
<style>
:root{--bg:#F7F6F2;--surface:#FFF;--ink:#232833;--muted:#6E7380;--line:#E6E4DC;
--grid:#EFEDE6;--accent:#B8432F;--accent-soft:#B8432F22;--good:#3E7A5E;--bad:#B8432F;
--mono:ui-monospace,SFMono-Regular,Menlo,monospace;
--sans:system-ui,-apple-system,"PingFang SC",sans-serif}
@media (prefers-color-scheme: dark){:root{--bg:#151922;--surface:#1C212C;--ink:#E8E6E0;
--muted:#9BA0AB;--line:#2A303C;--grid:#232936;--accent:#D96A52;--accent-soft:#D96A5226;--good:#5FA37F}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans)}
.wrap{max-width:880px;margin:0 auto;padding:28px 20px}
h1{font-size:1.15rem;margin:0;display:flex;align-items:center;gap:10px}
.dot{width:10px;height:10px;border-radius:50%;background:var(--bad);transition:background .3s}
.dot.on{background:var(--good)}
.sub{color:var(--muted);font-size:.8rem;margin:4px 0 18px}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:18px}
.chip{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:8px 14px;min-width:104px}
.chip .k{color:var(--muted);text-transform:uppercase;font-size:.62rem;letter-spacing:.08em}
.chip .v{font-family:var(--mono);font-size:1.25rem;font-weight:600;font-variant-numeric:tabular-nums}
.chip .u{color:var(--muted);font-size:.75rem}
.card{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:14px}
.card h2{margin:0 0 8px;font-size:.8rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:600}
canvas{width:100%;height:120px;display:block}
table{width:100%;border-collapse:collapse;font-size:.85rem}
td{padding:6px 4px;border-top:1px solid var(--grid)}
td.m{font-family:var(--mono);text-align:right;font-variant-numeric:tabular-nums}
.empty{color:var(--muted);font-size:.85rem;padding:6px 4px}
footer{color:var(--muted);font-size:.72rem;margin-top:10px}
</style></head><body><div class="wrap">
<h1><span class="dot" id="dot"></span>RTX 3080 · jiajuns-pc <span id="state" style="color:var(--muted);font-weight:400;font-size:.85rem"></span></h1>
<p class="sub">polled over SSH every 3s from the Mac mini · keep this tab open during training</p>
<div class="chips">
 <div class="chip"><div class="k">GPU util</div><div class="v" id="util">–</div><div class="u">%</div></div>
 <div class="chip"><div class="k">VRAM</div><div class="v" id="mem">–</div><div class="u" id="memu">GB</div></div>
 <div class="chip"><div class="k">Temp</div><div class="v" id="temp">–</div><div class="u">°C</div></div>
 <div class="chip"><div class="k">Power</div><div class="v" id="power">–</div><div class="u">W</div></div>
</div>
<div class="card"><h2>GPU utilization · last hour</h2><canvas id="cUtil"></canvas></div>
<div class="card"><h2>VRAM used · last hour</h2><canvas id="cMem"></canvas></div>
<div class="card"><h2>Processes on the GPU</h2><div id="procs"></div></div>
<footer id="foot"></footer>
</div>
<script>
const css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
function draw(id,pts,maxV,fmt){
  const c=document.getElementById(id),dpr=devicePixelRatio||1;
  const w=c.clientWidth,h=c.clientHeight;c.width=w*dpr;c.height=h*dpr;
  const x=c.getContext("2d");x.scale(dpr,dpr);x.clearRect(0,0,w,h);
  x.strokeStyle=css("--grid");x.lineWidth=1;
  [0.25,0.5,0.75].forEach(f=>{x.beginPath();x.moveTo(0,h*f);x.lineTo(w,h*f);x.stroke();});
  if(pts.length<2)return;
  const px=i=>i/(pts.length-1)*w, py=v=>h-(v/maxV)*(h-6)-3;
  x.beginPath();x.moveTo(px(0),h);pts.forEach((v,i)=>x.lineTo(px(i),py(v)));x.lineTo(w,h);x.closePath();
  x.fillStyle=css("--accent-soft");x.fill();
  x.beginPath();pts.forEach((v,i)=>i?x.lineTo(px(i),py(v)):x.moveTo(px(i),py(v)));
  x.strokeStyle=css("--accent");x.lineWidth=1.8;x.stroke();
  const lv=pts[pts.length-1];x.beginPath();x.arc(w-2,py(lv),3,0,7);x.fillStyle=css("--accent");x.fill();
  x.fillStyle=css("--muted");x.font="10px ui-monospace";x.textAlign="right";
  x.fillText(fmt(lv),w-8,py(lv)-6);
}
async function tick(){
  try{
    const d=await (await fetch("/data.json")).json();
    document.getElementById("dot").className="dot"+(d.online?" on":"");
    document.getElementById("state").textContent=d.online?(d.last&&d.last.util>10?"· training":"· idle"):"· offline";
    if(d.last){
      document.getElementById("util").textContent=d.last.util.toFixed(0);
      document.getElementById("mem").textContent=(d.last.mem/1024).toFixed(1);
      document.getElementById("memu").textContent="/ "+(d.last.mem_tot/1024).toFixed(0)+" GB";
      document.getElementById("temp").textContent=d.last.temp.toFixed(0);
      document.getElementById("power").textContent=d.last.power.toFixed(0);
    }
    draw("cUtil",d.history.map(s=>s.util),100,v=>v.toFixed(0)+"%");
    const mt=d.history.length?d.history[d.history.length-1].mem_tot:10240;
    draw("cMem",d.history.map(s=>s.mem),mt,v=>(v/1024).toFixed(1)+"G");
    const P=document.getElementById("procs");
    const shown=d.procs.slice(0,12);
    P.innerHTML=shown.length
      ?"<table>"+shown.map(p=>{
          const hot=p.name.toLowerCase().includes("python");
          const nm=hot?`<b style="color:var(--accent)">${p.name}</b>`:p.name;
          const mem=p.mem==null?"–":(p.mem/1024).toFixed(2)+" GB";
          return `<tr><td>${nm}</td><td class="m">${mem}</td></tr>`;}).join("")
        +"</table>"+(d.procs.length>12?`<div class='empty'>+${d.procs.length-12} more</div>`:"")
      :"<div class='empty'>nothing running on the GPU</div>";
    document.getElementById("foot").textContent="last sample "+(d.last?new Date(d.last.t*1000).toLocaleTimeString():"–");
  }catch(e){document.getElementById("dot").className="dot";document.getElementById("state").textContent="· monitor unreachable";}
}
tick();setInterval(tick,3000);
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path == "/data.json":
            body = json.dumps({"online": STATE["online"], "procs": STATE["procs"],
                               "last": STATE["last"], "history": list(HISTORY)[-1200:]}).encode()
            ct = "application/json"
        else:
            body = PAGE.encode(); ct = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    threading.Thread(target=poll_loop, daemon=True).start()
    print(f"GPU monitor: http://localhost:{PORT}  (polling {HOST} every {POLL_S}s)")
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
