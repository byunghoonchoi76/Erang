import re, requests, os, time
FID = "1HgT1GBRh8bNwomNuaL_lmM60Tan03AfK"
OUT = os.path.expanduser("~/geoit_sr_refiner/data/goit_supperres_data.zip")
s = requests.Session()
r = s.get("https://drive.google.com/uc?export=download", params={"id": FID}, stream=True)
if "text/html" in r.headers.get("Content-Type", ""):
    html = r.text
    action = re.search(r'action="([^"]+)"', html).group(1)
    fields = dict(re.findall(r'name="([^"]+)" value="([^"]*)"', html))
    r = s.get(action, params=fields, stream=True)
total = int(r.headers.get("Content-Length", "0"))
done = 0; t0 = time.time(); last = 0
print(f"START total={total/1e9:.2f}GB", flush=True)
with open(OUT, "wb") as f:
    for ch in r.iter_content(1 << 20):
        f.write(ch); done += len(ch)
        if done - last >= 500 * (1 << 20):
            last = done; el = time.time() - t0
            print(f"{done/1e9:.2f}/{total/1e9:.2f} GB  {done/el/1e6:.1f} MB/s  {el:.0f}s", flush=True)
print(f"DONE bytes={os.path.getsize(OUT)}", flush=True)
