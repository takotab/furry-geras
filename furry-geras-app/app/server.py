from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from starlette.endpoints import HTTPEndpoint
import uvicorn, aiohttp, asyncio
from io import BytesIO
from pathlib import Path
import os
import sys
import motion

from .model_url_dest import mdl_url_dest

path = Path(__file__).parent
app = Starlette()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["X-Requested-With", "Content-Type"],
)
app.mount("/static", StaticFiles(directory="app/static"))


async def download_file(url, dest):
    if dest.exists():
        return
    print("downloading...")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, "wb") as f:
                f.write(data)
    print(f"done downloading {url}")


async def setup_mdl(key):
    c_url_dst = mdl_url_dest[key]
    await download_file(c_url_dst["url"], Path(c_url_dst["dest"]))


loop = asyncio.get_event_loop()
tasks = [
    asyncio.ensure_future(setup_mdl("pose")),
    asyncio.ensure_future(setup_mdl("orient")),
    asyncio.ensure_future(setup_mdl("detect_human_ssd")),
]
loop.run_until_complete(asyncio.gather(*tasks))
loop.close()
vid2pose = motion.Video2Pose(device="cuda:0")


@app.route("/")
def index(request):
    html = path / "view" / "index.html"
    return HTMLResponse(html.open().read())


@app.route("/output/{file}.mp4")
class User(HTTPEndpoint):
    async def get(self, request):
        file = request.path_params["file"]
        print(file)
        return FileResponse(os.path.join("output", str(file), "posevid.mp4"))


@app.route("/analyze", methods=["POST"])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data["file"].read())
    filename = os.path.join(motion.filename_maker(), "pose_vid.mp4")
    with open(filename, "bw") as f:
        f.write(img_bytes)
    pose_vid = vid2pose.make_posevid(filename)
    return JSONResponse({"result": pose_vid.split("/")[-2]})


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8080)
