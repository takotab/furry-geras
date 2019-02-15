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

# https://syncwithtech.blogspot.com/p/direct-download-link-generator.html
model_file_url = (
    "https://drive.google.com/uc?export=download&id=1RVKXdDggdXnb9UU4efosCBLZMkj0vXtN"
)
model_file = "saved_model/pose_resnet_50_256x192.pth.tar"

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
    print("done downloading")


async def setup_mdl():
    await download_file(model_file_url, Path(model_file))


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_mdl())]
mdl = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


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
    pose_vid = motion.make_posevid(filename)
    return JSONResponse({"result": pose_vid.split("/")[-2]})


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="0.0.0.0", port=8080)
