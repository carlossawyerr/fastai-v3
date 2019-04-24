from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

#export_file_url = 'https://drive.google.com/uc?export=download&id=1-U9w-BnrlZz36QcrYfhtrJvX6AEAID7d'
#export_file_name = 'export-2.pkl'
#classes = ['Coats', 'Dresses', 'Jeans', 'Shoes', 'Shorts', 'Skirts', 'Tops']

export_file_url = 'https://drive.google.com/uc?export=download&id=1-dSEYIP-QFo3VimSBmC5ft7fRyV4UtJE'
export_file_name = 'export-3.pkl'
classes =['3/4_Sleeve',
 'A_Line',
 'Accessories',
 'Baby_Doll',
 'Back_Detail',
 'Ball_Gown',
 'Bell_Sleeve',
 'Below_the_Knee',
 'Belt',
 'Body-Con',
 'Choker',
 'Cold_Shoulder',
 'Cutout_Dress',
 'Denim',
 'Dress',
 'Embellished',
 'Empire_Waist',
 'Fit_&_Flare',
 'Floral',
 'Gown',
 'Halter',
 'Halter_Dress',
 'High_Low',
 'Jacket',
 'Jacket_Dress',
 'Knee_Length',
 'Lace_Dress',
 'Lace_Up',
 'Long',
 'Long_Sleeve',
 'Maternity',
 'Maxi',
 'Maxi_Dress',
 'Mermaid',
 'Midi',
 'Mini',
 'Mock_Neck',
 'Nursing',
 'Off_The_Shoulder',
 'One_Shoulder',
 'Outfit',
 'Pants',
 'Peasant',
 'Peplum_Dress',
 'Pump',
 'Ruffle',
 'Scarf',
 'Sequined',
 'Sheath',
 'Shift',
 'Shirt_Dress',
 'Shoe',
 'Short',
 'Short_Sleeve',
 'Shorts',
 'Skirts',
 'Sleeveless',
 'Slip_Dress',
 'Strapless',
 'Suit',
 'Sun_Dress',
 'Sweater',
 'Sweater_Dress',
 'Sweatshirt',
 'Swimsuit',
 'T-Shirt',
 'Top',
 'Trapeeze',
 'Two_Piece_Dress',
 'Ugly_Christmas',
 'Velvet',
 'Wedding_Dress',
 'Wrap_Dress']

path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)[0] 
    return JSONResponse({'result': str(prediction)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
