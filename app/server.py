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

export_file_url = 'https://drive.google.com/uc?export=download&id=1-7d_1Efqi5llfzVxAlGDNvR_wB90QEpe'
export_file_name = 'export-7.pkl'
classes =['100%_Cotton_Performance',
 '3/4_Sleeve',
 'A_Line',
 'Acrylic',
 'All_Weather',
 'Animal_Print',
 'Baby_Doll',
 'Back_Detail',
 'Ball_Gown',
 'Beads',
 'Bell_Sleeve',
 'Below_the_Knee',
 'Black',
 'Blouse',
 'Blue',
 'Boat_Neck',
 'Body-Con',
 'Brown',
 'Camo',
 'Check',
 'Chiffon',
 'Choker',
 'Cold_Shoulder',
 'Collared',
 'Colorblock',
 'Compression',
 'Cotton',
 'Cotton_&_Cotton_Blend',
 'Cotton_Blend',
 'Cotton_Jersey',
 'Cowl_Neck',
 'Crepe',
 'Crew_Neck',
 'Cutout_Dress',
 'Denim',
 'Destructed',
 'Dolman',
 'Dots',
 'Embellished',
 'Embroidered',
 'Embroidery',
 'Empire_Waist',
 'Exposed_Zipper',
 'Faux_Leather',
 'Fit_&_Flare',
 'Floral',
 'Flutter',
 'Geometric',
 'Georgette',
 'Gingham',
 'Gold',
 'Gown',
 'Graphic',
 'Gray',
 'Green',
 'Halter',
 'Halter_Dress',
 'Hardware',
 'Henley',
 'Herringbone',
 'High_Low',
 'Hooded',
 'Illusion',
 'Ivory/Cream',
 'Jacket_Dress',
 'Jersey',
 'Keyhole',
 'Knee_Length',
 'Knit',
 'Lace',
 'Lace_Dress',
 'Lace_Up',
 'Leather',
 'Leather_or_Suede',
 'Linen',
 'Long',
 'Long_Sleeve',
 'Lycra/Spandex',
 'Matte_Jersey',
 'Maxi',
 'Maxi_Dress',
 'Mermaid',
 'Mesh',
 'Metallic_Knit',
 'Midi',
 'Mock_Neck',
 'Multi',
 'Nylon',
 'Off_The_Shoulder',
 'Off_the_Shoulder',
 'One_Shoulder',
 'Orange',
 'Other',
 'Peasant',
 'Peplum_Dress',
 'Pink',
 'Plaid',
 'Poly_Blend',
 'Polyester',
 'Ponte',
 'Print',
 'Puff_Sleeve',
 'Purple',
 'Rayon',
 'Rayon_Spandex',
 'Red',
 'Rhinestones',
 'Roll_Tab',
 'Ruffle',
 'Satin',
 'Scoop_Neck',
 'Scuba',
 'Sequin',
 'Sequined',
 'Sequins',
 'Shawl',
 'Sheath',
 'Shift',
 'Shirt_Dress',
 'Short',
 'Short_Sleeve',
 'Silk',
 'Silver',
 'Sleeveless',
 'Slip_Dress',
 'Solid',
 'Split_Neck',
 'Split_Sleeve',
 'Square',
 'Strapless',
 'Stretch',
 'Stretch_Twill',
 'Stripe',
 'Studs',
 'Suede',
 'Sun_Dress',
 'Sweater_Dress',
 'Sweatshirt',
 'Sweetheart',
 'Synthetic',
 'T-Shirt',
 'Tan/Beige',
 'Tie_Dye',
 'Turtleneck',
 'Tweed',
 'Two_Piece_Dress',
 'V-Neck',
 'Velvet',
 'Wedding_Dress',
 'White',
 'Wool_&_Wool_Blend',
 'Wrap_Dress',
 'Yellow']

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
