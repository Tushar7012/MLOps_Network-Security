from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import HTMLResponse, RedirectResponse
from uvicorn import run as app_run

from typing import Optional

# Importing constants and pipeline modules from the project
from src.constants import APP_HOST, APP_PORT
from src.pipline.prediction_pipeline import PhishingData, PhishingClassifier
from src.pipline.training_pipeline import TrainPipeline

# Initialize FastAPI application
app = FastAPI()

# Mount the 'static' directory for serving static files (like CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 template engine for rendering HTML templates
templates = Jinja2Templates(directory='templates')

# Allow all origins for Cross-Origin Resource Sharing (CORS)
origins = ["*"]

# Configure middleware to handle CORS, allowing requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.Having_IP_Address: Optional[int] = None
        self.URL_Length: Optional[int] = None
        self.Shortining_Service: Optional[int] = None
        self.Having_At_Symbol: Optional[int] = None
        self.Prefix_Suffix: Optional[int] = None
        self.SSLfinal_State: Optional[int] = None
        self.Domain_registeration_length: Optional[int] = None
        self.Favicon: Optional[int] = None
        self.HTTPS_token: Optional[int] = None
        self.Request_URL: Optional[int] = None

    async def get_phishing_data(self):
        form = await self.request.form()
        self.Having_IP_Address = int(form.get("Having_IP_Address"))
        self.URL_Length = int(form.get("URL_Length"))
        self.Shortining_Service = int(form.get("Shortining_Service"))
        self.Having_At_Symbol = int(form.get("Having_At_Symbol"))
        self.Prefix_Suffix = int(form.get("Prefix_Suffix"))
        self.SSLfinal_State = int(form.get("SSLfinal_State"))
        self.Domain_registeration_length = int(form.get("Domain_registeration_length"))
        self.Favicon = int(form.get("Favicon"))
        self.HTTPS_token = int(form.get("HTTPS_token"))
        self.Request_URL = int(form.get("Request_URL"))

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
            "phishing.html", {"request": request, "context": "Rendering"})

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful!!!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_phishing_data()

        phishing_data = PhishingData(
            Having_IP_Address=form.Having_IP_Address,
            URL_Length=form.URL_Length,
            Shortining_Service=form.Shortining_Service,
            Having_At_Symbol=form.Having_At_Symbol,
            Prefix_Suffix=form.Prefix_Suffix,
            SSLfinal_State=form.SSLfinal_State,
            Domain_registeration_length=form.Domain_registeration_length,
            Favicon=form.Favicon,
            HTTPS_token=form.HTTPS_token,
            Request_URL=form.Request_URL
        )

        phishing_df = phishing_data.get_input_data_frame()
        model_predictor = PhishingClassifier()
        value = model_predictor.predict(dataframe=phishing_df)[0]
        status = "Phishing Website" if value == -1 else "Legitimate Website"

        return templates.TemplateResponse(
            "phishing.html",
            {"request": request, "context": status},
        )

    except Exception as e:
        return {"status": False, "error": f"{e}"}

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)