from fastapi import FastAPI
import inngest.fast_api

app = FastAPI()
inngest.fast_api.serve(app, inngest_client, [grade_submission])