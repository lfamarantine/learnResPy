
# ----- fastapi
# benefits over flask api:
# - async (faster than through WSGI (flask))
# - built-in data validation
# - automatic docs generation
# - separation of server code from business logic

import uvicorn
from fastapi import FastAPI
app = FastAPI()
@app.get("/")
def home(name: str):
    return {"message": f"Hello! {name}"}
if __name__ == "__main__":
    uvicorn.run(app, host='127.0.0.1', port=8000, debug=True)



