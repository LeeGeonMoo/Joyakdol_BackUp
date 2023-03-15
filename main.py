from fastapi import FastAPI

app = FastAPI() # 핵심 개체

# 이 아래부터 hello 함수 전까지는 전부 frontend 와의 연계를 위한.
from starlette.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",    # 또는 "http://127.0.0.1:5173". 나는 이거 안되더라.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello") #url /hello가 실행되었을때
def hello():
    return {"message" : "안녕 파이보"} # 딕셔너리(json) 반환