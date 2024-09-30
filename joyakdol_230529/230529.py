from fastapi import FastAPI, UploadFile, Form
import time
import uuid
import numpy as np
import cv2
from fastapi.responses import FileResponse # 사진을 바로 웹에서 보여주기 위해

app = FastAPI()  # 핵심 개체

# 이 아래부터 hello 함수 전까지는 전부 frontend 와의 연계를 위한.
from starlette.middleware.cors import CORSMiddleware

origins = [
    "http://localhost:5173",  # 또는 "http://127.0.0.1:5173". 나는 이거 안되더라.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins='*',  # * or origins need.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/hello")  # url /hello가 실행되었을때
def hello():
    return {"조약돌": "드디어 성공이다 !!!!!!"}  # 딕셔너리(json) 반환


@app.get("/beombeom1")
def beombeom1():
    return [{"tyrenol": 1}]


@app.get("/beombeom2")
def beombeom2():
    return [{"tyrenol": "1"}]


@app.get("/beombeom")
def beombeom():
    return [
        {
            "id": 1,
            "name": "Leanne Graham",
            "username": "Bret",
            "email": "Sincere@april.biz",
            "address": {
                "street": "Kulas Light",
                "suite": "Apt. 556",
                "city": "Gwenborough",
                "zipcode": "92998-3874",
                "geo": {
                    "lat": "-37.3159",
                    "lng": "81.1496"
                }
            },
            "phone": "1-770-736-8031 x56442",
            "website": "hildegard.org",
            "company": {
                "name": "Romaguera-Crona",
                "catchPhrase": "Multi-layered client-server neural-net",
                "bs": "harness real-time e-markets"
            }
        },
        {
            "id": 2,
            "name": "Ervin Howell",
            "username": "Antonette",
            "email": "Shanna@melissa.tv",
            "address": {
                "street": "Victor Plains",
                "suite": "Suite 879",
                "city": "Wisokyburgh",
                "zipcode": "90566-7771",
                "geo": {
                    "lat": "-43.9509",
                    "lng": "-34.4618"
                }
            },
            "phone": "010-692-6593 x09125",
            "website": "anastasia.net",
            "company": {
                "name": "Deckow-Crist",
                "catchPhrase": "Proactive didactic contingency",
                "bs": "synergize scalable supply-chains"
            }
        },
        {
            "id": 3,
            "name": "Clementine Bauch",
            "username": "Samantha",
            "email": "Nathan@yesenia.net",
            "address": {
                "street": "Douglas Extension",
                "suite": "Suite 847",
                "city": "McKenziehaven",
                "zipcode": "59590-4157",
                "geo": {
                    "lat": "-68.6102",
                    "lng": "-47.0653"
                }
            },
            "phone": "1-463-123-4447",
            "website": "ramiro.info",
            "company": {
                "name": "Romaguera-Jacobson",
                "catchPhrase": "Face to face bifurcated interface",
                "bs": "e-enable strategic applications"
            }
        },
        {
            "id": 4,
            "name": "Patricia Lebsack",
            "username": "Karianne",
            "email": "Julianne.OConner@kory.org",
            "address": {
                "street": "Hoeger Mall",
                "suite": "Apt. 692",
                "city": "South Elvis",
                "zipcode": "53919-4257",
                "geo": {
                    "lat": "29.4572",
                    "lng": "-164.2990"
                }
            },
            "phone": "493-170-9623 x156",
            "website": "kale.biz",
            "company": {
                "name": "Robel-Corkery",
                "catchPhrase": "Multi-tiered zero tolerance productivity",
                "bs": "transition cutting-edge web services"
            }
        },
        {
            "id": 5,
            "name": "Chelsey Dietrich",
            "username": "Kamren",
            "email": "Lucio_Hettinger@annie.ca",
            "address": {
                "street": "Skiles Walks",
                "suite": "Suite 351",
                "city": "Roscoeview",
                "zipcode": "33263",
                "geo": {
                    "lat": "-31.8129",
                    "lng": "62.5342"
                }
            },
            "phone": "(254)954-1289",
            "website": "demarco.info",
            "company": {
                "name": "Keebler LLC",
                "catchPhrase": "User-centric fault-tolerant solution",
                "bs": "revolutionize end-to-end systems"
            }
        },
        {
            "id": 6,
            "name": "Mrs. Dennis Schulist",
            "username": "Leopoldo_Corkery",
            "email": "Karley_Dach@jasper.info",
            "address": {
                "street": "Norberto Crossing",
                "suite": "Apt. 950",
                "city": "South Christy",
                "zipcode": "23505-1337",
                "geo": {
                    "lat": "-71.4197",
                    "lng": "71.7478"
                }
            },
            "phone": "1-477-935-8478 x6430",
            "website": "ola.org",
            "company": {
                "name": "Considine-Lockman",
                "catchPhrase": "Synchronised bottom-line interface",
                "bs": "e-enable innovative applications"
            }
        },
        {
            "id": 7,
            "name": "Kurtis Weissnat",
            "username": "Elwyn.Skiles",
            "email": "Telly.Hoeger@billy.biz",
            "address": {
                "street": "Rex Trail",
                "suite": "Suite 280",
                "city": "Howemouth",
                "zipcode": "58804-1099",
                "geo": {
                    "lat": "24.8918",
                    "lng": "21.8984"
                }
            },
            "phone": "210.067.6132",
            "website": "elvis.io",
            "company": {
                "name": "Johns Group",
                "catchPhrase": "Configurable multimedia task-force",
                "bs": "generate enterprise e-tailers"
            }
        },
        {
            "id": 8,
            "name": "Nicholas Runolfsdottir V",
            "username": "Maxime_Nienow",
            "email": "Sherwood@rosamond.me",
            "address": {
                "street": "Ellsworth Summit",
                "suite": "Suite 729",
                "city": "Aliyaview",
                "zipcode": "45169",
                "geo": {
                    "lat": "-14.3990",
                    "lng": "-120.7677"
                }
            },
            "phone": "586.493.6943 x140",
            "website": "jacynthe.com",
            "company": {
                "name": "Abernathy Group",
                "catchPhrase": "Implemented secondary concept",
                "bs": "e-enable extensible e-tailers"
            }
        },
        {
            "id": 9,
            "name": "Glenna Reichert",
            "username": "Delphine",
            "email": "Chaim_McDermott@dana.io",
            "address": {
                "street": "Dayna Park",
                "suite": "Suite 449",
                "city": "Bartholomebury",
                "zipcode": "76495-3109",
                "geo": {
                    "lat": "24.6463",
                    "lng": "-168.8889"
                }
            },
            "phone": "(775)976-6794 x41206",
            "website": "conrad.com",
            "company": {
                "name": "Yost and Sons",
                "catchPhrase": "Switchable contextually-based project",
                "bs": "aggregate real-time technologies"
            }
        },
        {
            "id": 10,
            "name": "Clementina DuBuque",
            "username": "Moriah.Stanton",
            "email": "Rey.Padberg@karina.biz",
            "address": {
                "street": "Kattie Turnpike",
                "suite": "Suite 198",
                "city": "Lebsackbury",
                "zipcode": "31428-2261",
                "geo": {
                    "lat": "-38.2386",
                    "lng": "57.2232"
                }
            },
            "phone": "024-648-3804",
            "website": "ambrose.net",
            "company": {
                "name": "Hoeger LLC",
                "catchPhrase": "Centralized empowering task-force",
                "bs": "target end-to-end models"
            }
        }
    ]


# 추가 부분 - 이미지 업로드를 위해 / chatgpt 참고
def generate_filename(user_id):
    timestamp = int(time.time() * 1000)  # 타임스탬프 (밀리초 단위)
    unique_id = str(uuid.uuid4().hex)  # UUID (32자리 16진수)
    filename = f"{user_id}_{timestamp}_{unique_id}.jpg"
    return filename


@app.post("/image_upload/")
async def image_upload(image: UploadFile, user_id: int = Form(0)):
    # image : ~~ 이런거는 매개변수의 타입을 정해주는 것임. 이 함수로 post를 할때, 데이터정보를 무조건 image 변수에
    # 담아서 저장해야함. user_id도 넣어서 보내주면 좋음. 만약 안보내준다면 Form(0)에 의해서 user_id = 0으로 설정한다.
    contents = await image.read() # image 변수에 데이터가 들어올때까지 기다린다.

    # 파일명 생성
    filename = generate_filename(user_id) #generate 함수를 이용해서 파일이름을 생성. 이 파일은 바로 아래에서 서버(혹은 로컬)의 디스크에 저장된다.
    # 파일 이름같은거 겹치면 큰일나니까 이런식으로 '고유한' 파일 이름을 불러주는 것임.

    # 이미지를 디스크에 저장
    # with open(f"/path/to/save/{filename}", "wb") as f:
    #     f.write(contents) # 디폴트 경로.
    with open(f"C:\\Users\\moo\\Desktop\\image\\{filename}.jpg", "wb") as f: # wb로 불러들이는 것은, UploadFile로 들어온 이미지 데이터가 binary로 저장되었기 때문.
        # 그래서 이거 읽어주려면, 다시 cv 같은거 이용해서 따로 사진을 보여주는 그런 과정이 필요하대.
        f.write(contents) # 위에서 선언한 contents 변수 안에 바이너리 파일이 들어있다.

# 이렇게 파일 저장까지 해줬으니, 바로 cv로 불러오는 것까지 해보자(근데 이럴꺼면 저장을 하는 이유가 있나..?)

    image_open = cv2.imread(f"C:\\Users\\moo\\Desktop\\image\\{filename}.jpg")

    cv2.imshow("Image", image_open)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return {"filename": filename} # 그냥 파일 이름 리턴해주도록 해준거.

# 이미지를 웹에 띄우는. cv 말고.
# @app.get("/image_show/{user_id}")
# async def get_image(user_id: str):
#     # 저장된 이미지에 접근하여 응답으로 보내줌
#
#
#     return FileResponse("/path/to/save/image.jpg")