from fastapi import FastAPI, UploadFile, Form
import time
import uuid
import numpy as np
from fastapi.responses import FileResponse # 사진을 바로 웹에서 보여주기 위해

# 다른 잡다한 모듈들
from module import beom3, beom
from medicine_list import med_list

# 이미지 머신러닝 함수
from joyakdol_230715.yolov5_master.joyakdol_yolo_230804 import check_medicine_photo


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

@app.get("/beombeom3")
def beombeom3():
    return beom3

# 왜인진 모르겠지만 모듈로 안되는 군 .. -> changing variable than ok !

@app.get("/beombeom")
def beombeom():
    return beom


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
#    filename = generate_filename(user_id) #generate 함수를 이용해서 파일이름을 생성. 이 파일은 바로 아래에서 서버(혹은 로컬)의 디스크에 저장된다.
    # 파일 이름같은거 겹치면 큰일나니까 이런식으로 '고유한' 파일 이름을 불러주는 것임.

    # 이미지를 디스크에 저장
    # with open(f"/path/to/save/{filename}", "wb") as f:
    #     f.write(contents) # 디폴트 경로.
    with open(f"/home/ubuntu/projects/myapi/image/{user_id}.jpg", "wb") as f:
        # 앞에서는 filename 썼는데, 그냥 check 용이하게 하기 위해서 user_id로 이름 바꾸어버림.

        # wb로 불러들이는 것은, UploadFile로 들어온 이미지 데이터가 binary로 저장되었기 때문.
        # 그래서 이거 읽어주려면, 다시 cv 같은거 이용해서 따로 사진을 보여주는 그런 과정이 필요하대.
        f.write(contents) # 위에서 선언한 contents 변수 안에 바이너리 파일이 들어있다.

# 이렇게 파일 저장까지 해줬으니, 바로 cv로 불러오는 것까지 해보자(근데 이럴꺼면 저장을 하는 이유가 있나..?)

    # image_open = cv2.imread(f"C:\\Users\\moo\\Desktop\\image\\{filename}.jpg")
    #
    # cv2.imshow("Image", image_open)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    return {"filename": user_id} # 그냥 파일 이름 리턴해주도록 해준거.

#이미지를 웹에 띄우는. cv 말고.
@app.get("/image_show/{user_id}")
async def get_image(user_id: str):
    # 저장된 이미지에 접근하여 응답으로 보내줌

    return FileResponse(f"/home/ubuntu/projects/myapi/image/{user_id}.jpg")


#image machine learning and show result
@app.get("/result/{user_id}")
async def ml_result(user_id : str):

    num = check_medicine_photo(user_id)

    if num is None:
        # num이 None인 경우 사용자에 대한 데이터가 존재하지 않음을 나타낼 수 있도록 처리
	# now don't need. already excluded.
        return {"message": "User data not found."}
    return [{"title" :  med_list[num], "corp":"종근당","ingredient":
    '''\nDL-메틸에페드린염산염(DL-Methylephedrine HCl) 12.5mg\n구아이페네신(Guaifenesin) 41.6mg\n덱스트로메토르판브롬화수소산염수화물(Dextromethorphan HBr Hydrate) 8mg\n슈도에페드린염산염(Pseudoephedrine HCl) 15mg\n아세트아미노펜(Acetaminophen) 200mg\n클로르페니라민말레산염(Chlorpheniramine Maleate) 1.25mg'''
    ,"effect":"""\n감기의 제증상(여러증상)(콧물, 코막힘, 재채기, 인후(목구멍)통, 기침, 가래, 오한 (춥고 떨리는 증상), 발열, 두통, 관절통, 근육통)의 완화""",
    "usage": """\n- 만 15세 이상 및 성인: 1일 3회, 1회 2캡슐 식후 30분에 복용
- 만 11세 이상-만 15세 미만: 1일 3회, 1회 1⅓캡슐 식후 30분에 복용
- 만 7세 초과-만 11세 미만: 1일 3회, 1회 1 캡슐 식후 30분에 복용
"""
,    "warning":
"""1. 경고
매일 세잔 이상 정기적으로 술을 마시는 사람이 이 약이나 다른 해열진통제를 복용해 야 할 경우 반드시 의사 또는 약사와 상의해야 한다. 이러한 사람이 이 약을 복용 하면 간 손상이 유발될 수 있다.

2. 다음과 같은 사람은 이 약을 복용하지 말 것.
1) 이 약 및 이 약의 구성성분에 대한 과민반응 및 그 병력이 있는 사람
2) 이 약 및 이 약의 구성성분, 다른 해열진통제, 감기약 복용 시 천식을 일으킨 적이 있는 사람
3) 만 3개월 미만의 영아 (갓난아기)
4) MAO억제제(항우울제, 항정신병제, 감정조절제, 항파킨슨제 등)를 복용하고 있거나 복용을 중단한 후 2주 이내의 사람

3. 이 약을 복용하는 동안 다음의 약을 복용하지 말 것.
진해(기침을 그치게 함)거담제(가래약), 다른 감기약, 해열진통제, 진정제, 항히스타민제를 함유하는 내복약(비염(코염)용 경구제, 멀미약, 알레르기용약)

4. 다음과 같은 사람은 이 약을 복용하기 전에 의사, 치과의사, 약사와 상의 할 것.
1) 수두 또는 인플루엔자에 감염되어 있거나 또는 의심되는 영아(갓난아기) 및 만 15세 미만의 어린이(구역이나 구토를 수반하는 행동의 변화가 있다면, 드물지만 심각한 질병인 레이증후군의 초기 증상일수 있으므로 의사와 상의할 것.)
2) 만 3개월 미만의 영아(갓난아기)에는 복용을 피하고 만 3개월 이상인 경우고 만 2세 미만의 영아(갓난아기), 유아는 의사의 진료를 받아야 하며, 꼭 필요한 경우가 아니면 이약을 복용시키지 않도록 한다. 만 2세 미만 영아(갓난아기), 유아에게 이 약을 투여할 경우 보호자에게 알리고 주의 깊게 모니터해야 한다.
3) 본인, 양친 또는 형제 등이 두드러기, 접촉성피부염, 기관지 천식, 알레르기성비염(코염), 편두통, 음식물알레르기 등을 일으키기 쉬운 체질을 갖고 있는 사람
4) 지금까지 약에 의해 알레르기 증상(예: 발열, 발진, 관절통, 천식, 가려움증 등)을 일으킨 적이 있는 사람
5) 간장질환, 신장(콩팥)질환, 심장질환, 갑상선질환, 당뇨병, 고혈압, 위십이지장궤양, 녹내장(예: 눈의 통증, 눈이 침침함 등), 배뇨(소변을 눔)곤란 등이 있는 사람, 고령자(노인), 몸이 약한 사람 또는 고열이 있는 사람
6) 속쓰림, 위부불쾌감, 위통과 같은 위장문제가 지속 혹은 재발되거나 궤양, 출혈문제를 가지고 있는 사람
7) 임부 또는 임신하고 있을 가능성이 있는 여성, 수유부
8) 의사 또는 치과의사의 치료를 받고 있는 사람 (당뇨약, 통풍약, 관절염약, 항응고제, 스테로이드제 등 다른 약물을 투여 받고 있는 사람)
9) 다음과 같은 기침이 있는 사람
흡연, 천식, 만성 기관지염, 폐기종, 과도한 가래가 동반되는 기침, 1주 이상 지속 또는 재발되는 기침, 만성 기침, 발열ㆍ발진이나 지속적인 두통이 동반되는 기침

5. 다음과 같은 경우 이 약의 복용을 즉각 중지하고 의사, 치과의사, 약사와 상의할 것. 상담 시 가능한 이 첨부문서를 소지할 것.
1) 이 약의 복용에 의해 다음의 증상이 나타난 경우
발진ㆍ발적(충혈되어 붉어짐), 가려움, 구역, 구토, 식욕부진, 변비, 부종(부기), 배뇨(소변을 눔)곤란, 목마름(지속적이거나 심한), 어지러움, 불안, 떨림, 불면
2) 이 약의 복용에 의해 드물게 아래의 중증(심한증상) 증상이 나타난 경우
① 쇽(아나필락시): 복용후 바로 두드러기, 부종(부기), 가슴답답함 등과 함께 안색이 창백하고, 손발이 차고, 식은땀, 숨쉬기 곤란함 등이 나타날 수 있다.
② 피부점막안증후군(스티븐스-존슨증후군), 중독성표피괴사용해(리엘증후군): 고열을 동반하고, 발진ㆍ발적(충혈되어 붉어짐), 화상과 같이 물집이 생기는 등의 심한 증상이 전신피부, 입이나 눈 점막에 나타날 수 있다.
③ 천식
④ 간기능장애: 전신의 나른함, 황달(피부 또는 눈의 흰자위가 황색을 띄게 됨)이 나타날 수 있다.
⑤ 간질성폐렴: 기침을 동반하고, 숨이 차고, 호흡곤란, 발열 등이 나타난다.
3) 5~6회 복용하여도 증상이 좋아지지 않을 경우

6. 기타 이 약의 복용시 주의할 사항
1) 정해진 용법ㆍ용량을 잘 지킬 것.
2) 장기간 계속 복용하지 말 것.
3) 어린이에게 복용시킬 경우에는 보호자의 지도감독 하에 복용시킬 것.
4) 복용시에는 음주하지 말 것.
5) 복용하는 동안 졸음이 오는 경우가 있으므로 자동차 운전 또는 기계류의 운전 조작을 피할 것.
6) 바르비탈계 약물, 삼환계 항우울제 및 알코올을 투여한 환자는 다량의 아세트아미노펜을 대사시키는 능력이 감소되어 아세트아미노펜의 혈장 반감기를 증가시킬 수 있다.

7. 저장상의 주의사항
1) 어린이의 손에 닿지 않는 장소에 보관할 것.
2) 직사광선을 피하고 될 수 있는 한 습기가 적은 서늘한 곳에 밀폐하여 보관 할 것.
3) 오용(잘못 사용)을 막고 품질의 보존을 위하여 다른 용기에 바꾸어 넣지 말 것.
"""}]

@app.get("/beombeom/{user_id}")
async def beom_result(user_id : str):
    return [{"title":"타이레놀","corp":"제조","ingredient":"성분","effect":"효능","usage":"용법"}]
