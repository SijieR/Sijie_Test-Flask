from fastapi import FastAPI
import uvicorn


app = FastAPI()


@app.get('/test/a={a}/b={b}')
def calculate(a: int = None, b: int = None):
    c = a + b
    res = {"res": c}
    return res


if __name__ == '__main__':
    uvicorn.run(app=app,
                host="127.0.0.1",
                port=8080,
                workers=1)