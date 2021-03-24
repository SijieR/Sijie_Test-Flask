# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
app = FastAPI()
class Item(BaseModel):
    a: int = None
    b: int = None
    @app.get('/test/a={a}/b={b}')
    def calculate(a: int=None, b: int=None):
        c = a + b
        res = {"res":c}
        return res
        @app.post('/test')
        def calculate(request_data: Item):
            a = request_data.a
            b = request_data.b
            c = a + b
            res = {"res":c}
            return res
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app=app,  host="localhost",   port=8000,       workers=1)




