from flask import Flask
app = Flask(__name__)
@app.route('/')
def index():
    return 'Hello Liangruanruan , this my first wed file!'
if __name__ == '__main__':
    app.debug = True # 设置调试模式，生产模式的时候要关掉debug
    app.run()