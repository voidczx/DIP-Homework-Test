# _*_ coding utf-8 _*_
# Author：94342
# Time：  2020/10/621:27
# File：  New11.py
# Engine：PyCharm

from flask import Flask

app = Flask(__name__)

@app.route('/')
def Home():
    return '<h1>Hello World</h1>'


if __name__ == '__main__':
    app.run()