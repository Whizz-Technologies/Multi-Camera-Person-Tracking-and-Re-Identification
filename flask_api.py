from flask import Flask, request
import tensorflow as tf
from yolo_v4 import YOLO4
from run import main


gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

app = Flask(__name__)

app.secret_key = "lumos"

@app.route('/runpy', methods=['POST'])
def inp():
    path = request.form['path']
    main(YOLO4(),path)
    print('Function Executed')
    return path


if __name__ == '__main__':
    app.run(debug=True)