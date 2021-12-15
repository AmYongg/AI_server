from flask import Flask, render_template, Response
from Router import get_class_info, AI_model_run

app = Flask(__name__)

@app.route('/')
def index():
    return 'index page'

@app.route('/AMP/Router/get_DB/<int:cls_id>')
def get_class(cls_id):
    amp = get_class_info.get_cls_info(cls_id)
    return render_template('.view/test.html', data=amp)

@app.route('/AMP/run')
def AMP_run():
    """Video streaming home page."""
    #app.run(AI_model_run)
    return render_template('view_video_feed.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        AI_model_run.gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

if __name__ == '__main__':
    #app.debug = True
    app.run(host='0.0.0.0', debug=True, threaded=True)

