import os

from flask import Flask, render_template, request

from ocr_core import ocr_core
from image_pre_processing_v4 import pre_process
import timeit
UPLOAD_FOLDER = '/static/uploads/'
RESULT_FOLDER = '/static/generated_images/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home_page():
    return render_template('index.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_page():
    if request.method == 'POST':
        # check if the post request has the file part
    	if request.form['action'] == 'Submit':

    		if 'file' not in request.files:
    			return render_template('upload.html', msg='No file selected')
    		file = request.files['file']
			# if user does not select file, browser also
			# submit a empty part without filename
    		if file.filename == '':
    			return render_template('upload.html', msg='No file selected')

    		if file and allowed_file(file.filename):
				# os.remove("static/uploads/"+file.filename)
    			file.save(os.path.join(os.getcwd() + UPLOAD_FOLDER, file.filename))
    			start = timeit.default_timer()
    			f1,f2,f3,f4,f5=pre_process("static/uploads/"+file.filename,False)
    			stop = timeit.default_timer()
				# call the OCR function on it
				
    			if f1 and f2 and f3 and f4 and f5 :
    				extracted_text = ocr_core('static/generated_images/'+file.filename.split(".")[0]+'/cropped.png')
				# extract the text and display it
    				return render_template('upload.html',
									   msg='Successfully processed',
									   extracted_text=extracted_text.split("\n"),
									   img_src=UPLOAD_FOLDER + file.filename,
									   img_crop=RESULT_FOLDER+file.filename.split(".")[0]+'/cropped.png',
									   time = round(stop-start,2))
    			else:
    				return render_template('upload.html', msg='No file selected')
					
    	elif request.form['action'] == 'Run All':
    		path  = request.form['path']
    		start = timeit.default_timer()
    		tmp_path = path.replace('\\','/')
    		if not tmp_path.lower().endswith('/'):
    			tmp_path  = tmp_path+'/'
    		_path = tmp_path
    		files = os.listdir(_path)

    		for filename in files:
    			print(filename)
    			if filename.lower().endswith('.jpg') or filename.lower().endswith('.png'):
    				pre_process(_path+filename,True)
    		stop = timeit.default_timer()
    		return render_template('upload.html', msg1='Processed all files',time_all = round(stop-start,2))
    elif request.method == 'GET':
        return render_template('upload.html')

if __name__ == '__main__':
    app.run(port=5004)
