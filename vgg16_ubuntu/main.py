from flask import Flask, render_template, request
import vgg16_inference
import glob

app = Flask(__name__, static_folder='/home/naoki/kk_proto/kk_proto/src_vgg16/test_img/')

# getのときの処理
@app.route('/', methods=['GET'])
def get():
	img_result = 'empty'
	return render_template('index.html', \
		title = 'AI検索プログラム', \
		message = '検索ワードを入力してください', \
		image = img_result)

# postのときの処理	
@app.route('/text', methods=['POST'])
def post():
	result = []
	file_path = []
	search_file = []
	name = request.form['name']
	'''
	①データベースorJSONファイルの検索処理を入れる
	②データベースorJSONファイルの検索の結果、
		検索ワードがデータベースにない場合:
			ダウンロード処理を実行し、その後、モデルの学習実行
		検索ワードがデータベースにある場合:
			ダウンロード処理を実行しない
	③推論を実行し検索ワードに一致するラベルだけ出力する
	'''
	files = glob.glob("/home/naoki/kk_proto/kk_proto/src_vgg16/test_img/*")
	for file in files:
		res = vgg16_inference.img_predict(file)
		result.append(res)
		file_path.append(file)

	loop_num = 0
	find_num = 0
	for res in result:
		if res == name:
			file_name = file_path[loop_num]
			file_name = file_name[file_name.rfind('/') + 1:]
			search_file.append(file_name)
			find_num += 1
		loop_num += 1
		#print(i)
	print(search_file)

	#print(find_num)
	if find_num == 0:
		img_result = 'empty'
		message_result = '見つかりませんでした'
	else:
		img_result = search_file[0]
		message_result = '検索結果は' + name + "です"

	return render_template('index.html', \
		title = 'AI検索プログラム', \
		message = message_result, \
		image = img_result)

if __name__ == '__main__':
	app.run(host='0.0.0.0', debug=True, port=5000)
