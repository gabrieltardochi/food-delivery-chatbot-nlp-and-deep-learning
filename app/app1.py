from flask import Flask, request
from flask import render_template
from flask import redirect

app = Flask(__name__)
tasks = ["oioi",'oii']

@app.route('/')
def tasks_list():
    return render_template('list.html', tasks=tasks)


@app.route('/task', methods=['POST'])
def add_task():
    global tasks
    content = request.form['content']
    if not content:
        return 'Error'
    tasks += [content]
    return redirect('/')


if __name__ == '__main__':
    app.run()