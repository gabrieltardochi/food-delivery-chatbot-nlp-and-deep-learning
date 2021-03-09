from flask import Flask, request, Markup, render_template, redirect

app = Flask(__name__)
msgs = []

@app.route('/')
def msgs_list():
    global msgs
    msgs = []
    return render_template('list.html', msgs=msgs)


@app.route('/chatting', methods=['POST'])
def add_msg():
    global msgs
    content = request.form['content']
    if not content:
        return render_template('list.html', msgs=msgs)
    #  msgs += [Markup(f'<p class="p-custom">{content}</p>')]
    msgs += [Markup(f'<p class="other">{content}</p>')]
    return render_template('list.html', msgs=msgs)

if __name__ == '__main__':
    app.run()