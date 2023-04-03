from aiohttp import web, WSMsgType
import asyncio
from subprocess import Popen, PIPE
import threading
from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

def enqueue_output(file, queue):
    for line in iter(file.readline, ''):
        queue.put(line)
    file.close()


def read_popen_pipes(p):

    with ThreadPoolExecutor(2) as pool:
        q_stdout, q_stderr = Queue(), Queue()
        pool.submit(enqueue_output, p.stdout, q_stdout)
        pool.submit(enqueue_output, p.stderr, q_stderr)
        while True:
            if p.poll() is not None and q_stdout.empty() and q_stderr.empty():
                break
            out_line = err_line = ''
            try:
                out_line = q_stdout.get_nowait()
            except Empty:
                pass
            try:
                err_line = q_stderr.get_nowait()
            except Empty:
                pass
            yield (out_line, err_line)
            
async def index(request):
    # return web.Response(text=f'Redirect Server, nothing here to see')
    return web.Response(text="""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Terminal Viewer</title>
    <script>
      var ws = new WebSocket("ws://" + window.location.host + "/ws");
      ws.onmessage = function(event) {
        var terminal = document.getElementById("terminal");
        terminal.textContent += event.data;
        terminal.scrollTop = terminal.scrollHeight;
      };
    </script>
  </head>
  <body>
    <pre id="terminal"></pre>
  </body>
</html>
""", content_type='text/html')

async def send_to_clients(post):
    global ws_talkers
    for ws_talker in ws_talkers:
        if not ws_talker is None:
            try:
                await ws_talker.send_str(post)
            except Exception as e:
                await ws_talker.close()
                ws_talkers.remove(ws_talker)

async def handle(request):
    global ws_talkers
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    print('Websocket connection opended!!!')
    ws_talkers.append(ws)
    
    async for msg in ws:
        await ws.send_str("Done!")
    print('Websocket connection closed!!!')
    return ws

async def main():
    with Popen(['python', 'demo_server.py'], stdout=PIPE, stderr=PIPE, text=True) as main_p:
        for out_line, err_line in read_popen_pipes(main_p):
            # Do stuff with each line, e.g.:
            all_line = out_line + err_line
            # print(all_line, end = '')
            if len(all_line) > 0:
                await send_to_clients(all_line)
            
def start_main():
    loop = asyncio.new_event_loop()  # <-- create new loop in this thread here
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

if __name__ == '__main__':
    app = web.Application()
    app.add_routes([web.get('/', index), web.get('/ws', handle)])

    ws_talkers = []
    
    threading.Thread(target=start_main, daemon=True).start()
    
    web.run_app(app, port=8081)