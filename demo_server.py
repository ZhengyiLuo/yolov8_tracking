#!/usr/bin/env python3
import glob
import os
import sys
import pdb
import os.path as osp
sys.path.append(os.getcwd())

import cv2
import joblib
import numpy as np
import time

import tensorflow as tf
import tensorflow_hub as hub


import asyncio
import cv2
import numpy as np
import threading
from scipy.spatial.transform import Rotation as sRot

import time
import torch
from collections import deque
from datetime import datetime
from torchvision import transforms as T
import poseviz
import time
from zen_tracker import run, parse_opt
from mdm_talker import MDMTalker

from aiohttp import web
import aiohttp
import aiohttp_jinja2
import jinja2
import json
import scipy.interpolate as interpolate
import subprocess
from io import StringIO

STANDING_POSE = np.array([[[-0.1443, -0.9426, -0.2548],
         [-0.2070, -0.8571, -0.2571],
         [-0.0800, -0.8503, -0.2675],
         [-0.1555, -1.0663, -0.3057],
         [-0.2639, -0.5003, -0.2846],
         [-0.0345, -0.4931, -0.3108],
         [-0.1587, -1.2094, -0.2755],
         [-0.2534, -0.1022, -0.3361],
         [-0.0699, -0.1012, -0.3517],
         [-0.1548, -1.2679, -0.2675],
         [-0.2959, -0.0627, -0.2105],
         [-0.0213, -0.0424, -0.2277],
         [-0.1408, -1.4894, -0.2892],
         [-0.2271, -1.3865, -0.2622],
         [-0.0715, -1.3832, -0.2977],
         [-0.1428, -1.5753, -0.2303],
         [-0.3643, -1.3792, -0.2646],
         [ 0.0509, -1.3730, -0.3271],
         [-0.3861, -1.1423, -0.3032],
         [ 0.0634, -1.1300, -0.3714],
         [-0.4086, -0.9130, -0.2000],
         [ 0.1203, -0.8943, -0.3002],
         [-0.4000, -0.8282, -0.1817],
         [ 0.1207, -0.8087, -0.2787]]]).repeat(5, axis = 0)

def fps_20_to_30(mdm_jts):
    jts = []
    N = mdm_jts.shape[0]
    for i in range(24):
        int_x = mdm_jts[:, i, 0]
        int_y = mdm_jts[:, i, 1]
        int_z = mdm_jts[:, i, 2]
        x = np.arange(0, N)
        f_x = interpolate.interp1d(x, int_x)
        f_y = interpolate.interp1d(x, int_y)
        f_z = interpolate.interp1d(x, int_z)
        
        new_x = f_x(np.linspace(0, N-1, int(N * 1.5)))
        new_y = f_y(np.linspace(0, N-1, int(N * 1.5)))
        new_z = f_z(np.linspace(0, N-1, int(N * 1.5)))
        jts.append(np.stack([new_x, new_y, new_z], axis = 1))
    jts = np.stack(jts, axis = 1)
    return jts

def xyxy2xywh(bbox):
    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path

async def mjpghandler(request):
    global tracking_res
    interval = 1
    print("Redirect to MJPEG local host")
    # Without the Content-Type, most (all?) browsers will not render
    # partially downloaded content. Note, the response type is
    # StreamResponse not Response.
    resp = web.StreamResponse()
    # The StreamResponse is a FSM. Enter it with a call to prepare.

    resp.content_type = ('multipart/x-mixed-replace; '
                         'boundary=--jpegboundary')
    await resp.prepare(request)
    while True:
        if "img_show" in tracking_res:
            try:
                raw_image = tracking_res['img_show']
                _, image_jpg = cv2.imencode('.jpg', raw_image)
                image_jpg = image_jpg.tobytes()

                await resp.write(bytes('--jpegboundary\r\n'
                                        'Content-Type: image/jpeg\r\n'
                                        'Content-Length: {}\r\n\r\n'.format(len(image_jpg)), 'utf-8') + image_jpg + b'\r\n')
                await asyncio.sleep(.01)

            except Exception as e:
                # So you can observe on disconnects and such.
                print("Exception: ", repr(e), e)
                raise
        else:
            time.sleep(1)
    else:
        return web.Response(text="Hello, world")
    

def start_pose_estimate():
    print("string pose estimate with Metrabs!!")
    global pose_mat, trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, tracking_res, images_acc, mdm_talker, tracking_res
    offset = np.zeros((5, 1))
    
    ## debug 

    from scipy.spatial.transform import Rotation as sRot
    global_transform = sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv().as_matrix()
    transform = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()

    t_s = time.time()
    print('### Run Model...')
    
    # model = tf.saved_model.load(download_model('metrabs_mob3l_y4'))
    # model = tf.saved_model.load(download_model('metrabs_eff2s_y4'))
    model = hub.load('https://bit.ly/metrabs_l') # or _s

    skeleton = 'smpl_24'
    # viz = poseviz.PoseViz(joint_names, joint_edges)
    print("==================================> Metrabs model loaded <==================================")
    
    with torch.no_grad():
        while True:
            if tracking_res['mode']  == "video":
                if 'img' in tracking_res and "detections" in tracking_res and tracking_res['new_img']:
                    tracking_res['new_img'] = False
                    
                    tracking_boxes = tracking_res['detections']
                    frame = tracking_res['img']
                    bbox = tracking_boxes[:, :4]
                
                    # pred = model.detect_poses(frame, skeleton=skeleton, default_fov_degrees=55, detector_threshold=0.5, num_aug=5)
                    pred = model.estimate_poses(frame, tf.constant(bbox, dtype=tf.float32), skeleton=skeleton, default_fov_degrees=55, num_aug=1)
                    
                    dt = time.time() - t_s
                    print(f'\r {1/dt:.2f} fps', end='')
                    # camera = poseviz.Camera.from_fov(55, frame.shape[:2])
                    # viz.update(frame, pred['boxes'], pred['poses3d'], camera)
                    pred_j3d = pred['poses3d'].numpy()
                    num_ppl = min(pred_j3d.shape[0], 5)
                    
                    j3d_curr = pred_j3d[:num_ppl]/1000
                    if num_ppl < 5:
                        j3d[num_ppl:, 0, 0] = np.arange(5 - num_ppl) + 1
                        
                    j2d =  pred['poses2d'].numpy()
                    t_s = time.time()
                    
                    if reset_offset:
                        offset[:num_ppl] = - offset_height - j3d_curr[:num_ppl, [0], 1]
                        reset_offset = False
                    
                    j3d_curr[:offset.shape[0], ..., 1] += offset[:num_ppl]
                    
                    j3d = j3d.copy() # Trying to handle race condition
                    j3d[:num_ppl] = j3d_curr
                        
                    tracking_res['j2d'] = j2d
            else:
                time.sleep(1)
                
             

def tracking_from_tracker():
    global opt
    run( return_dict = tracking_res, **vars(opt))
    
async def websocket_handler(request):
    print('Websocket connection starting')
    global pose_mat, trans, dt, sim_talker, ws_talkers
    sim_talker = aiohttp.web.WebSocketResponse()
    ws_talkers.append(sim_talker)
    await sim_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in sim_talker:
        if msg.type == aiohttp.WSMsgType.TEXT:
            if msg.data == "get_pose":
                await sim_talker.send_json({
                    "pose_mat": pose_mat.tolist(),
                    "trans": trans.tolist(),
                    "dt": dt,
                })

    print('Websocket connection closed')
    return sim_talker

def write_frames_to_video(frames, out_file_name = "output.mp4", frame_rate = 30, add_text = None, text_color = (255, 255, 255)):
    print(f"######################## Writing number of frames {len(frames)} ########################")
    if len(frames) == 0:
        return 
    y_shape, x_shape, _ = frames[0].shape
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'FMP4'), frame_rate, (x_shape, y_shape))
    transform_dtype = False
    transform_256 = False

    if frames[0].dtype != np.uint8:
        transform_dtype = True
    if np.max(frames[0]) < 1:
        transform_256 = True

    for i in range(len(frames)):
        curr_frame = frames[i]

        if transform_256:
            curr_frame = curr_frame * 256
        if transform_dtype:
            curr_frame = curr_frame.astype(np.uint8)
        if not add_text is None:
            cv2.putText(curr_frame, add_text , (0,  20), 3, 1, text_color)

        out.write(curr_frame)
    out.release()


async def pose_getter(request):
    # query env configurations
    global pose_mat, trans, dt, j3d, tracking_res, ticker, reset_offset, reset_buffer, mdm_motions, cycle_motion
    curr_paths = {}
    
    
    if tracking_res['mode']  == "video":
        json_resp = {
            "j3d": j3d.tolist(),
            "dt": dt,
        }
    else:
        if reset_offset:
            offset = - offset_height - mdm_motions[0, 0, 1]
            mdm_motions[..., 1] += offset
            reset_offset = False
            
        if reset_buffer:
            if buffer > 0:
                mdm_motions = np.concatenate([np.repeat(mdm_motions[0:1], buffer, axis = 0), mdm_motions])
            else:
                mdm_motions = mdm_motions[-buffer:]
                
            reset_buffer = False
        
        if cycle_motion:
            j3d_curr = mdm_motions[ticker % len(mdm_motions)]
        else:
            j3d_curr = mdm_motions[min(len(mdm_motions)-1, ticker)]
            
        j3d[0] = j3d_curr
        json_resp = {
            "j3d": j3d.tolist(),
            "dt": dt,
        }
        ticker += 1
        
    return web.json_response(json_resp)

def generate_text(prompts):
    global offset_height, mdm_talker, buffer, mdm_motions, ticker
    
    prompts = prompts.split("\n")
    num_prompt = len(prompts)
    gen_mdm_motions, abs_path = mdm_talker.generate_text(prompts)
    mat = sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix()
    gen_mdm_motions = np.matmul(gen_mdm_motions, mat.dot(mat))
    
    offset = - offset_height - gen_mdm_motions[:, 0:1, 0:1, 1]
    
    gen_mdm_motions[..., 1] += offset
    gen_mdm_motions[..., [0, 2]] -= gen_mdm_motions[..., 0:1, 0:1, [0, 2]]
    
    
    curr_motions = []
    for idx in range(num_prompt):
        curr_motion = gen_mdm_motions[idx]
        print("prev",  curr_motion.shape)
        curr_motion = fps_20_to_30(curr_motion)
        if idx == 0:
            curr_motion = np.concatenate([np.repeat(curr_motion[0:1], buffer, axis = 0), curr_motion])
        else:
            curr_motion = np.concatenate([np.repeat(curr_motion[0:1], 15, axis = 0), curr_motion])
            curr_motion[..., [0, 2]] += x_offset
        
        x_offset = curr_motion[-1:, 0:1, [0, 2]]
        curr_motions.append(curr_motion)
    
    mdm_motions = np.concatenate(curr_motions, axis = 0)
    path = osp.join(abs_path, "sample00_rep00.mp4")
    ticker = 0

async def send_to_clients(post):
    global ws_talkers
    for ws_talker in ws_talkers:
        if not ws_talker is None:
            try:
                print(f"Sending to client: {post}")
                await ws_talker.send_str(post)
            except Exception as e:
                ws_talker.close()
                ws_talkers.remove(ws_talker)

async def talk_server(request):
    global ws_talkers, tracking_res, mdm_motions, to_metrabs, offset_height, reset_offset, buffer, reset_buffer, cycle_motion
    post = await request.text()
    json_post = json.loads(post)
    try:
        if json_post['action'] == "set_mode":
            print(f"Setting mode to {json_post['query']['mode']}")
            tracking_res['mode']  = json_post['query']['mode']
            return web.Response(text=f'Setting mode to {tracking_res["mode"]}')
        elif json_post['action'] == "set_prompt":
            if tracking_res['mode']  == "language":
                prompt = json_post['query']['prompt']
                threading.Thread(target=generate_text, args=(prompt,), daemon=True).start()
                return web.Response(text=f'MDM generating for prompt {prompt}, this might take a while...')
            else:
                print("Not in language mode!!!")
                return web.Response(text='Error! Not in language mode!!!')
        elif json_post['action'] == "set_offset":
            offset_height = float(json_post['query']['offset'])
            print("setting offset to ", offset_height)
            reset_offset = True
            return web.Response(text=f'Setting offset to {offset_height}')
        
        elif json_post['action'] == "set_cycle_motion":
            cycle_motion = bool(json_post['query']['cycle_motion'])
            print("Cycle motion set to ", cycle_motion)
            return web.Response(text=f'Setting cycle_motion to {cycle_motion}')
        elif json_post['action'] == "set_buffer":
            buffer = float(json_post['query']['buffer'])
            reset_buffer = True
            print("setting buffer to ", buffer)
            return web.Response(text=f'Setting language buffer to {buffer}')
        elif json_post['action'] == "start_record":
            await send_to_clients(post)
            subprocess.Popen(["simplescreenrecorder", "--start-recording"])
            return web.Response(text=f'Recording started')
        elif json_post['action'] == "end_record":
            await send_to_clients(post)
            return web.Response(text=f'Recording ended')
        elif json_post['action'] == "reset":
            await send_to_clients(post)
            return web.Response(text=f'Reset server')
        elif json_post['action'] == "set_default_pose":
            mdm_motions[:] = STANDING_POSE[:1].copy()
            
            return web.Response(text=f'Reset to default pose')
            

    except Exception as e:
        print(e)
        import ipdb; ipdb.set_trace()
        return web.Response(text=f'Error!')
        
            
async def cmd_websocket_handler(request):
    print('Websocket connection starting')
    global reset_offset, trans, offset_height, recording, images_acc, ws_talkers
    ws_talker = aiohttp.web.WebSocketResponse()
    ws_talkers.append(ws_talker)
    await ws_talker.prepare(request)
    print('Websocket connection ready')

    async for msg in ws_talker:
        #       print(msg)
        if msg.type == aiohttp.WSMsgType.TEXT:
            print("\n" + msg.data)
            if msg.data.startswith("r:"):
                splits = msg.data.split(":")
                if len(splits) > 1:
                    offset_height = float(splits[-1])
                reset_offset = True
            elif msg.data.startswith("s"):
                recording = True
                tracking_res['recording'] =  True
                print(f"----------------> recording: {recording}")
                # if recording:
                    # pass
                if recording and not sim_talker is None:
                    await sim_talker.send_json({"action": "start_record"})
            elif msg.data.startswith("e"):
                recording = False
                tracking_res['recording'] =  False
                print(f"----------------> recording: {recording}")
                if not recording and not sim_talker is None:
                    await sim_talker.send_json({"action": "end_record"})

            elif msg.data.startswith("w"):
                curr_date_time = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
                out_file_name = f"output/hybrik_{curr_date_time}.mp4"
                print(f"----------------> writing video: {out_file_name}")
                write_frames_to_video(tracking_res['images_acc'], out_file_name = out_file_name)
                tracking_res['images_acc'] = deque(maxlen = 24000)
                tracking_res['images_acc_show'] = deque(maxlen = 24000)
                
            elif msg.data.startswith("t:"):
                recording = False
                splits = msg.data.split(":")
                if len(splits) > 1:
                    document_name = splits[-1]
                
            elif msg.data.startswith("get_pose"):
                await sim_talker.send_json({
                    "j3d": j3d.tolist(),
                    "dt": dt,
                })

            await ws_talker.send_str("Done!")

    print('Websocket connection closed')
    return ws_talker


# def start_pose_estimate():
#     loop = asyncio.new_event_loop()  # <-- create new loop in this thread here
#     asyncio.set_event_loop(loop)
#     loop.run_until_complete(main())


@aiohttp_jinja2.template('src/html/server.html')
def main(request):
    return {'name': 'Andrew'}

if __name__ == "__main__":
    print("Running PHC Demo")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
        
    opt = parse_opt() # Tracker
    ####### Get standing default ######
    
    # j3d = joblib.load("")
    
    tracking_res = {}
    bbox, pose_mat, j3d, j2d, trans, dt, ws_talkers, reset_offset, offset_height, images_acc, recording, sim_talker, num_ppl = np.zeros([5, 4]), np.zeros([24, 3, 3]), np.zeros([5, 24, 3]), None, np.zeros([3]), 1 / 10, [], True, 0.92, deque(maxlen = 24000), False, None, 0
    cycle_motion, tracking_res['mode'], mdm_motions, ticker, to_metrabs, buffer, reset_buffer = True, "video", np.zeros([120, 24, 3]), 0, sRot.from_euler('xyz', np.array([-np.pi / 2, 0, 0]), degrees=False).as_matrix(), 120, False
    
    mdm_talker = MDMTalker()
    
    j3d = STANDING_POSE.copy()
    mdm_motions[:] = STANDING_POSE[:1].copy()
    
    frame = None
    superfast = True
    
    tracking_res['recording'] = recording
    tracking_res['images_acc'] = deque(maxlen = 24000)
    # main()
    app = web.Application(client_max_size=1024**2)
    app.router.add_route('GET', '/ws', websocket_handler)
    app.router.add_route('GET', '/ws_talk', cmd_websocket_handler)
    app.router.add_route('GET', '/get_pose', pose_getter)
    # app.add_routes([web.get('/video.mjpeg', mjpghandler)])
    app.add_routes([web.get('/', main)])
    # app.add_routes([web.post('/talk_client', talk_client)])
    app.add_routes([web.post('/talk_server', talk_server)])
    
    threading.Thread(target=tracking_from_tracker, daemon=True).start()
    threading.Thread(target=start_pose_estimate, daemon=True).start()
    
    # tracking_from_tracker()
    
    app.router.add_static('/node_modules', 'node_modules', name='node_modules')
    app.router.add_static('/src', 'src', name='src')
    
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(str(".")))

    
    print("=================================================================")
    print("r: reset offset (use r:0.91), s: start recording, e: end recording, w: write video")
    print("=================================================================")
    web.run_app(app, port=8080)

