import os
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

import matplotlib.pyplot as plt
import pandas as pd

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def update(self):
        ####
        person_height = 165
        frame_offset = 20
        max_diff_angle = 15
        max_diff_distance = 10
        N_angle = 23
        N_distance = 20
        #
        frames = []
        ground_points = []
        head_points = []
        final_result = []
        final_angles = {'Frame': []}
        final_min_angles = {'Frame': []}
        final_max_angles = {'Frame': []}
        final_distances = {'Frame': []}
        final_min_distances = {'Frame': []}
        final_max_distances = {'Frame': []}
        #
        for i in range(1, N_angle + 1):
            final_angles['Angle_'+str(i)] = []
            final_min_angles['Angle_' + str(i)] = []
            final_max_angles['Angle_' + str(i)] = []
        for i in range(1, N_distance+1):
            final_distances['Distance_'+str(i)] = []
            final_min_distances['Distance_'+str(i)] = []
            final_max_distances['Distance_'+str(i)] = []
        #
        frame = 0
        min_angle = 180
        max_angle = 0
        min_distance = person_height + 100
        max_distance = 0
        #####
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitelyd
        while True:
            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                return
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4
                #pred = hm_data.cpu().data.numpy()

                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0,136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0,26)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area)

                _result = []
                for k in range(len(scores)):
                    _result.append(
                        {
                            'keypoints':preds_img[k],
                            'kp_score':preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx':ids[k],
                            'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                        }
                    )

                result = {
                    'imgname': im_name,
                    'result': _result
                }


                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                final_result.append(result)
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, self.opt)
                    #####
                    frame += 1
                    if frame <= frame_offset:
                        ground_point, head_point = self.calc_bound_points(result, vis_thres=0.4)
                        if ground_point is not None:
                            ground_points.append(ground_point)
                            x_point = [x for x, _ in ground_points]
                            y_point = [y for _, y in ground_points]
                            ground_point = (int(np.average(x_point)), int(np.average(y_point)))
                        if head_point is not None:
                            head_points.append(head_point)
                            x_point = [x for x, _ in head_points]
                            y_point = [y for _, y in head_points]
                            head_point = (int(np.average(x_point)), int(np.average(y_point)))
                        if ground_point is not None and head_point is not None:
                            dist_height = np.linalg.norm(np.array(head_point) - np.array(ground_point))
                            height_ratio = person_height / (dist_height + 1e-6)
                        else:
                            height_ratio = 0

                    distances = self.calc_distances(result, ground_point, head_point,
                                                    height_ratio, vis_thres=0.4)
                    angles = self.calc_angles(result, vis_thres=0.4)
                    frames.append(frame)
                    final_angles['Frame'].append(frame)
                    final_min_angles['Frame'].append(frame)
                    final_max_angles['Frame'].append(frame)
                    final_distances['Frame'].append(frame)
                    final_min_distances['Frame'].append(frame)
                    final_max_distances['Frame'].append(frame)
                    ##
                    for angle_name, angle in angles.items():
                        angle = int(angle)
                        if angle < 0 and frame > frame_offset:
                            angle = final_angles[angle_name][frame-2]
                        ##

                        final_angles[angle_name].append(angle)
                        ##
                        if frame <= frame_offset:
                            if angle >= 0 and angle < min_angle:
                                final_min_angles[angle_name].append(angle)
                            else:
                                final_min_angles[angle_name].append(min_angle)
                            if angle >= 0 and angle > max_angle:
                                final_max_angles[angle_name].append(angle)
                            else:
                                final_max_angles[angle_name].append(max_angle)
                        else:
                            previous_min_angle = final_min_angles[angle_name][frame-2]
                            previous_max_angle = final_max_angles[angle_name][frame-2]
                            diff_angle = abs(final_angles[angle_name][frame-1] - final_angles[angle_name][frame-2])
                            if angle >=0 and angle < previous_min_angle and diff_angle < max_diff_angle:
                                final_min_angles[angle_name].append(angle)
                            else:
                                final_min_angles[angle_name].append(previous_min_angle)
                            if angle >=0 and angle > previous_max_angle and diff_angle < max_diff_angle:
                                final_max_angles[angle_name].append(angle)
                            else:
                                final_max_angles[angle_name].append(previous_max_angle)
                        ##
                        plt.figure()
                        plt.plot(frames[frame_offset+1:], final_angles[angle_name][frame_offset+1:])
                        plt.plot(frames[frame_offset+1:], final_min_angles[angle_name][frame_offset+1:], linestyle='--', dashes=(5, 3))
                        plt.plot(frames[frame_offset+1:], final_max_angles[angle_name][frame_offset+1:], linestyle='--', dashes=(5, 3))
                        plt.xlabel('Frames')
                        plt.ylabel('Angle (degree)')
                        plt.title(angle_name)
                        plt.grid(True)
                        plt.savefig(os.path.join(self.opt.outputpath_plot, angle_name+".jpg"))
                        plt.close()
                    ##
                    for distance_name, distance in distances.items():
                        distance = round(distance, 2)
                        if distance < 0 and frame > frame_offset:
                            distance = final_distances[distance_name][frame-2]
                        ##
                        final_distances[distance_name].append(distance)
                        ##
                        if frame <= frame_offset:
                            if distance >= 0 and distance < min_distance:
                                final_min_distances[distance_name].append(distance)
                            else:
                                final_min_distances[distance_name].append(min_distance)
                            if distance >= 0 and distance > max_distance:
                                final_max_distances[distance_name].append(distance)
                            else:
                                final_max_distances[distance_name].append(max_distance)
                        else:
                            previous_min_distance = final_min_distances[distance_name][frame-2]
                            previous_max_distance = final_max_distances[distance_name][frame-2]
                            diff_distance = abs(final_distances[distance_name][frame - 1] -
                                                final_distances[distance_name][frame - 2])
                            if distance_name is 'Distance_10' or distance_name is 'Distance_11':
                                diff_distance *= 100
                            if distance >=0 and distance < previous_min_distance and diff_distance < max_diff_distance:
                                final_min_distances[distance_name].append(distance)
                            else:
                                final_min_distances[distance_name].append(previous_min_distance)
                            if distance >=0 and distance > previous_max_distance and diff_distance < max_diff_distance:
                                final_max_distances[distance_name].append(distance)
                            else:
                                final_max_distances[distance_name].append(previous_max_distance)
                        ##
                        plt.figure()
                        plt.plot(frames[frame_offset+1:], final_distances[distance_name][frame_offset+1:])
                        plt.plot(frames[frame_offset+1:], final_min_distances[distance_name][frame_offset+1:], linestyle='--', dashes=(5, 3))
                        plt.plot(frames[frame_offset+1:], final_max_distances[distance_name][frame_offset+1:], linestyle='--', dashes=(5, 3))
                        plt.xlabel('Frames')
                        plt.ylabel('Distance (cm)')
                        plt.title(distance_name)
                        plt.grid(True)
                        plt.savefig(os.path.join(self.opt.outputpath_plot, distance_name+".jpg"))
                        plt.close()
                    ##
                    df_angle = pd.DataFrame.from_dict(final_angles)
                    df_min_angle = pd.DataFrame.from_dict(final_min_angles)
                    df_max_angle = pd.DataFrame.from_dict(final_max_angles)
                    with pd.ExcelWriter(os.path.join(self.opt.outputpath_plot,"Angles.xlsx")) as writer:
                        df_angle.to_excel(writer, sheet_name='Angles', index=False)
                        df_min_angle.to_excel(writer, sheet_name='Min_Angles', index=False)
                        df_max_angle.to_excel(writer, sheet_name='Max_Angles', index=False)
                    ##
                    df_distance = pd.DataFrame.from_dict(final_distances)
                    df_min_distance = pd.DataFrame.from_dict(final_min_distances)
                    df_max_distance = pd.DataFrame.from_dict(final_max_distances)
                    with pd.ExcelWriter(os.path.join(self.opt.outputpath_plot,"Distances.xlsx")) as writer:
                        df_distance.to_excel(writer, sheet_name='Distances', index=False)
                        df_min_distance.to_excel(writer, sheet_name='Min_Distances', index=False)
                        df_max_distance.to_excel(writer, sheet_name='Max_Distances', index=False)
                    #########
                    self.write_image(img, im_name, stream=stream if self.save_video else None, frame=frame)


    def write_image(self, img, im_name, stream=None, frame=1):
        img = cv2.putText(img, f'frame: {frame}', (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo", img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'


    def calc_angles(self, im_res, vis_thres=0.4):
        def find_angle(p1, p2, p3):
            a = np.array(p1)
            b = np.array(p2)
            c = np.array(p3)
            ba = a - b
            bc = c - b
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)+1e-9)
            angle = np.arccos(cosine_angle) * (180 / np.pi)
            return angle

        kp_num = len(im_res['result'][0]['keypoints'])
        if kp_num == 26:
            angle_vertices = {
                              'Angle_1':  (17, 18, 19), 'Angle_2':  (0, 18, 19),
                              'Angle_3':  (18, 5, 7),   'Angle_4':  (18, 6, 8),
                              'Angle_5':  (5, 7, 9),    'Angle_6':  (6, 8, 10),
                              'Angle_7':  (10, 18, 9),  'Angle_8':  (19, 11, 13),
                              'Angle_9':  (19, 12, 14), 'Angle_10': (12, 19, 11),
                              'Angle_11': (11, 13, 15), 'Angle_12': (12, 14, 16),
                              'Angle_13': (16, 19, 15), 'Angle_14': (20, 24, 15),
                              'Angle_15': (21, 25, 16), 'Angle_16': (5, 11, 13),
                              'Angle_17': (6, 12, 14),  'Angle_18': (11, 5, 7),
                              'Angle_19': (12, 6, 8),   'Angle_20': (0, 18, 5),
                              'Angle_21': (0, 18, 6),   'Angle_22': (18, 19, 24),
                              'Angle_23': (18, 19, 25),
                              }

        else:
            raise NotImplementedError

        for human in im_res['result']:
            part_angle = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']

            # Valid keypoints
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= vis_thres:
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                part_angle[n] = (cor_x, cor_y)

            # Calc angles
            angles = {}
            for angle_name, (start_p, center_p, end_p) in angle_vertices.items():
                if start_p in part_angle and end_p in part_angle and center_p in part_angle:
                    start_xy = part_angle[start_p]
                    center_xy = part_angle[center_p]
                    end_xy = part_angle[end_p]
                    angle = find_angle(start_xy, center_xy, end_xy)
                    angles[angle_name] = angle
                else:
                    angles[angle_name] = -10

        return angles


    def calc_bound_points(self, im_res, vis_thres=0.4):
        kp_num = len(im_res['result'][0]['keypoints'])
        if kp_num == 26:
            ground_vertices = [24, 25, 20, 21]
            head_vertices = [17]
        else:
            raise NotImplementedError
        for human in im_res['result']:
            ground_points = {}
            head_points = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']

            # Valid keypoints
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= vis_thres:
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                if n in head_vertices:
                    head_points[n] = (cor_x, cor_y)
                elif n in ground_vertices:
                    ground_points[n] = (cor_x, cor_y)

            # Calc bound points
            if len(ground_points) == 0:
                ground_point = None
            else:
                x_points = [x for x,_ in ground_points.values()]
                y_points = [y for _,y in ground_points.values()]
                ground_point = (np.average(x_points), np.average(y_points))

            if len(head_points) == 0:
                head_point = None
            else:
                x_points = [x for x,_ in head_points.values()]
                y_points = [y for _,y in head_points.values()]
                head_point = (np.average(x_points), np.average(y_points))

            return ground_point, head_point


    def calc_distances(self, im_res, ground_point, head_point, height_ratio, vis_thres=0.4):
        if ground_point is None:
            ground_point = (-10, -10)
        if head_point is None:
            head_point = (-10, -10)

        def find_distance(p1, p2):
            a = np.array(p1)
            b = np.array(p2)
            ba = a - b
            distance = np.linalg.norm(ba)
            return distance

        kp_num = len(im_res['result'][0]['keypoints'])
        if kp_num == 26:
            distance_vertices = {
                              'Distance_1':  (20, 26),   'Distance_2':  (21, 26),
                              'Distance_3':  (24, 26),   'Distance_4':  (25, 26),
                              'Distance_5':  (20, 25),   'Distance_6':  (21, 24),
                              'Distance_7':  (24, 25),   'Distance_8':  (24, 11),
                              'Distance_9':  (25, 12),   'Distance_10': (17, 26),
                              'Distance_11': (17, 26),   'Distance_12': (9, 20),
                              'Distance_13': (10, 21),   'Distance_14': (9, 0),
                              'Distance_15': (10, 0),    'Distance_16': (9, 10),
                              'Distance_17': (7, 3),     'Distance_18': (8, 4),
                              'Distance_19': (26, 17),   'Distance_20': (17, 27),
                              }

        else:
            raise NotImplementedError

        for human in im_res['result']:
            part_distance = {}
            kp_preds = human['keypoints']
            kp_scores = human['kp_score']

            # Valid keypoints
            for n in range(kp_scores.shape[0]):
                if kp_scores[n] <= vis_thres:
                    continue
                cor_x, cor_y = int(kp_preds[n, 0]), int(kp_preds[n, 1])
                part_distance[n] = (cor_x, cor_y)
            part_distance[26] = ground_point
            part_distance[27] = head_point

            # Calc distances
            distances = {}
            for distance_name, (start_p, end_p) in distance_vertices.items():
                if start_p in part_distance and end_p in part_distance:
                    start_xy = part_distance[start_p]
                    end_xy = part_distance[end_p]
                    distance = find_distance(start_xy, end_xy)
                    if end_p == 26:
                        start_y = part_distance[start_p][1]
                        end_y = part_distance[end_p][1]
                        distance = abs(end_y - start_y)
                    distances[distance_name] = distance * height_ratio
                else:
                    distances[distance_name] = -10
            ##
            distances['Distance_10'] = (2*distances['Distance_5']) / (distances['Distance_10']+1e-6)
            distances['Distance_11'] = (2*distances['Distance_6']) / (distances['Distance_11']+1e-6)

        return distances





