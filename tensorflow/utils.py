import numpy as np
def calculateScales(img):
    """
        进行尺度的计算，返回一个包含多尺度信息的数组
    :param img:
    :return:
    """
    copy_image = img.copy()
    pr_scale = 1.0
    h, w, _ = copy_image.shape

    if min(h,w) > 500:
        pr_scale = 500.0 / min(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)
    elif max(h,w) < 500:
        pr_scale = 500.0 / max(h,w)
        w = int(w*pr_scale)
        h = int(h*pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min(h,w)
    while minl >= 12:
        scales.append(pr_scale*pow(factor,factor_count))
        minl *= factor
        factor_count += 1
    return scales


def detect_face_12net(cls_prob,roi,out_side,scale,width,height,threshold):
    """
        对Pnet处理过的数据进行解码优化，返回的是候选框信息
    :param cls_prob: 是否包含人脸的概率
    :param roi: 候选框的信息
    :param out_side:确保不同尺度下的候选框具有相同的大小
    :param scale:
    :param width:
    :param height:
    :param threshold:
    :return:
    """
    cls_prob = np.swapaxes(cls_prob,0,1)
    roi = np.swapaxes(roi,0,2)

    strides = 0
    if out_side != 1:
        strides = float(2*out_side-1) / (out_side-1)
    (x,y) = np.where(cls_prob>=threshold)

    boundingbox = np.array([x,y]).T
    #尺寸为12*12，这是在对Pnet的输出数据进行处理，因此此时会用到12，12
    bb1 = np.fix((strides*(boundingbox)+0) * scale)
    bb2 = np.fix((strides*(boundingbox)+11) * scale)

    boundingbox = np.concatenate((bb1,bb2),axis=1)

    dx1 = roi[0][x,y]
    dx2 = roi[1][x,y]
    dx3 = roi[2][x,y]
    dx4 = roi[3][x,y]
    score = np.array([cls_prob[x,y]]).T
    offset = np.array([dx1,dx2,dx3,dx4]).T
    #同理，此时是对Pnet的输出数据进行处理
    boundingbox = boundingbox + offset * 12.0 *scale
    rectangles = np.concatenate((boundingbox,score),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1, y1 = int(max(0,rectangles[i][0])), int(max(0,rectangles[i][1]))
        x2, y2 = int(min(width,rectangles[i][2])), int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2 > x1 and y2 > y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3)

def rect2square(rectangles):
    """
        将矩阵调整为正方形
    :param rectangles:
    :return:
    """
    #首先获取宽高，然后获取宽高中的最大值，
    w = rectangles[:,2] - rectangles[:,0]
    h = rectangles[:,3] - rectangles[:,1]
    l = np.maximum(w,h).T
    #调整左上角坐标，确保矩阵中心不变
    rectangles[:,0] = rectangles[:,0] + w * 0.5 - l * 0.5
    rectangles[:,1] = rectangles[:,1] + h * 0.5 - l * 0.5
    #左上角坐标加上宽高即为右下角坐标
    rectangles[:,2:4] = rectangles[:,0:2] + np.repeat([l],2,axis=0).T
    return rectangles


def NMS(rectangles,threshold):
    """
        进行非极大值抑制抑制，除去冗余框
    :param rectangles:
    :param threshold:
    :return:
    """
    if  len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    scores = boxes[:,4]
    area = np.multiply(x2-x1+1,y2-y1+1)
    I = np.array(scores.argsort())
    pick = []
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]],x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]],y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]],x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]],y2[I[0:-1]])
        w, h = np.maximum(0.0,xx2-xx1+1), np.maximum(0.0,yy2-yy1+1)
        inter = w * h
        iou = inter / (area[I[-1]]+area[I[0:-1]]-inter)
        pick.append(I[-1])
        #更新I，只保留iou小于阈值的部分框
        I = I[np.where(iou<=threshold)[0]]
    result_rectangles = boxes[pick].tolist()
    return result_rectangles


def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    """
    对Rnet处理后的数据进行优化，与filter_face_12net功能相似
    :param cls_prob:
    :param roi:
    :param rectangles:
    :param width:
    :param height:
    :param threshold:
    :return:
    """
    prob = cls_prob[:,1]
    pick = np.where(prob>=threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick,0]
    y1 = rectangles[pick,1]
    x2 = rectangles[pick,2]
    y2 = rectangles[pick,3]
    score = np.array([prob[pick]]).T
    #这是候选框的bbox信息
    dx1 = roi[pick,0]
    dx2 = roi[pick,1]
    dx3 = roi[pick,2]
    dx4 = roi[pick,3]

    h, w = y2 - y1, x2 - x1
    x1 = np.array([(x1+dx1*w)[0]]).T
    y1 = np.array([(y1+dx2*h)[0]]).T
    x2 = np.array([(x2+dx3*w)[0]]).T
    y2 = np.array([(y2+dx4*h)[0]]).T

    rectangles = np.concatenate((x1,y1,x2,y2,score),axis=1)
    rectangles = rect2square(rectangles)
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0,rectangles[i][0]))
        y1 = int(max(0,rectangles[i][1]))
        x2 = int(min(width,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    return NMS(pick,0.3)


def filter_face_48net(cls_prob, roi, pts,rectangles, width, height, threshold):
    """
        这是对Onet处理后的数据进行优化，与上类似
    :param cls_prob:
    :param roi:
    :param pts:
    :param rectangles:
    :param width:
    :param height:
    :param threshold:
    :return:
    """
    prob = cls_prob[:, 1]
    pick = np.where(prob >= threshold)
    rectangles = np.array(rectangles)

    x1 = rectangles[pick, 0]
    y1 = rectangles[pick, 1]
    x2 = rectangles[pick, 2]
    y2 = rectangles[pick, 3]

    sc = np.array([prob[pick]]).T

    dx1 = roi[pick, 0]
    dx2 = roi[pick, 1]
    dx3 = roi[pick, 2]
    dx4 = roi[pick, 3]

    w = x2 - x1
    h = y2 - y1
    #将各个特征点还原到对应尺寸上
    pts0 = np.array([(w * pts[pick, 0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick, 5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick, 1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick, 6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick, 2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick, 7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick, 3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick, 8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick, 4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick, 9] + y1)[0]]).T

    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dx2 * h)[0]]).T
    x2 = np.array([(x2 + dx3 * w)[0]]).T
    y2 = np.array([(y2 + dx4 * h)[0]]).T

    rectangles = np.concatenate((x1, y1, x2, y2, sc, pts0, pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9),axis=1)

    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        if x2 > x1 and y2 > y1:
            pick.append([x1, y1, x2, y2, rectangles[i][4],
                         rectangles[i][5], rectangles[i][6], rectangles[i][7], rectangles[i][8], rectangles[i][9],
                         rectangles[i][10], rectangles[i][11], rectangles[i][12], rectangles[i][13], rectangles[i][14]])
    return NMS(pick, 0.3)
