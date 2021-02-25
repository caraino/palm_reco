import parameter as pm
import numpy as np
import cv2
import math
import time
import serial
import pyttsx3
import os
import matplotlib.pyplot as plt
from GetRoi import hand_analysis



baudRate = 9600
ser = serial.Serial("COM3", baudRate, timeout=0.5)  # Arduino发送的延时要小于500ms
time.sleep(3)




def get_train_and_test_img_features():
    train_path = "./Palmprint/training/"
    test_path = "./Palmprint/testing/"
    train_dataset = []  # 存储训练集中每张图片的SIFT特征描述向量
    test_dataset = []  # 存储测试集中每张图片的SIFT特征描述向量

    train_img_list = os.listdir(train_path)
    test_img_list = os.listdir(test_path)

    for train_img in train_img_list:
        img = cv2.imread(train_path + train_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)  # step1.预处理图片，灰度均衡化
        kp_query, des_query = get_sift_features(equalize)  # step2.获取SIFT算法生成的关键点kp和描述符des(特征描述向量)
        train_dataset.append(des_query)
    for test_img in test_img_list:
        img = cv2.imread(test_path + test_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)
        kp_query, des_query = get_sift_features(equalize)
        test_dataset.append(des_query)
    return train_dataset, test_dataset


def get_sift_features(img, dect_type='sift'):
    if dect_type == 'sift':
        sift = cv2.xfeatures2d.SIFT_create()
    elif dect_type == 'surf':
        sift = cv2.xfeatures2d.SURF_create()
    kp, des = sift.detectAndCompute(img, None)  # kp为关键点，des为描述符
    return kp, des


def sift_detect_match_num(des_query, des_train, ratio=0.70):
    # step3.使用KNN计算查询图像与训练图像之间匹配的点数目,采用k(k=2)近邻匹配，最近的点距离与次近点距离之比小于阈值ratio就认为是成功匹配。
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_query, des_train, k=2)
    match_num = 0
    for first, second in matches:
        if first.distance < ratio * second.distance:
            match_num = match_num + 1
    return match_num


def get_one_palm_match_num(des_query, index, train_dataset, ratio=0.70):
    # 获取查询图像与训练图像中属于每一组的3张图像的匹配点的数量和
    match_num_sum = 0
    for i in range(index, index + 3):
        match_num_sum += sift_detect_match_num(des_query, train_dataset[i], ratio=ratio)
    return match_num_sum


def get_match_result(des_query, train_dataset, ratio=0.70):
    # step4.根据最大的匹配点数量和，确定查询图片的类别
    index = 0
    train_length = len(train_dataset)
    result = np.zeros(train_length // 3, dtype=np.int32)
    while index < train_length:
        result[index // 3] = get_one_palm_match_num(des_query, index, train_dataset, ratio=ratio)
        index += 3
    resultargmax = result.argmax()
    print("**********************", result)
    if result.max() < 50:
        resultargmax = -1
    return resultargmax


def predict(train_features, test_features, ratio=0.70):
    predict_true = 0
    for i, feature in enumerate(test_features):
        print('Processing image...')
        # 预测每张测试图片的类别
        category = get_match_result(feature, train_features, ratio=ratio)
        if category == i // 3:
            predict_true += 1
        print('Predict result:', category + 1)
    return predict_true / len(test_features), category + 1
#############################################################################################################3333    






def ToArduino( theta, distance):
    # 假设最右边为0度的位置时；
    theta = int(theta) 
    lengthTmp = int(distance)  # 规定只能 -20~20mm进行运动。
    lenghtTmp = min(max(0, lengthTmp), 40) # 调试的时候是20是为了安全，最后测试的时候可以最大为40.
    length = lenghtTmp 
    angle = chr(theta)  # 单位为角度
    length = chr(length)  # ASCII只有0-128
    # ***********************************要根据导轨的初始位置进行限位操作，不然会出现电机到顶的现象****************#
    ser.write(str.encode(chr(0)))
    ser.write(str.encode(chr(1)))  # 0X01是帧头。
    ser.write(str.encode(angle))  # 0~120对应-60~60
    # time.sleep(0.01) # 延时10ms再发送数据
    ser.write(str.encode(length))  #
    return


def fromArduino():
    count1 = 0
    count2 = 0
    while True:
        str1 = ser.read()
        a = ord(str1)
        if a < 42 and a > 22:  # 温度部分整数用0-42
            t1 = a
            count1 = 1

        if a > 100 and a != 250 and a != 165 and a != 186:
            t2 = a - 100
            # print("t2",t2)
            count2 = 1

        if count1 == 1 and count2 == 1:
            # print("**********")
            tmp = t2 / 100
            t = t1 + tmp
            if t > 35.0 and t < 42.0:
                print("当前人体温度：", t)
                return t



def talk(state):
    # msg = '世界那么大，我想去看看'
    # 变换声音（文字为英文或数字时才有多种声音）
    engine = pyttsx3.init()

# 调节语速
    rate = engine.getProperty('rate')
    engine.setProperty('rate', rate + 20)

    # 调节音量
    volume = engine.getProperty('volume')
    engine.setProperty('volume', volume - 0.25)
    voices = engine.getProperty('voices')
    if state == 1:
        msg = '请保持手臂三秒不动'
    if state == 2:
        msg = '检测成功，请通行'
  
    if state == 3:
        msg = '体温不符合，禁止通行'  # 身份与体温最好分开讲
    if state == 4:
        msg = "身份不匹配，禁止通行"

    for i in voices:
        
        engine.setProperty('voice', i.id)
        engine.say(msg)
    engine.runAndWait()
    return


def camera():
    test_theta = []
    test_distance = []
    n = 2       
    count = 0
    while True:
        test_theta = []
        test_distance = []
    
        

        mindisparity = 32
        SADWindowSize = 16
        ndisparities = 176

        P1 = 4 * 1 * SADWindowSize * SADWindowSize
        P2 = 32 * 1 * SADWindowSize * SADWindowSize
 
        # ****************参数设置************************#
        ret, imgL = pm.cap2.read()
        ret, imgR = pm.cap1.read()
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))


        grayImageL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
        grayImageR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
        #grayImageR = clahe.apply(grayImageR)


        palmL = pm.PalmDetect.detectMultiScale(grayImageL, 1.3, 5)
        palmR = pm.PalmDetect.detectMultiScale(grayImageR, 1.3, 5)

        rectifyImageL = cv2.remap(grayImageL, pm.mapLx, pm.mapLy, cv2.INTER_LINEAR)
        rectifyImageR = cv2.remap(grayImageR, pm.mapRx, pm.mapRy, cv2.INTER_LINEAR)

        sgbm = cv2.StereoSGBM_create(mindisparity, ndisparities, SADWindowSize)
        sgbm.setP1(P1)
        sgbm.setP2(P2)

        sgbm.setPreFilterCap(60)
        sgbm.setUniquenessRatio(30)
        sgbm.setSpeckleRange(2)
        sgbm.setSpeckleWindowSize(200)
        sgbm.setDisp12MaxDiff(1)
        disp = sgbm.compute(rectifyImageL, rectifyImageR)

        xyz = cv2.reprojectImageTo3D(disp, pm.Q, handleMissingValues=True)
        xyz = xyz * 16
        # print("1111")
        # theta, distance = self.Calculate(xyz,palmR,imgR)
        for (x, y, w, h) in palmR:
            cv2.rectangle(imgR, (x, y), (x + w, y + h), (0, 255, 0), 2)
            x = x + 0.5 * w
            y = y + 0.5 * h
            x = int(x)
            y = int(y)
            theta_tmp=math.atan2(xyz[y, x, 0]/(-xyz[y, x, 0]+80),1.0) # 半径为80mm?
            theta_tmp = theta_tmp*180.0/np.pi
            distance_tmp = xyz[y, x, 2]
            if  100 < xyz[y, x, 2] and xyz[y, x, 2] < 250:
                test_theta.append(theta_tmp)
                test_distance.append(distance_tmp)
                count += 1

            if  count==n  : 
                
                theta_diff = np.diff(test_theta)  
                theta_diff = abs(theta_diff)
                distance_diff = np.diff(test_distance)  
                distance_diff = abs(distance_diff)

                theta_bool = all([abs(theta1)<10 for theta1 in theta_diff]) # 所有都小于5才于真
                distance_bool = all([abs(dist)<50 for dist in distance_diff]) 
                if (theta_bool and distance_bool):
                    distance = xyz[y, x, 2]
                    distance = int(distance) # 
                    theta = int(60 + theta_tmp)  # -60~60到0~60

                    if distance - 145 > 0:
                        distance = distance - 145
                        distance = min(distance, 40)
                        print("dist: ", distance)
                        print("angle: ", theta_tmp )
                        return distance, theta
                    else:
                        distance = 0
                        return distance, theta
 
                    #语音输出，提醒不要移动
                    
                else:
                    test_theta = list(test_theta)
                    test_distance = list(test_distance)
                    test_theta.clear()
                    test_distance.clear()
                    count = -1 

        disp = disp.astype(np.float32) / 16.0
        disp8U = cv2.normalize(disp, disp, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp8U = cv2.medianBlur(disp8U, 9)

        #cv2.imshow("left", imgL)
        cv2.imshow("right", imgR)
        cv2.imshow("disparity", disp8U)

        if cv2.waitKey(1) == ord('q'):
            break


    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    




def main():


    distance, angle = camera()
    temperature = fromArduino()
    state = 1
    talk(state)
    print("!!!")
    #####################################################
    
   
    ToArduino(angle, distance)
    time.sleep(0.4)
    #temperature = fromArduino()
    #temperature = 36
    #print("当前人体温度：", temperature)
    # 加入身份检测代码
    time.sleep(4)
    ret, img = pm.cap1.read()
    
    
    cv2.imshow("*********", img)
    cv2.imwrite('palm1.jpg',img)
    hand_analysis('palm1.jpg','./Palmprint/testing/test.jpg')
    ##########################################################################3
    train_sift_features, test_sift_features = get_train_and_test_img_features()  # 存储每张图片的SIFT特征描述向量
    ratio = 0.65
    best_acc = 0
    best_ratio = 0
    ratio_list = []
    acc_list = []
    max_ratio = 0.65
    while ratio <= max_ratio:  # 循环测试具有最高准确率的ratio
        acc, flag_zy = predict(train_sift_features, test_sift_features, ratio)
        acc_list.append(acc)
        ratio_list.append(ratio)
        if acc > best_acc:
            best_acc = acc
            best_ratio = ratio
        ratio += 0.01
    if flag_zy == 0:
        state = 4 # 身份不匹配，禁止通行。
    if 35.0 < temperature < 37.2 and flag_zy > 0:
        state = 2
    if 37.2 < temperature < 40.0:
        state = 3
    talk(state)
   
    state = 0


if __name__ == '__main__':
    while 1:
        main()
