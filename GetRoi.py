import urllib
import base64
import json
import time
from PIL import Image, ImageDraw
import numpy as np
import cv2
from matplotlib import pyplot as plt
import urllib.request


# In[ ]:


def show(img):
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])


# In[ ]:


def get_token():
    '''
    client_id = 'jz7AqXksOE6FS88peATfmqXR'
    client_secret = 'bkHFsWp4q2CzAsWGCC5F7Mgmx8hWp12z'
    host = 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=' + client_id + '&client_secret=' + client_secret
    request = urllib.request.Request(host)
    request.add_header('Content-Type', 'application/json; charset=UTF-8')
    response = urllib.request.urlopen(request)
    token_content = response.read()
    if token_content:
        token_info = json.loads(token_content)
        token_key = token_info['access_token']
    return token_key
    '''
    return '24.0bb4105225144cff150a13ce0d26b1da.2592000.1599287998.282335-21277253'


# In[ ]:


def hand_analysis(originfilename,resultfilename):
    
    access_token = get_token()
    request_url = "https://aip.baidubce.com/rest/2.0/image-classify/v1/hand_analysis"



    #二进制方式打开图片文件
    with open(originfilename, 'rb') as f:
        img = base64.b64encode(f.read())
    '''
    image = cv2.imencode('.jpg', originfilename)[1]
    img = str(base64.b64encode(image))[2:-1]
    '''


    params = dict()

    params['image'] = img

    params = urllib.parse.urlencode(params).encode("utf-8")

    request_url = request_url + "?access_token=" + access_token

    request = urllib.request.Request(url=request_url, data=params)

    request.add_header('Content-Type', 'application/x-www-form-urlencoded')

    response = urllib.request.urlopen(request)

    content = response.read()
    

    if content:
        content=content.decode('utf-8')
        data = json.loads(content)
        try:
            x1 = int((data['hand_info'][0]['hand_parts']['5']['x']+data['hand_info'][0]['hand_parts']['9']['x'])/2)
            y1 = int((data['hand_info'][0]['hand_parts']['5']['y']+data['hand_info'][0]['hand_parts']['9']['y'])/2)
            x2 = int((data['hand_info'][0]['hand_parts']['13']['x']+data['hand_info'][0]['hand_parts']['17']['x'])/2)
            y2 = int((data['hand_info'][0]['hand_parts']['13']['y']+data['hand_info'][0]['hand_parts']['17']['y'])/2)
            get_roi(originfilename,resultfilename,np.array([x1,y1]),np.array([x2,y2]))
        except:
            print('No Hands Detected!')
    else:
        print('No Contents Received!')


# In[ ]:


def get_roi(originfilename,resultfilename,v1,v2):
    
    #print(v1,v2)
    v1[1] = v1[1]+80
    v2[1] = v2[1]+80
    
    img_original = cv2.imread(originfilename,0)
    #img_original = originfilename
    h, w = img_original.shape
    img = np.zeros((h+160,w), np.uint8)
    img[80:-80,:] = img_original
    
    #'''
    #不显示图片结果请反注释
    plt.subplot(131)
    show(img)
    plt.plot(v1[0], v1[1],'rx')
    plt.plot(v2[0], v2[1],'bx')
    #'''
    
    theta = np.arctan2((v2-v1)[1], (v2-v1)[0])*180/np.pi
    R = cv2.getRotationMatrix2D(tuple(v2),theta,1)
    img_r = cv2.warpAffine(img,R,(w,h))
    v1 = (R[:,:2] @ v1 + R[:,-1]).astype(np.int)
    v2 = (R[:,:2] @ v2 + R[:,-1]).astype(np.int)
    ux = v1[0]
    uy = v1[1] + (v2-v1)[0]//3
    lx = v2[0]
    ly = v2[1] + 4*(v2-v1)[0]//3
    
    #'''
    #不显示图片结果请反注释
    print(ux,uy,lx,ly)
    plt.subplot(132)
    plt.plot(v1[0], v1[1],'rx')
    plt.plot(v2[0], v2[1],'bx')
    show(img_r)
    
    
    img_c = cv2.cvtColor(img_r, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_c, (lx,ly),(ux,uy),(0,255,0),2)
    plt.subplot(133)
    show(img_c)

    plt.tight_layout()
    plt.show()
    #'''
    
    img_r = img_r[uy:ly,ux:lx]
    cv2.imwrite(resultfilename,img_r)


# In[ ]:


#hand_analysis('   .jpg','  .jpg')
#使用方法，import本文件后hand_analysis(要截取roi图片名字,要保存roi图片名字)
#只能用于左手，手掌不能偏移竖直方向太多，尽量保证大部份手在图片内。


# In[ ]:




