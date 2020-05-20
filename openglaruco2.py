import numpy as np
import cv2
import cv2.aruco as aruco
import math
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from PIL import Image

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX 
g_angle = 0
pi = 3.14
t = 0

INVERSE_MATRIX = np.array([[ 1.0, 1.0, 1.0, 1.0],
                           [1.0,1.0,1.0,1.0],
                           [-1.0,-1.0,-1.0,-1.0],
                           [ 1.0, 1.0, 1.0, 1.0]])

with np.load('AR.npz') as X:
    camera_matrix, dist_coeff, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
def init_gl():       
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) 
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glShadeModel(GL_SMOOTH)
        glMatrixMode(GL_MODELVIEW)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_TEXTURE_2D)
        material_Ka= [0.11, 0.06, 0.11, 1.0]
        material_Kd= [0.43, 0.47, 0.54, 1.0]
        material_Ks= [0.33, 0.33, 0.52, 1.0]
        material_Ke= [0.1, 0.0, 0.1, 1.0]
        material_Se= 10
        glMaterialfv(GL_FRONT, GL_AMBIENT	, material_Ka)
        glMaterialfv(GL_FRONT, GL_DIFFUSE	, material_Kd)
        glMaterialfv(GL_FRONT, GL_SPECULAR	, material_Ks)
        glMaterialfv(GL_FRONT, GL_EMISSION	, material_Ke)
        glMaterialf (GL_FRONT, GL_SHININESS	, material_Se)
        
def detect_markers(img):
    aruco_list=[]
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)  
    parameters =  aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img,aruco_dict, parameters=parameters)  
    if ids is not None: 
        rvect, tvect, _ = aruco.estimatePoseSingleMarkers(corners, 50, camera_matrix, dist_coeff) 
        (rvect-tvect).any()
        for i in range(rvect.shape[0]):
            aruco.drawAxis(img, camera_matrix, dist_coeff, rvect[i, :, :], tvect[i, :, :], 30)
            aruco.drawDetectedMarkers(img, corners,ids)
            aruco_id=ids[i][0]
            aruco_centre=(((corners[i][0][0][0]+corners[i][0][2][0])/2,(corners[i][0][0][1]+corners[i][0][2][1])/2))
            rvec,tvec=rvect[i][0],tvect[i][0]
            aruco_list.append([aruco_id,aruco_centre,rvec,tvec])
    else:  
        cv2.putText(img, "Not found", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)  
    return aruco_list
def drawGL():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    ret, frame = cap.read()
    ar_list = []
    if ret == True:
        ar_list = detect_markers(frame)
        cv2.imshow('frame',frame)
        glutPostRedisplay()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        for i in ar_list:
            if i[0] == 1:
                animate(frame,ar_list,i[0])    
        key = cv2.waitKey(1)
        if key == 27:       
            print('esc to exit')  
            cap.release()
            cv2.destroyAllWindows()
    glutSwapBuffers()

def animate(img, ar_list, ar_id):
        for x in ar_list:
                if ar_id == x[0]:
                        centre, rvec, tvec = x[1], x[2], x[3]
        rmtx = cv2.Rodrigues(rvec)[0]
        view_matrix = np.array([[rmtx[0][0],rmtx[0][1],rmtx[0][2],(centre[0]-320)/125],
                                [rmtx[1][0] ,rmtx[1][1],rmtx[1][2],0],
                                [rmtx[2][0],rmtx[2][1],rmtx[2][2],tvec[2]/80],
                                [0.0      ,0.0       ,0.0       ,1.0    ]])   
        view_matrix = view_matrix * INVERSE_MATRIX 
        view_matrix = np.transpose(view_matrix)
        glLoadMatrixf(view_matrix)
        glScalef(1.0,1.6,0.8)
        glutSolidCube(0.5)
        leftleg(view_matrix)
        rightleg(view_matrix)
        leftarm(view_matrix)
        rightarm(view_matrix)
        head(view_matrix)
        
def leftleg(M):
    global pi
    LM = np.zeros(16)
    LeftT1 = [[1,0,0,0],
                    [0,1,0,0],
					[0,0,1,0],
					[0,-0.55,0,1 ]]
    Angle = (math.cos(4*pi*t)*pi)/4.0
    LeftT2 = [[math.cos(Angle),math.sin(Angle),0,0],
						[-math.sin(Angle),math.cos(Angle),0,0],
						[0,0,1,0],
						[0,0,0,1]]
    LeftT3 = [[1,0,0,0],
				[0,1,0,0],
				[0,0,1,0],
				[0,0,0.1,1]]
    MR1 = np.dot(LeftT2,M)
    MR2 = np.dot(LeftT1,MR1)
    MR = np.dot(LeftT3,MR2)
    glLoadMatrixf(MR)
    glScalef(0.3,1.2,0.3)
    glutSolidCube(0.5)
def rightleg(M):
    global pi
    RM = np.zeros(16)
    RightT1 = [[1,0,0,0],
                    [0,1,0,0],
					[0,0,1,0],
					[0,-0.55,0,1 ]]
    Angle = (math.cos(4*pi*t)*pi)/4.0
    RightT2 = [[math.cos(-Angle),math.sin(-Angle),0,0],
						[-math.sin(-Angle),math.cos(-Angle),0,0],
						[0,0,1,0],
						[0,0,0,1]]
    RightT3 = [[1,0,0,0],
				[0,1,0,0],
				[0,0,1,0],
				[0,0,-0.1,1]]
    RMR1 = np.dot(RightT2,M)
    RMR2 = np.dot(RightT1,RMR1)
    RMR = np.dot(RightT3,RMR2)
    glLoadMatrixf(RMR)
    glScalef(0.3,1.2,0.3)
    glutSolidCube(0.5)
def leftarm(M):
    global pi
    LAM = np.zeros(16)
    LAT1 = [[1,0,0,0],
                    [0,1,0,0],
					[0,0,1,0],
					[0,-0.2,0,1 ]]
    Angle = (math.cos(4*pi*t)*pi)/3.0
    LAT2 = [[math.cos(-Angle),math.sin(-Angle),0,0],
						[-math.sin(-Angle),math.cos(-Angle),0,0],
						[0,0,1,0],
						[0,0,0,1]]
    LAT3 = [[1,0,0,0],
				[0,1,0,0],
				[0,0,1,0],
				[0,0,0.3,1]]
    LAMR1 = np.dot(LAT2,M)
    LAMR2 = np.dot(LAT1,LAMR1)
    LAMR = np.dot(LAT3,LAMR2)
    glLoadMatrixf(LAMR)
    glScalef(0.3,1.2,0.3)
    glutSolidCube(0.5)
def rightarm(M):
    global pi
    RAM = np.zeros(16)
    RAT1 = [[1,0,0,0],
                    [0,1,0,0],
					[0,0,1,0],
					[0,-0.2,0,1 ]]
    Angle = (math.cos(4*pi*t)*pi)/3.0
    RAT2 = [[math.cos(Angle),math.sin(Angle),0,0],
						[-math.sin(Angle),math.cos(Angle),0,0],
						[0,0,1,0],
						[0,0,0,1]]
    RAT3 = [[1,0,0,0],
				[0,1,0,0],
				[0,0,1,0],
				[0,0,-0.3,1]]
    RAMR1 = np.dot(RAT2,M)
    RAMR2 = np.dot(RAT1,RAMR1)
    RAMR = np.dot(RAT3,RAMR2)
    glLoadMatrixf(RAMR)
    glScalef(0.3,1.2,0.3)
    glutSolidCube(0.5)
def head(M):
    RH = np.zeros(16)
    RHT1 = [[1,0,0,0],
            [0,1,0,0],
			[0,0,1,0],
			[0,0.55,0,1 ]]
    RH = np.dot(RHT1,M)
    glLoadMatrixf(RH)
    glutSolidSphere(0.2,20,20)
def timer(n):
    glutPostRedisplay()
    global t
    t = t+0.02
    if (t>=1):
        t = 0
    glutTimerFunc(16,timer,0)
def resize(w,h):
    ratio = 1.0* w / h
    glMatrixMode(GL_PROJECTION)
    glViewport(0,0,w,h)
    gluPerspective(45, ratio, 0.1, 100.0)

def main():
    glutInit()
    glutInitWindowSize(640, 480)
    glutInitWindowPosition(625, 100)
    glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE)
    window_id = glutCreateWindow("OpenGL")
    init_gl()
    glutDisplayFunc(drawGL)
    
    glutIdleFunc(drawGL)
    glutReshapeFunc(resize)
    glutTimerFunc(16,timer,0)
    glutMainLoop()
if __name__ == "__main__":
        main()

