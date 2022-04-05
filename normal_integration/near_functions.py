# Toshiba Research Europe Ltd reserves all rights, as outlined in license.txt
import numpy as np
import os
import cv2
#import matplotlib.pyplot as plt


def calculate_V_L_a_fields(Lpos, Ldir, mu, X, Y, Z): 
    Nlight=Lpos.shape[0]
    Nmask=X.shape[0]
    assert(len(X.shape)==1)    

    L=np.zeros((Nmask,Nlight,3))
    A=np.zeros((Nmask,Nlight))
    norm_f=150*150   
    #norm_f=meanZ*meanZ
    
    V=np.zeros((Nmask,3))
    vm=np.sqrt(X*X + Y*Y +Z*Z)    
    #print('Z min',np.amin(Z))
   # print('vm min',np.amin(vm))
   
    V=-np.stack([X,Y,Z],axis=1)    
    vmm=np.transpose(np.tile(vm, (3,1)))    
    V/=vmm
   
    for img in range(Nlight):
        Lx = Lpos[img,0]-X
        Ly = Lpos[img,1]-Y
        Lz = Lpos[img,2]-Z
        LM2=Lx*Lx + Ly*Ly+ Lz*Lz             
        LM=np.sqrt(LM2)
        
        Lx/=LM
        Ly/=LM
        Lz/=LM
            
        L[:,img,0]=Lx
        L[:,img,1]=Ly
        L[:,img,2]=Lz
        
        LdirDot=-(Lx*Ldir[img,0]+Ly*Ldir[img,1]+Lz*Ldir[img,2])   
        LdirDot[LdirDot<1e-4]=1e-4
        mufactor=np.power(LdirDot,mu[img])  
        A[:,img]=mufactor*(norm_f/LM2)       
    #Am=np.max(A,axis=1,keepdims=True)
    #A/=Am
    return V,L,A
def u_f_v_f(mask_indx,width,height,f,x0,y0, mapx=None, mapy=None):  
    if  mapx is None:  
        u_f = (np.arange(width, dtype=np.float32) - x0)
        v_f = np.reshape((np.arange(height, dtype=np.float32) - y0), (height, 1)) # need a column vector
        u_f = np.ones((height, 1), dtype=np.float32) * u_f
        v_f = v_f * np.ones((1, width), dtype=np.float32)
    else:
        u_f=mapx.copy()-x0
        v_f=mapy.copy()-y0
    #u_f+=0.5
    #v_f+=0.5   
    um=np.amin(u_f)
    uM=np.amax(u_f)
    vm=np.amin(v_f)
    vM=np.amax(v_f)
    
    #print('u_f.shape ',u_f.shape)    
    print('image plane coord limits %.1f %.1f %.1f %.1f'%(um,uM,vm,vM))    
    #get vectors for domain values only
    u_f = np.reshape(u_f,(height*width,))
    u_f=u_f[mask_indx]*(1/f)    
    v_f = np.reshape(v_f,(height*width,))
    v_f=v_f[mask_indx]*(1/f)   
    return u_f,v_f
def initial_shape(u_f, v_f,mean_distance):
    Nmask=u_f.shape[0]    
    Z = mean_distance * np.ones((Nmask,), dtype=np.float32)
    N = np.zeros((Nmask,3))
    N[:,2]=-1   
    
    X = u_f * Z 
    Y = v_f * Z 
    return X,Y,Z,N
def normalsFromZ(z,Gu,Gv,u_f,v_f,f):
    
    zu=Gu.dot(z)
    zv=Gv.dot(z)
            
    zz=-(z/f + zu*u_f+ zv*v_f)
    
    nmag=np.sqrt(zu*zu+zv*zv+zz*zz)
    nmag[nmag==0]=1

    zu=zu/nmag
    zv=zv/nmag
    zz=zz/nmag

    N=np.vstack((zu,zv,zz))
    return N.transpose()
def normalsFromZortho(z,zu,zv):    
    zz=-1+0*zu
    
    nmag=np.sqrt(zu*zu+zv*zv+zz*zz)
    nmag[nmag==0]=1

    zu=zu/nmag
    zv=zv/nmag
    zz=zz/nmag

    N=np.vstack((zu,zv,zz))
    return N
# classic photometric stereo for lambertian reflectance
def lambertian(Iv,L):
    if(Iv.shape[1]==3) and ( Iv.shape[2]>0): #rgb
        #super simple rgb to gray
        Iv= np.mean(Iv, axis=1)
    Nmask=Iv.shape[0]
    numLights=Iv.shape[1]
    Nm=np.zeros((Nmask,3))  
    
    for p in range(Nmask):
        Ip=Iv[p,:].reshape(numLights, -1)
        #print("Ip is\n\n", Ip)
        ss=Ip<0.02
        #print(ss)
        Ip[ss==True]=0
        #print("Ip is\n\n", Ip)
        if (numLights-ss.sum())<3:
            Nm[p,2]=-1            
            continue
        Lp=L[p,:,:].reshape(numLights,3)          
        Lp[ss[:,0]==True,0]=0        
        temp=np.linalg.lstsq(Lp,Ip,rcond=None)[0]       
        Nm[[p],:]=temp.transpose()

    rho=np.linalg.norm(Nm,axis=1,keepdims=True)   
    Nm/=rho    
    
    return Nm,rho
