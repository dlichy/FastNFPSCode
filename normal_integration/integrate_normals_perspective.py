# Toshiba Research Europe Ltd reserves all rights, as outlined in license.txt
# rougly following the method of :
#"Edge-Preserving Integration of a Normal Field: Weighted Least Squares and L1 Approaches", 
#Quéau and Durou, SSVM2015
import numpy as np
from scipy import sparse
from scipy.sparse import csc_matrix
#from scipy.sparse.linalg import spsolve, cg
from sksparse.cholmod import cholesky
#from matplotlib import pyplot as plt
import time

def  integrate_normals_perspective(N, Z0, f,u_f, v_f,mean_distance, itter_l1, Gx,Gy, II,JJ ):
    #Numerically integrate a normal map taking perspective projection into account and get Z    
    lambda_reg_z=1e-6#TODO verify 
    #lambda_reg_n=1e-6
    alpha = 0.5   
    Nmask = N.shape[0]  
    ones_v=np.ones((Nmask,))
    zero_v=np.zeros((Nmask,))   
    #u_f, v_f=nf.u_f_v_f(mask,f,cc[0],cc[1])     
    #LZ0m=np.zeros((Nmask,),float) #log depth regulariser
    LZ0m=np.log(Z0/mean_distance)
    #LZ0m=np.log(Z0)
    #LZ0m+=0.01
    W0m=ones_v #weight-1 for now    
    
    nx=N[:,0] 
    ny=N[:,1]     
    nz=N[:,2]    
   
    nzf=+u_f*nx + v_f*ny + nz
    
    nzf[nzf>-1e-1]=-1e-1 #for numerical stability at very oblique regions    
    
    b1=  np.stack( (nzf,   zero_v, lambda_reg_z*ones_v, zero_v ),axis=-1)
    b2=  np.stack( (zero_v,  nzf, zero_v, lambda_reg_z*ones_v  ),axis=-1)
    sm= np.stack( (-nx/f,-ny/f,zero_v,zero_v),axis=-1)
    Neq=b1.shape[1]
    #L1 temporary variables
    s_breg=sm.copy()
    d=np.zeros((Nmask,Neq))
    e=np.zeros((Nmask,Neq))
    residual_mat=np.zeros((Nmask,Neq))

    #pde combined coeffs
    b11= np.sum(b1*b1,1); 
    b22= np.sum(b2*b2,1); 
    b12=zero_v      
    
    Asys=make_A_matrix(b11,b12,b22,II,JJ,W0m,lambda_reg_z)

    start_time=time.time()    
    logz=LZ0m.copy()
    
    #print('s_breg shape ',s_breg.shape)
    #print('b1 shape ',b1.shape)
    I_p_x=JJ[0:Nmask]
    I_m_x=JJ[Nmask:2*Nmask]
    I_p_y=JJ[2*Nmask:3*Nmask]
    I_m_y=JJ[3*Nmask:4*Nmask]    
    
    factor = cholesky(Asys)
       
    for it_breg in range(itter_l1):
        lz_before = logz.copy();
		# z update
		# Compute System update	
        if it_breg==0:
            ll=lambda_reg_z
        else:
            ll=2*lambda_reg_z/alpha
            
        b1s = np.sum(b1*s_breg,1);
        b2s = np.sum(b2*s_breg,1);
        Bsys=make_B_vect(b1s,b2s,LZ0m,W0m,ll,I_p_x,I_m_x,I_p_y,I_m_y)
        
        #log z update through L2 minimisation.            
        logz = factor(Bsys)    
        #Residual
        zx=Gx.dot(logz)
        zy=Gy.dot(logz)  
         
        for k in range(Neq):
            residual_mat[:,k] =zx*b1[:,k]+zy*b2[:,k]-sm[:,k]        
        #d update		
        res_plus_e = (residual_mat+e)
        abs_res_plus_e = np.absolute(res_plus_e)       	
        #print('abs_res_plus_e mat ',np.amin(abs_res_plus_e), ' ',np.amax(abs_res_plus_e))
        
        abs_res_plus_e[abs_res_plus_e==0]=1	#avoid division by zero
        d = res_plus_e*np.maximum(abs_res_plus_e-2*alpha,0)/(abs_res_plus_e)        
        #print('d mat ',np.amin(d), ' ',np.amax(d))

        d[abs_res_plus_e==0]=0	
		# e update
        e = res_plus_e-d			
        #CV test			
        residual = np.linalg.norm(lz_before-logz)/np.linalg.norm(logz)
            
        print('L1 Integration-Iteration %d : residual = %.08f'%(it_breg,residual),end='\r')	
        	
        if(residual<1e-4):
            break;        
        
        s_breg=sm+d-e;
     
    res=Asys.dot(logz)-Bsys
    #logz-=np.mean(logz) #not sure why but this fixes problems        
    print('')
    #print('residual is ',np.sqrt(np.mean(res*res)))    #np.linalg.norm(res)
    #print('system solution time ',(time.time()-start_time))
    Z=np.exp(logz)   
    Z*=(mean_distance/np.mean(Z))
    #exit(0)
    #exit(0)
    return Z

def make_gradient(mask_img):
    Nmask = (mask_img == 1).sum()
    width = mask_img.shape[1]
    height = mask_img.shape[0]
    print("nmask ",Nmask," width ",width," height ",height)

    mapping_mat=-np.ones(mask_img.shape,dtype=int)

    mapping_mat[mask_img>0]=np.arange(Nmask)

    I_p_x=-np.ones((Nmask,),dtype=int)
    I_m_x=-np.ones((Nmask,),dtype=int)
    I_p_y=-np.ones((Nmask,),dtype=int)
    I_m_y=-np.ones((Nmask,),dtype=int)

    triangle_list=-np.ones((2*Nmask,4),dtype=int)
    #stupid nested loop. doto better way

    count_triangles=0
    for jj in range(0, height):
        for ii in range(0, width):
            if mask_img[jj,ii]==False:
                continue
            count=mapping_mat[jj,ii]
            #check if fw differences
            if (ii< (width-1)) and mask_img[jj,ii+1]==True:
                I_p_x[count]=mapping_mat[jj,ii+1]
                I_m_x[count]=count
            #backwards
            elif (ii> 0) and mask_img[jj,ii-1]==True:    
                I_p_x[count]=count
                I_m_x[count]=mapping_mat[jj,ii-1]
            else: #degenerate
                I_p_x[count]=count
                I_m_x[count]=count
        
            #check if up differences
            if (jj< (height-1)) and mask_img[jj+1,ii]==True:
                I_p_y[count]=mapping_mat[jj+1,ii]
                I_m_y[count]=count 
            elif (jj> 0) and mask_img[jj-1,ii]==True:    
                I_p_y[count]=count
                I_m_y[count]=mapping_mat[jj-1,ii]
            else: #degenerate
                I_p_y[count]=count
                I_m_y[count]=count  
            #count=count+1            
            if (ii< (width-1)) and mask_img[jj,ii+1]==True and (jj< (height-1)) and mask_img[jj+1,ii]==True:
                triangle_list[count_triangles,0]=3
                triangle_list[count_triangles,1]=mapping_mat[jj,ii+1]
                triangle_list[count_triangles,2]=mapping_mat[jj,ii]
                triangle_list[count_triangles,3]=mapping_mat[jj+1,ii]
                count_triangles=count_triangles+1
            if (ii> 0) and mask_img[jj,ii-1]==True and (jj> 0) and mask_img[jj-1,ii]==True:  
                triangle_list[count_triangles,0]=3
                triangle_list[count_triangles,1]=mapping_mat[jj,ii-1]
                triangle_list[count_triangles,2]=mapping_mat[jj,ii]
                triangle_list[count_triangles,3]=mapping_mat[jj-1,ii]
                count_triangles=count_triangles+1   
    #endofdoubleloop 
    triangle_list=triangle_list[0:count_triangles,:]     
    #common for Gx,Gy      
    indx=np.arange(Nmask, dtype=int)
    row = np.concatenate((indx,indx)) 
    data =np.concatenate((np.ones(Nmask, dtype=np.float32),-np.ones(Nmask, dtype=np.float32)))
    #Gx
    col=np.concatenate((I_p_x,I_m_x)) 
    Gx=csc_matrix((data, (row, col)), shape=(Nmask, Nmask))
    #Gy
    col=np.concatenate((I_p_y,I_m_y)) 
    Gy=csc_matrix((data, (row, col)), shape=(Nmask, Nmask))
    #make system L2 matrix i.e. (Gx+Gy)'*(Gx+Gy)
    II=np.concatenate((I_p_x,I_p_x,I_p_x,I_p_x, \
        I_m_x,I_m_x,I_m_x,I_m_x, \
        I_p_y,I_p_y,I_p_y,I_p_y, \
        I_m_y,I_m_y,I_m_y,I_m_y))
    JJ=np.concatenate((I_p_x,I_m_x,I_p_y,I_m_y))
    JJ=np.concatenate((JJ,JJ,JJ,JJ))
    #add the regularisation terms
    II=np.concatenate((II,indx))
    JJ=np.concatenate((JJ,indx))
    return (Gx,Gy,II,JJ,triangle_list)

def make_A_matrix(b11,b12,b22,II,JJ,W0m,lambda_reg):
    Nmask=b11.shape[0]

    KK=np.concatenate((+b11,-b11,+b12,-b12,\
    -b11,+b11,-b12,+b12, \
    +b12,-b12,+b22,-b22, \
    -b12,+b12,-b22,+b22))

    KK=np.concatenate((KK,lambda_reg*W0m))
    Asys=csc_matrix((KK, (II, JJ)), shape=(Nmask, Nmask),dtype=np.float64) #Maybe JJ,II   

    return Asys
def make_B_vect(b1s,b2s,Z0m,W0m,lambda_reg,I_p_x,I_m_x,I_p_y,I_m_y):
    Bsys=(lambda_reg*Z0m*W0m).astype(np.float64)
     
    Bsys[I_p_x] = Bsys[I_p_x]+b1s
    Bsys[I_m_x] = Bsys[I_m_x]-b1s

    Bsys[I_p_y] = Bsys[I_p_y]+b2s
    Bsys[I_m_y] = Bsys[I_m_y]-b2s
    return Bsys

