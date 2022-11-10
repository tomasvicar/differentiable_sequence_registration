import imageio
import ffmpeg
import numpy as np


def load_video(file_name,np_dtype):
    
    with imageio.get_reader(file_name) as f:
        imgs = []
        for frame_num, frame in enumerate(f):
            imgs.append(frame[:,:,0].astype(np_dtype)/255)
        return np.stack(imgs, axis=0)
    
def load_video_nocorrupted(file_name,np_dtype,corr_inds):
    
    with imageio.get_reader(file_name) as f:
        imgs = []
        for frame_num, frame in enumerate(f):
            if frame_num in corr_inds:
                continue
            imgs.append(frame[:,:,0].astype(np_dtype)/255)
        return np.stack(imgs, axis=0)
    
    
def save_video(filename,output):
    
    frame_rate = 25 

    ff_proc = (
        ffmpeg
        .input('pipe:',format='rawvideo',pix_fmt='gray',s=str(output.shape[2]) + 'x' + str(output.shape[1]),r=str(frame_rate))
        .output(filename,vcodec='ffv1', an=None)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )
    
    for frame_num in range(output.shape[0]):
    
        frame = output[frame_num, :, :]
        
        frame = np.round(frame).astype(np.uint8)
        
        ff_proc.stdin.write(frame)
    
    ff_proc.stdin.close()
    
