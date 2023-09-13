from ICT_model import ICT_model
import pickle
from glob import glob
import os
import trimesh
import numpy as np
from utils_pc2 import writePC2
expression_bases_ICT = [
    'browDown_L', 'browDown_R', 'browInnerUp_L', 'browInnerUp_R', 'browOuterUp_L', 'browOuterUp_R', 'cheekPuff_L', 'cheekPuff_R', 'cheekSquint_L', 'cheekSquint_R', 'eyeBlink_L', 'eyeBlink_R', 'eyeLookDown_L', 'eyeLookDown_R', 'eyeLookIn_L', 'eyeLookIn_R', 'eyeLookOut_L', 'eyeLookOut_R', 'eyeLookUp_L', 'eyeLookUp_R', 'eyeSquint_L', 'eyeSquint_R', 'eyeWide_L', 'eyeWide_R', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimple_L', 'mouthDimple_R', 'mouthFrown_L', 'mouthFrown_R', 'mouthFunnel', 'mouthLeft', 'mouthLowerDown_L', 'mouthLowerDown_R', 'mouthPress_L', 'mouthPress_R', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmile_L', 'mouthSmile_R', 'mouthStretch_L', 'mouthStretch_R', 'mouthUpperUp_L', 'mouthUpperUp_R', 'noseSneer_L', 'noseSneer_R'
] 

expression_bases_mediapipe = ['_neutral', 'browDownLeft', 'browDownRight', 'browInnerUp', 'browOuterUpLeft', 'browOuterUpRight', 'cheekPuff', 'cheekSquintLeft', 'cheekSquintRight', 'eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft', 'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft', 'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft', 'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight', 'jawForward', 'jawLeft', 'jawOpen', 'jawRight', 'mouthClose', 'mouthDimpleLeft', 'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel', 'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight', 'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight', 'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower', 'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight', 'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft', 'mouthUpperUpRight', 'noseSneerLeft', 'noseSneerRight']

double_bases = ['cheekPuff', 'browInnerUp']

mapping = {ict: (mediapipe, mediapipe_idx) for ict_idx, ict in enumerate(expression_bases_ICT) for mediapipe_idx, mediapipe in enumerate(expression_bases_mediapipe) if ict.split('_')[0] in mediapipe}
mapping_list = [mapping[ict][1] for ict in expression_bases_ICT]
def mediapipe2ict(facs):
    # Convert the mediapipe FACS to ICT FACS codes
    # facs: (n_frames, 52)
    # return: (n_frames, 53)
    if len(facs.shape) == 1:
        facs = facs[np.newaxis]
    facs_ict = facs[:, mapping_list]
    return facs_ict
        
    
def facs2obj(facs, save_dir, ict_model):
    # Load the ICT model
    
    # Load the FACS codes
    if len(facs.shape) == 1:
        facs = facs[np.newaxis]
    facs_ict = mediapipe2ict(facs)
    out_verts = ict_model.deform(facs_ict)
    faces = ict_model.faces
    writePC2(os.path.join(save_dir, 'facs.pc2'), out_verts)

    
    
if __name__ == '__main__':
    ict_model = ICT_model.load(2)
    facs = pickle.load(open('C:\\Users\\wwpokenet\\WORK\\face_ani\\mediapipe\\output\\pkl\\__f2KtcXAxI_0.pkl', 'rb'))['blendshapes']
    
    os.makedirs('./__f2KtcXAxI_0', exist_ok=True)
    
    facs2obj(facs, './__f2KtcXAxI_0', ict_model)
