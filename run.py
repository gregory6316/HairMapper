import os
from argparse import Namespace
import os.path
import argparse
import sys
os.chdir('./encoder4editing')
sys.path.append(".")
sys.path.append("..")
from models.psp import pSp
os.chdir('..')
import bz2
import scipy.ndimage
import PIL.Image
import dlib
import torchvision.transforms as transforms
import PIL.Image
import cv2
from styleGAN2_ada_model.stylegan2_ada_generator import StyleGAN2adaGenerator
from classifier.src.feature_extractor.hair_mask_extractor import get_hair_mask, get_parsingNet
from mapper.networks.level_mapper import LevelMapper
import torch
from diffuse.inverter_remove_hair import InverterRemoveHair
import numpy as np
from PIL import ImageFile



ImageFile.LOAD_TRUNCATED_IMAGES = True

def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):

    lm = np.array(face_landmarks)
    lm_chin = lm[0: 17]
    lm_eyebrow_left = lm[17: 22]
    lm_eyebrow_right = lm[22: 27]
    lm_nose = lm[27: 31]
    lm_nostrils = lm[31: 36]
    lm_eye_left = lm[36: 42]
    lm_eye_right = lm[42: 48]
    lm_mouth_outer = lm[48: 60]
    lm_mouth_inner = lm[60: 68]


    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)

    img.save(dst_file, 'PNG')

def unpack_bz2(src_path):
    data = bz2.BZ2File(src_path).read()
    dst_path = src_path[:-4]
    with open(dst_path, 'wb') as fp:
        fp.write(data)
    return dst_path

class LandmarksDetector:
    def __init__(self, predictor_model_path):
        self.detector = dlib.get_frontal_face_detector() # cnn_face_detection_model_v1 also can be used
        self.shape_predictor = dlib.shape_predictor(predictor_model_path)

    def get_landmarks(self, image):
        img = dlib.load_rgb_image(image)
        dets = self.detector(img, 1)

        for detection in dets:
            face_landmarks = [(item.x, item.y) for item in self.shape_predictor(img, detection).parts()]
            yield face_landmarks

def run_on_batch(inputs, net):
    latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return latents


parser = argparse.ArgumentParser(description='HairMapper pipeline')
parser.add_argument('--input_image_path', help='Input image path', type=str, required=True)
parser.add_argument('--output_image_path', help='output image path', type=str, required=True)


if __name__ == "__main__":
    landmarks_model_path = unpack_bz2('./ckpts/shape_predictor_68_face_landmarks.dat.bz2')
    args = parser.parse_args()

    landmarks_detector = LandmarksDetector(landmarks_model_path)

    aligned_image_path = args.input_image_path[:-4] + '.png'

    for i, face_landmarks in enumerate(landmarks_detector.get_landmarks(args.input_image_path), start=1):
        image_align(args.input_image_path, aligned_image_path, face_landmarks)

    img_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
    )
    model_path = "./ckpts/e4e_ffhq_encode.pt"
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    net = pSp(opts)
    net.eval()
    net.cuda()
    name = os.path.basename(aligned_image_path)[:-4]
    code_path = f'{name}.npy'
    input_image = PIL.Image.open(aligned_image_path)
    transformed_image = img_transforms(input_image)
    with torch.no_grad():
        latents = run_on_batch(transformed_image.unsqueeze(0), net)
        latent = latents[0].cpu().numpy()
        latent = np.reshape(latent, (1, 18, 512))
        np.save(code_path, latent)
        print(f'save to {code_path}')
    model_name = 'stylegan2_ada'
    latent_space_type = 'wp'
    print(f'Initializing generator.')
    model = StyleGAN2adaGenerator(model_name, logger=None, truncation_psi=1.0)

    mapper = LevelMapper(input_dim=512).eval().cuda()
    ckpt = torch.load('./ckpts/best_model.pt')
    alpha = float(ckpt['alpha']) * 1.2
    mapper.load_state_dict(ckpt['state_dict'], strict=True)
    kwargs = {'latent_space_type': latent_space_type}
    parsingNet = get_parsingNet(save_pth='./ckpts/face_parsing.pth')
    inverter = InverterRemoveHair(
        model_name,
        Generator=model,
        learning_rate=0.01,
        reconstruction_loss_weight=1.0,
        perceptual_loss_weight=5e-5,
        truncation_psi=1.0,
        logger=None
    )

    name = os.path.basename(code_path)[:-4]

    latent_codes_origin = np.reshape(np.load(code_path), (1, 18, 512))

    mapper_input = latent_codes_origin.copy()
    mapper_input_tensor = torch.from_numpy(mapper_input).cuda().float()
    edited_latent_codes = latent_codes_origin
    edited_latent_codes[:, :8, :] += alpha * mapper(mapper_input_tensor).to('cpu').detach().numpy()

    origin_img = cv2.imread(aligned_image_path)

    outputs = model.easy_style_mixing(latent_codes=edited_latent_codes,
                                      style_range=range(7, 18),
                                      style_codes=latent_codes_origin,
                                      mix_ratio=0.8,
                                      **kwargs
                                      )

    edited_img = outputs['image'][0][:, :, ::-1]

    hair_mask = get_hair_mask(img_path=aligned_image_path, net=parsingNet, include_hat=True, include_ear=True)

    mask_dilate = cv2.dilate(hair_mask,
                             kernel=np.ones((50, 50), np.uint8))
    mask_dilate_blur = cv2.blur(mask_dilate, ksize=(30, 30))
    mask_dilate_blur = (hair_mask + (255 - hair_mask) / 255 * mask_dilate_blur).astype(np.uint8)

    face_mask = 255 - mask_dilate_blur

    index = np.where(face_mask > 0)
    cy = (np.min(index[0]) + np.max(index[0])) // 2
    cx = (np.min(index[1]) + np.max(index[1])) // 2
    center = (cx, cy)

    res_save_path = args.output_image_path

    mixed_clone = cv2.seamlessClone(origin_img, edited_img, face_mask[:, :, 0], center, cv2.NORMAL_CLONE)

    cv2.imwrite(args.output_image_path, mixed_clone)
    os.remove(f'{name}.npy')
