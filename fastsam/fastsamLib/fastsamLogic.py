from slicer.ScriptedLoadableModule import *
import slicer
import numpy as np
import pickle
import vtk
import SimpleITK as sitk
import sys
from functools import partial
import time
import random

class fastsamLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        """
        Called when the logic class is instantiated. Can be used for initializing member variables.
        """
        ScriptedLoadableModuleLogic.__init__(self)
        self._parameterNode = self.getParameterNode()

        self.sam, self.predictor, self.device = None, None, None
        self.img, self.voxel_sizes, self.mask, self.embeddings = np.zeros((1,1,1)), np.zeros(3), np.zeros((1,1,1)), []
        self.min_mask_region_area = 500
        self.ind = 0
        self.image_size = 128
        self.torch = None

        self.include_coords = {}
        self.exclude_coords = {}

        self.emb_slice_d = {'Yellow': 2, 'Green': 1, 'Red': 0}
        self.slice_direction = 'Red'
        self.dimension = 3

        self.mask_locations = set()
        self.interp_slice_direction = set()
        self.mask_backup = None
        self.low_res_masks = None
        self.selfensemblingnumber = 100

    def setupPythonRequirements(self):

        # Install PyTorch
        try:
            import PyTorchUtils
        except ModuleNotFoundError as e:
            slicer.util.errorDisplay("This module requires PyTorch extension. Install it from the Extensions Manager.")
            return False

        minimumTorchVersion = "1.7"
        minimumTorchVisionVersion = "0.8"
        torchLogic = PyTorchUtils.PyTorchUtilsLogic()
        if not torchLogic.torchInstalled():
            slicer.util.delayDisplay("PyTorch Python package is required. Installing... (it may take several minutes)")
            torch = torchLogic.installTorch(askConfirmation=True, torchVersionRequirement=f">={minimumTorchVersion}",
                                            torchvisionVersionRequirement=f">={minimumTorchVisionVersion}")
            if torch is None:
                raise ValueError('PyTorch extension needs to be installed to use this module.')
        else:
            # torch is installed, check version
            from packaging import version
            if version.parse(torchLogic.torch.__version__) < version.parse(minimumTorchVersion):
                raise ValueError(f'PyTorch version {torchLogic.torch.__version__} is not compatible with this module.'
                                 + f' Minimum required version is {minimumTorchVersion}. You can use "PyTorch Util" module to install PyTorch'
                                 + f' with version requirement set to: >={minimumTorchVersion}')
        self.torch = torchLogic.importTorch()
        #Install SAM
        # try:
        #     from segment_anything import sam_model_registry3D, SamPredictor
        # except ModuleNotFoundError:
        #     slicer.util.pip_install("https://github.com/swedfr/fastsam/archive/refs/heads/main.zip")
        #     from segment_anything import sam_model_registry3D, SamPredictor
        #     print(1)
        return True

    def create_sam(self, sam_weights_path, modeltype):
        slicer.util.delayDisplay("Loading SAM ... ")
        if not self.setupPythonRequirements():
            return
        from segment_anything.build_sam3D import sam_model_registry3D
        # from segment_anything import sam_model_registry3D
        print('build SAM3D from ' + sam_weights_path)
        try:
            self.sam = sam_model_registry3D[modeltype](checkpoint=sam_weights_path)
        except FileNotFoundError:
            slicer.util.infoDisplay("SAM weights not found, use Download button")
            print("weights not found")
            return

        if self.torch.cuda.is_available():
            self.device = "cuda:0"
            self.sam.to(device="cuda")
        else:
            self.device = "cpu"
        # model_dict = self.torch.load(sam_weights_path, map_location=self.device)
        # state_dict = model_dict['model_state_dict']
        # self.sam.load_state_dict(state_dict)
        # # self.predictor = SamPredictor(self.sam)
        # self.predictor.is_image_set = True
        print("Done")
    def generateaffine(self):
        self.img = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("fastsamInputVolume"))
        self.origin = self._parameterNode.GetNodeReference("fastsamInputVolume").GetOrigin()
        self.spacing = self._parameterNode.GetNodeReference("fastsamInputVolume").GetSpacing()
        self.affine = np.zeros((4,4))        
        dir = np.eye(3)
        self._parameterNode.GetNodeReference("fastsamInputVolume").GetIJKToRASDirections(dir)
        for i in range(0,3):
            self.affine[i,i] = self.spacing[i] * dir[i,i]
        # self.affine[0,0] = -self.affine[0,0]
        # # self.affine[1,1] = -self.affine[1,1]
        self.affine[3,3] = 1
        self.affine[:3, 3] = self.origin
    def create_embeddings(self, output_filepath):
        # self.img = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("fastsamInputVolume"))
        # self.tinyvit = ImageEncoderViT3D(                                    #
        #     depth=6,
        #     embed_dim=768,
        #     img_size=128,
        #     mlp_ratio=4,
        #     norm_layer=partial(self.torch.nn.LayerNorm, eps=1e-6),
        #     num_heads=6,
        #     patch_size=16,
        #     qkv_bias=True,
        #     use_rel_pos=True,
        #     global_attn_indexes=[2,5,8,11],
        #     window_size=0,
        #     out_chans=384,
        #     skip_layer = 2,
        # )
        # input = self.img
        # input_image_torch = self.torch.as_tensor(input, device=self.device)
        # model_dict = self.torch.load('fastsam.pth', map_location=self.device)
        # state_dict = model_dict['model_state_dict']
        # self.tinyvit.load_state_dict(state_dict)
        
        # embeddings = self.tinyvit(input_image_torch)
        # if self.img.ndim > 3 or self.img.ndim < 2:
        #     raise Exception("Unsupported image type.")
        # elif self.img.ndim == 2:
        #     self.img = self.img[:, :, np.newaxis]
        # input = self.torch.from_numpy(self.img)
        # print(input.shape)
        # embeddings = self.sam.image_encoder(input)
        # embeddings = [[], [], []]
        # slice_direction = ['x', 'y', 'z']
        # for i, d in enumerate(slice_direction):
        #     print(f"\nSlicing along {d} direction")
        #     for k in range(self.img.shape[i]):
        #         if i == 0:
        #             img_slice = self.img[k]
        #         elif i == 1:
        #             img_slice = self.img[:, k]
        #         else:
        #             img_slice = self.img[:, :, k]

        #         slicer.util.delayDisplay(f"Creating embeddings for {output_filepath} with dims: {self.img.shape} \n"
        #                                  f"Slicing along {d} direction, {k + 1}/{self.img.shape[i]} image")
        #         sys.stdout.write(f"\rCreating embedding for {k + 1}/{self.img.shape[i]} image")

        #         self.predictor.set_image(np.repeat(img_slice[:, :, np.newaxis], 3, axis=2))
        #         embeddings[i].append({'original_size': self.predictor.original_size,
        #                               'input_size': self.predictor.input_size,
        #                               'features': self.predictor.features.to('cpu')})

        #         self.predictor.reset_image()
        #         if self.torch.cuda.is_available():
        #             self.torch.cuda.empty_cache()

        # with open(output_filepath, 'wb') as f:
        #     pickle.dump(embeddings, f)
        #     print(f"\nSaved {output_filepath}")
        return output_filepath

    def read_img_embeddings(self, embeddings_filepath):
        
        self.img = slicer.util.arrayFromVolume(self._parameterNode.GetNodeReference("fastsamInputVolume"))
        ras2ijk = vtk.vtkMatrix4x4()
        self._parameterNode.GetNodeReference("fastsamInputVolume").GetRASToIJKMatrix(ras2ijk)
        self.voxel_sizes[:] = slicer.util.arrayFromVTKMatrix(ras2ijk).diagonal()[:3]

        print("Reading embeddings ... ", end="")
        with open(embeddings_filepath, 'rb') as f:
            self.embeddings = pickle.load(f)
        print("Done")

        # checking image vs embeddings dimensions
        if (np.any(np.array(self.img.shape)[[1, 2]] != np.array(self.embeddings[0][0]['original_size'])) or
                np.any(np.array(self.img.shape)[[0, 2]] != np.array(self.embeddings[1][0]['original_size'])) or
                np.any(np.array(self.img.shape)[[0, 1]] != np.array(self.embeddings[2][0]['original_size']))):
            slicer.util.errorDisplay(f"Embeddings dimensions {(len(self.embeddings[0]), len(self.embeddings[1]), len(self.embeddings[2]))} "
                                     f"don't match image {self.img.shape}")
            self.embeddings = []

    def pass_mask_to_slicer(self):
        slicer.util.updateSegmentBinaryLabelmapFromArray(self.mask,
                                                         self._parameterNode.GetNodeReference("fastsamSegmentation"),
                                                         self._parameterNode.GetParameter("fastsamCurrentSegment"),
                                                         self._parameterNode.GetNodeReference("fastsamInputVolume"))
    
    # def pass_logitsmask_to_slicer(self):
    #     slicer.util.updateSegmentBinaryLabelmapFromArray(self.logitsmask,
    #                                                 self._parameterNode.GetNodeReference("fastsamSegmentation"),
    #                                                 self._parameterNode.GetParameter("fastsamCurrentSegment"),
    #                                                 self._parameterNode.GetNodeReference("fastsamInputVolume"))
    
    def findboxcontainallpoints(self,include_coords,exclude_coords,padded_data):
        minpoints = []
        maxpoints = []
        for i in range(0,self.dimension):
            points = []
            for coords in include_coords:
                points.append(coords[i])
            for coords in exclude_coords:
                points.append(coords[i])
            maxp = np.max(points)
            minp = np.min(points)
            if maxp-minp > self.image_size:
                slicer.util.errorDisplay(f"points outside the embedding range, please reselect the points")
                return
            bound = padded_data.shape[i]
            crop = int((self.image_size-(maxp-minp))/2)
            minpoints.append(int(minp - min(minp,crop) - crop + min((bound-maxp),crop)))
            maxpoints.append(int(maxp + min((bound-maxp),crop) + crop-min(minp,crop)))
            if maxpoints[i] - minpoints[i] != self.image_size:
                if minpoints[i] > 0:
                    minpoints[i] -= 1
                else:
                    maxpoints[i] += 1
        return minpoints,maxpoints
    
    def reverse_padd(self, padding):
        return self.mask[padding[0][0]:self.mask.shape[0] - padding[0][1],padding[1][0]:self.mask.shape[1] - padding[1][1],padding[2][0]:self.mask.shape[2] - padding[2][1]]            
        
    def get_mask(self, first_freeze):
        # t = time.time()
        target_shape = [self.image_size]*self.dimension
        if self.dimension == 3:
            pad_width = [(max(0, target_shape[i] - self.img.shape[i]) // 2, max(0, target_shape[i] - self.img.shape[i]) // 2)for i in range(self.dimension)]
            for i in range(0,self.dimension):
                if pad_width[i][0] + pad_width[i][1] + self.img.shape[i] < target_shape[i]:
                    l = list(pad_width[i])
                    l[0] += 1
                    pad_width[i] = l            
            include_points = [[coords[0], coords[1],coords[2]] for coords in self.include_coords.values()]
            # print(self.img[0,0,include_points[0],include_points[1],include_points[2]]) 
            exclude_points = [[coords[0], coords[1],coords[2]] for coords in self.exclude_coords.values()]
            offsets = [pad_width[i][0] for i in range(self.dimension)]
            adjusted_include_points = [[coord + offset for coord, offset in zip(point, offsets)] for point in include_points]
            adjusted_exclude_points = [[coord + offset for coord, offset in zip(point, offsets)] for point in exclude_points]
            include_points = adjusted_include_points
            exclude_points = adjusted_exclude_points
            padded_data = np.pad(self.img, pad_width, 'constant')      
            minpoints,maxpoints = self.findboxcontainallpoints(include_points,exclude_points,padded_data)
            #但裁完变成128x128x128
            inputimage = padded_data[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]]
            inputimage = inputimage[np.newaxis,np.newaxis,:,:,:]
            inputimage = self.torch.as_tensor(inputimage,dtype = self.torch.float32)
            #根据源图片做的坐标。include根据input偏移了，include points exclude points 根据才出来的坐标重新定位

            offsets = [minpoints[i] for i in range(self.dimension)]
            adjusted_include_points = [[coord - offset for coord, offset in zip(point, offsets)] for point in include_points]
            adjusted_exclude_points = [[coord - offset for coord, offset in zip(point, offsets)] for point in exclude_points]
            include_points = adjusted_include_points
            exclude_points = adjusted_exclude_points
            # if first_freeze:
            #     self.backup_mask()
            if len(self.include_coords) != 0:
                prev_masks = self.torch.zeros_like(inputimage).to(self.device)
                low_res_masks = self.torch.nn.functional.interpolate(prev_masks.float(), size=(self.image_size//4,self.image_size//4,self.image_size//4))
                image_embedding = self.sam.image_encoder(inputimage.to(self.device))
                points = self.torch.as_tensor(np.array(include_points + exclude_points)).to(self.device)
                points = points[None,:,:]
                label = self.torch.as_tensor(np.array([1] * len(include_points) + [0] * len(exclude_points))).to(self.device)
                label = label[None,:]
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=[points,label],
                        boxes=None,
                        masks = low_res_masks
                    )
                low_res_masks, _ = self.sam.mask_decoder(
                        image_embeddings=image_embedding.to(self.device), # (B, 384, 64, 64, 64)
                        image_pe=self.sam.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                        dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                        multimask_output=False,
                        )
                prev_masks = self.torch.nn.functional.interpolate(low_res_masks, size=inputimage.shape[-3:], mode='trilinear', align_corners=False)
                self.prev = prev_masks
                medsam_seg_prob = self.torch.sigmoid(prev_masks)  # (B, 1, 64, 64, 64)
                # convert prob to mask
                medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
                medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                self.selfensemblingmask = self.torch.from_numpy(medsam_seg)
                self.mask = np.zeros(padded_data.shape)
                self.mask[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]] = medsam_seg
                self.mask = self.reverse_padd(pad_width)
                # self.logitsmask = np.zeros(padded_data.shape)
                # self.logitsmask[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]] = medsam_seg_prob
                # self.logitsmask = self.reverse_padd(pad_width)
                # print(np.unique(self.logitsmask))
                self.pass_mask_to_slicer()
            else:
                self.undo()
        else:
            if len(self.include_coords) != 0:
                if self.slice_direction == 'Red':
                    include_points = [[coords[2], coords[1]] for coords in self.include_coords.values()]
                    exclude_points = [[coords[2], coords[1]] for coords in self.exclude_coords.values()]
                    self.ind = int(list(self.include_coords.values())[0][0])
                    img = self.img[int(list(self.include_coords.values())[0][0]),:,:]
                    originalsize = (self.img.shape[1],self.img.shape[2])
                elif self.slice_direction == 'Green':
                    include_points = [[coords[2], coords[0]] for coords in self.include_coords.values()]
                    exclude_points = [[coords[2], coords[0]] for coords in self.exclude_coords.values()]
                    self.ind = int(list(self.include_coords.values())[0][1])
                    img = self.img[:,int(list(self.include_coords.values())[0][1]),:]
                    originalsize = (self.img.shape[0],self.img.shape[2])
                else:  # Y
                    include_points = [[coords[1], coords[0]] for coords in self.include_coords.values()]
                    exclude_points = [[coords[1], coords[0]] for coords in self.exclude_coords.values()]
                    self.ind = int(list(self.include_coords.values())[0][2])
                    img = self.img[:,:,int(list(self.include_coords.values())[0][2])]
                    originalsize = (self.img.shape[0],self.img.shape[1])
            pad_width = [(max(0, target_shape[i] - img.shape[i]) // 2, max(0, target_shape[i] - img.shape[i]) // 2)for i in range(self.dimension)]
            for i in range(0,self.dimension):
                if pad_width[i][0] + pad_width[i][1] + img.shape[i] < target_shape[i]:
                    l = list(pad_width[i])
                    l[0] += 1
                    pad_width[i] = l
            offsets = [pad_width[i][0] for i in range(self.dimension)]
            adjusted_include_points = [[coord + offset for coord, offset in zip(point, offsets)] for point in include_points]
            adjusted_exclude_points = [[coord + offset for coord, offset in zip(point, offsets)] for point in exclude_points]
            include_points = adjusted_include_points
            exclude_points = adjusted_exclude_points
            padded_data = np.pad(img, pad_width, 'constant')
            minpoints,maxpoints = self.findboxcontainallpoints(include_points,exclude_points,padded_data)
            inputimage = padded_data[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1]]
            inputimage = inputimage[np.newaxis,np.newaxis,:,:]
            inputimage = self.torch.as_tensor(inputimage,dtype = self.torch.float32)
            inputimage3 = self.torch.repeat_interleave(inputimage, repeats=3, dim=1)
            offsets = [minpoints[i] for i in range(self.dimension)]
            adjusted_include_points = [[coord - offset for coord, offset in zip(point, offsets)] for point in include_points]
            adjusted_exclude_points = [[coord - offset for coord, offset in zip(point, offsets)] for point in exclude_points]
            include_points = adjusted_include_points
            exclude_points = adjusted_exclude_points
            # if first_freeze:
            #     self.backup_mask()
            if len(self.include_coords) != 0:
                prev_masks = self.torch.zeros_like(inputimage).to(self.device)
                image_embedding = self.sam.image_encoder(inputimage3.to(self.device))
                points = self.torch.as_tensor(np.array(include_points + exclude_points)).to(self.device)
                points = points[None,:,:]
                label = self.torch.as_tensor(np.array([1] * len(include_points) + [0] * len(exclude_points))).to(self.device)
                label = label[None,:]
                sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                        points=[points,label],
                        boxes=None,
                        masks = self.low_res_masks
                    )
                self.low_res_masks, _ = self.sam.mask_decoder(
                        image_embeddings=image_embedding.to(self.device), # (B, 384, 64, 64, 64)
                        image_pe=self.sam.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                        dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                        multimask_output=False,
                        )
                prev_masks = self.torch.nn.functional.interpolate(self.low_res_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False,)
                self.prev = prev_masks
                med_masks = self.postprocess_masks(self.low_res_masks,self.image_size,originalsize)
                med_masks = self.torch.sigmoid(med_masks)
                
                self.backup_mask()
                med_masks = med_masks.cpu().detach().numpy().squeeze()
                # if self.slice_direction == 'Red':
                #     self.logitsmask[self.ind] = med_masks 
                # elif self.slice_direction == 'Green':
                #     self.logitsmask[:, self.ind] = med_masks 
                # else:
                #     self.logitsmask[:, :, self.ind] = med_masks 
                med_masks = med_masks > 0.5
                new_mask = med_masks.astype(np.uint8)
                
                new_mask = self.remove_small_regions(new_mask, self.min_mask_region_area, "holes")
                new_mask = self.remove_small_regions(new_mask, self.min_mask_region_area, "islands")
                self.selfensemblingmask = self.torch.from_numpy(new_mask)
                if self.slice_direction == 'Red':
                    self.mask[self.ind] = new_mask
                elif self.slice_direction == 'Green':
                    self.mask[:, self.ind] = new_mask
                else:
                    self.mask[:, :, self.ind] = new_mask
                # print(np.unique(self.logitsmask))
                self.pass_mask_to_slicer()
            else:
                self.undo()
        # print(time.time()-t)
            
    def reverse_pddd(self, mask, padding):
        return mask[padding[0][0]:mask.shape[0] - padding[0][1],padding[1][0]:mask.shape[1] - padding[1][1],padding[2][0]:mask.shape[2] - padding[2][1]]            
        
    def selfensembling(self):
        if (len(self.include_coords) == 0):
            slicer.util.errorDisplay(f"please select an include points first")
            return
        else:
            target_shape = [self.image_size]*self.dimension
            if self.dimension == 3:
                pad_width = [(max(0, target_shape[i] - self.img.shape[i]) // 2, max(0, target_shape[i] - self.img.shape[i]) // 2)for i in range(self.dimension)]
                for i in range(0,self.dimension):
                    if pad_width[i][0] + pad_width[i][1] + self.img.shape[i] < target_shape[i]:
                        l = list(pad_width[i])
                        l[0] += 1
                        pad_width[i] = l            
                include_points = [[coords[0], coords[1],coords[2]] for coords in self.include_coords.values()]
                include_points = [include_points[-1]]
                # print(self.img[0,0,include_points[0],include_points[1],include_points[2]]) 
                offsets = [pad_width[i][0] for i in range(self.dimension)]
                adjusted_include_points = [[coord + offset for coord, offset in zip(point, offsets)] for point in include_points]
                include_points = adjusted_include_points
                padded_data = np.pad(self.img, pad_width, 'constant')        
                minpoints,maxpoints = self.findboxcontainallpoints(include_points,[],padded_data)
                #但裁完变成128x128x128
                inputimage = padded_data[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]]
                inputimage = inputimage[np.newaxis,np.newaxis,:,:,:]
                inputimage = self.torch.as_tensor(inputimage,dtype = self.torch.float32)
                #根据源图片做的坐标。include根据input偏移了，include points exclude points 根据才出来的坐标重新定位

                offsets = [minpoints[i] for i in range(self.dimension)]
                adjusted_include_points = [[coord - offset for coord, offset in zip(point, offsets)] for point in include_points]
                include_points = adjusted_include_points
                # if first_freeze:
                #     self.backup_mask()
                # prev_masks = self.torch.zeros_like(inputimage).to(self.device)
                # low_res_masks = self.torch.nn.functional.interpolate(prev_masks.float(), size=(self.image_size//4,self.image_size//4,self.image_size//4))
                image_embedding = self.sam.image_encoder(inputimage.to(self.device))
                for i in range(0,self.selfensemblingnumber):
                    prev_masks = self.prev.to(self.device)
                    low_res_masks = self.torch.nn.functional.interpolate(prev_masks.float(), size=(self.image_size//4,self.image_size//4,self.image_size//4))
                    point,label = self.get_next_click3D_torch_2(self.selfensemblingmask.to(self.device))
                    point = point[None,:,:]
                    # label = self.torch.as_tensor(np.array([1])).to(self.device)
                    label = label[None,:]
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                            points=[point,label],
                            boxes=None,
                            masks = low_res_masks
                        )
                    low_res_masks, _ = self.sam.mask_decoder(
                            image_embeddings=image_embedding.to(self.device), # (B, 384, 64, 64, 64)
                            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                            dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                            multimask_output=False,
                            )
                    prev_masks = self.torch.nn.functional.interpolate(low_res_masks, size=inputimage.shape[-3:], mode='trilinear', align_corners=False)
                    if i == 0:
                        medsam_seg_prob = self.torch.sigmoid(prev_masks)
                    else :
                        medsam_seg_prob += self.torch.sigmoid(prev_masks)
                    # (B, 1, 64, 64, 64)
                # convert prob to mask
                    # medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
                    # medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                    # mask = np.zeros(padded_data.shape)
                    # mask[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]] = medsam_seg
                    # mask = self.reverse_padd(pad_width)
                medsam_seg_prob = medsam_seg_prob / self.selfensemblingnumber
                medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
                colors = [ (250, 250, 210),
                          (200, 200, 235),
                          (48, 129, 126),
                          (144,238, 144),
                          (128,174,128),
                            (145, 30, 0),
                            (185, 102, 83),
                            (216, 101, 79),
                            (145, 60,66)]
                # a = ((medsam_seg_prob > 0.9) & (medsam_seg_prob <= 1.0)).astype(np.uint8)
                # b = ((medsam_seg_prob > 0.8) & (medsam_seg_prob <= 0.9)).astype(np.uint8)
                # print((a != b).all())
                for i in range(1,10):
                    medsam_seg = ((medsam_seg_prob > i*0.1) & (medsam_seg_prob <= (i + 1) * 0.1)).astype(np.uint8)
                    # if i > 0:
                    #     print(((medsam_seg == 1) == (m == 1)).all())
                    # m = medsam_seg
                    mask = np.zeros(padded_data.shape)
                    mask[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]] = medsam_seg
                    mask = self.reverse_pddd(mask,pad_width)
                    segmentID = self._parameterNode.GetNodeReference("fastsamselfensembling").GetSegmentation().AddEmptySegment()
                    self._parameterNode.GetNodeReference("fastsamselfensembling").GetSegmentation().GetSegment(segmentID).SetColor(colors[i-1][0]/255,colors[i-1][1]/255,colors[i-1][2]/255)
                    self._parameterNode.SetParameter("selfensemblingmask", segmentID)
                    slicer.util.updateSegmentBinaryLabelmapFromArray(mask,
                                                self._parameterNode.GetNodeReference("fastsamselfensembling"),
                                                self._parameterNode.GetParameter("selfensemblingmask"),
                                                self._parameterNode.GetNodeReference("fastsamInputVolume"))
                # self.logitsmask = np.zeros(padded_data.shape)
                # self.logitsmask[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]] = medsam_seg_prob
                # self.logitsmask = self.reverse_padd(pad_width)
                # print(np.unique(self.logitsmask))
            else:
                if len(self.include_coords) != 0:
                    if self.slice_direction == 'Red':
                        include_points = [[coords[2], coords[1]] for coords in self.include_coords.values()]
                        include_points = [include_points[-1]]
                        self.ind = int(list(self.include_coords.values())[0][0])
                        img = self.img[int(list(self.include_coords.values())[0][0]),:,:]
                        originalsize = (self.img.shape[1],self.img.shape[2])
                    elif self.slice_direction == 'Green':
                        include_points = [[coords[2], coords[0]] for coords in self.include_coords.values()]
                        include_points = [include_points[-1]]
                        self.ind = int(list(self.include_coords.values())[0][1])
                        img = self.img[:,int(list(self.include_coords.values())[0][1]),:]
                        originalsize = (self.img.shape[0],self.img.shape[2])
                    else:  # Y
                        include_points = [[coords[1], coords[0]] for coords in self.include_coords.values()]
                        include_points = [include_points[-1]]
                        self.ind = int(list(self.include_coords.values())[0][2])
                        img = self.img[:,:,int(list(self.include_coords.values())[0][2])]
                        originalsize = (self.img.shape[0],self.img.shape[1])
                pad_width = [(max(0, target_shape[i] - img.shape[i]) // 2, max(0, target_shape[i] - img.shape[i]) // 2)for i in range(self.dimension)]
                for i in range(0,self.dimension):
                    if pad_width[i][0] + pad_width[i][1] + img.shape[i] < target_shape[i]:
                        l = list(pad_width[i])
                        l[0] += 1
                        pad_width[i] = l
                offsets = [pad_width[i][0] for i in range(self.dimension)]
                adjusted_include_points = [[coord + offset for coord, offset in zip(point, offsets)] for point in include_points]
                include_points = adjusted_include_points
                padded_data = np.pad(img, pad_width, 'constant')
                minpoints,maxpoints = self.findboxcontainallpoints(include_points,[],padded_data)
                inputimage = padded_data[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1]]
                inputimage = inputimage[np.newaxis,np.newaxis,:,:]
                inputimage = self.torch.as_tensor(inputimage,dtype = self.torch.float32)
                inputimage3 = self.torch.repeat_interleave(inputimage, repeats=3, dim=1)
                offsets = [minpoints[i] for i in range(self.dimension)]
                adjusted_include_points = [[coord - offset for coord, offset in zip(point, offsets)] for point in include_points]
                include_points = adjusted_include_points
                # if first_freeze:
                #     self.backup_mask()
                image_embedding = self.sam.image_encoder(inputimage3.to(self.device))
                for i in range(0,self.selfensemblingnumber):
                    prev_masks = self.prev.to(self.device)
                    points,label = self.get_next_click3D_torch_2(self.selfensemblingmask.to(self.device))
                    points = points[None,:,:]
                    label = label[None,:]
                    sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
                            points=[points,label],
                            boxes=None,
                            masks = self.low_res_masks
                        )
                    low_res_masks, _ = self.sam.mask_decoder(
                            image_embeddings=image_embedding.to(self.device), # (B, 384, 64, 64, 64)
                            image_pe=self.sam.prompt_encoder.get_dense_pe(), # (1, 384, 64, 64, 64)
                            sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 384)
                            dense_prompt_embeddings=dense_embeddings, # (B, 384, 64, 64, 64)
                            multimask_output=False,
                            )
                    # prev_masks = self.torch.nn.functional.interpolate(low_res_masks, (self.image_size, self.image_size), mode="bilinear", align_corners=False,)
                    if i == 0:
                        medsam_seg_prob = self.torch.sigmoid(low_res_masks)
                    else :
                        medsam_seg_prob += self.torch.sigmoid(low_res_masks)
                    # (B, 1, 64, 64, 64)
                # convert prob to mask
                    # medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
                    # medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)
                    # mask = np.zeros(padded_data.shape)
                    # mask[minpoints[0]:maxpoints[0],minpoints[1]:maxpoints[1],minpoints[2]:maxpoints[2]] = medsam_seg
                    # mask = self.reverse_padd(pad_width)
                medsam_seg_prob = medsam_seg_prob / self.selfensemblingnumber
                medsam_seg_prob = self.postprocess_masks(medsam_seg_prob,self.image_size,originalsize)
                medsam_seg_prob = medsam_seg_prob.cpu().detach().numpy().squeeze()
                colors = [ (250, 250, 210),
                          (200, 200, 235),
                          (48, 129, 126),
                          (144,238, 144),
                          (128,174,128),
                            (145, 30, 0),
                            (185, 102, 83),
                            (216, 101, 79),
                            (145, 60,66)]
                # a = ((medsam_seg_prob > 0.9) & (medsam_seg_prob <= 1.0)).astype(np.uint8)
                # b = ((medsam_seg_prob > 0.8) & (medsam_seg_prob <= 0.9)).astype(np.uint8)
                # print((a != b).all())
                for i in range(1,10):
                    medsam_seg = ((medsam_seg_prob > i*0.1) & (medsam_seg_prob <= (i + 1) * 0.1)).astype(np.uint8)
                    # if i > 0:
                    #     print(((medsam_seg == 1) == (m == 1)).all())
                    # m = medsam_seg
                    mask = np.zeros(self.mask.shape)
                    if self.slice_direction == 'Red':
                        mask[self.ind] = medsam_seg
                    elif self.slice_direction == 'Green':
                        mask[:, self.ind] = medsam_seg
                    else:
                        mask[:, :, self.ind] = medsam_seg
                    segmentID = self._parameterNode.GetNodeReference("fastsamselfensembling").GetSegmentation().AddEmptySegment()
                    self._parameterNode.GetNodeReference("fastsamselfensembling").GetSegmentation().GetSegment(segmentID).SetColor(colors[i-1][0]/255,colors[i-1][1]/255,colors[i-1][2]/255)
                    self._parameterNode.SetParameter("selfensemblingmask", segmentID)
                    slicer.util.updateSegmentBinaryLabelmapFromArray(mask,
                                                self._parameterNode.GetNodeReference("fastsamselfensembling"),
                                                self._parameterNode.GetParameter("selfensemblingmask"),
                                                self._parameterNode.GetNodeReference("fastsamInputVolume"))
        
    def get_next_click3D_torch_2(self, gt_semantic_seg):

        batch_points = []
        # dice_list = []

        points = self.torch.argwhere(gt_semantic_seg > 0)
        point = points[np.random.randint(len(points))].cpu()
        # import pdb; pdb.set_trace()
        batch_points.append(point)
        batch_points = self.torch.as_tensor(np.array(batch_points)).to(self.device)
        batch_labels = self.torch.as_tensor(np.array([1])).to(self.device)
        return batch_points, batch_labels
    def generaterandompoints(self, point):
        if (self.dimension == 3):
            x = random.randint(0,self.image_size)
            y = random.randint(0,self.image_size)
            z = random.randint(0,self.image_size)
            p = [[x,y,z]]
            return self.torch.as_tensor(np.array(p)).to(self.device)
        else :
            x = random.randint(0,self.image_size)
            y = random.randint(0,self.image_size)
            p = [[x,y]]
            return self.torch.as_tensor(np.array(p)).to(self.device)
        
    def postprocess_masks(self,low_res_masks, image_size, original_size):
        masks = self.torch.nn.functional.interpolate(
            low_res_masks,
            (image_size, image_size),
            mode="bilinear",
            align_corners=False,
            )
        masks = self.torch.nn.functional.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks  
         
    def backup_mask(self):
        self.mask = slicer.util.arrayFromSegmentBinaryLabelmap(self._parameterNode.GetNodeReference("fastsamSegmentation"),
                                                               self._parameterNode.GetParameter("fastsamCurrentSegment"))
        # self.logitsmask = slicer.util.arrayFromSegmentBinaryLabelmap(self._parameterNode.GetNodeReference("fastsamSegmentation"),
        #                                                 self._parameterNode.GetParameter("fastsamCurrentSegment"))

    def undo(self):
        if self.mask_backup is not None:
            self.mask = self.mask_backup.copy()
            self.pass_mask_to_slicer()

    @staticmethod
    def remove_small_regions(mask, area_thresh, mode):
        """Function from https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/utils/amg.py"""
        assert mode in ["holes", "islands"]
        correct_holes = mode == "holes"
        working_mask = (correct_holes ^ mask).astype(np.uint8)
        sitk_image = sitk.GetImageFromArray(working_mask)
        connected_components = sitk.ConnectedComponent(sitk_image, True)
        regions = sitk.RelabelComponent(connected_components)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(regions)
        regions = sitk.GetArrayFromImage(regions)
        n_labels = stats.GetNumberOfLabels() + 1
        sizes = np.array([stats.GetPhysicalSize(label) for label in stats.GetLabels()])
        small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]

        if len(small_regions) == 0:
            return mask
        fill_labels = [0] + small_regions
        if not correct_holes:
            fill_labels = [i for i in range(n_labels) if i not in fill_labels]
            # If every region is below threshold, keep largest
            if len(fill_labels) == 0:
                fill_labels = [int(np.argmax(sizes)) + 1]
        mask = np.isin(regions, fill_labels)
        return mask
