import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os


class VideoGradCAM:

    def __init__(self, model):

        self.model = model

        self.gradients = None
        self.activations = None

        # EfficientNet last convolution layer
        self.target_layer = (
            self.model
            .backbone
            .features[-1]
        )

        self._register_hooks()



    ################################################
    # HOOKS
    ################################################

    def _register_hooks(self):


        def forward_hook(module, inp, out):

            self.activations = out



        def backward_hook(module, grad_in, grad_out):

            self.gradients = grad_out[0]



        self.target_layer.register_forward_hook(
            forward_hook
        )


        self.target_layer.register_full_backward_hook(
            backward_hook
        )



    ################################################
    # GENERATE CAM
    ################################################

    def generate(self, x):

        """
        Input:

        x:
        (B,T,C,H,W)


        Output:

        CAM:
        (T,H,W)

        """


        device = x.device


        ################################################
        # SAVE STATE
        ################################################

        was_training = self.model.training



        ################################################
        # IMPORTANT FIX
        # RNN BACKWARD NEEDS TRAIN MODE
        ################################################

        self.model.train()



        self.model.zero_grad(
            set_to_none=True
        )


        self.gradients = None
        self.activations = None



        ################################################
        # FORWARD
        ################################################

        logits, _ = self.model(
            x
        )


        # FAKE probability
        score = torch.softmax(
            logits,
            dim=1
        )[:,1]


        score.backward()



        ################################################
        # RESTORE MODEL
        ################################################

        if not was_training:

            self.model.eval()



        if self.gradients is None:

            raise RuntimeError(
                "GradCAM gradients missing"
            )


        if self.activations is None:

            raise RuntimeError(
                "GradCAM activations missing"
            )



        ################################################
        # ACTIVATION SHAPE
        ################################################

        """
        EfficientNet output:

        (B*T,C,H,W)

        """


        gradients = self.gradients

        activations = self.activations



        BT,C,H,W = activations.shape



        weights = gradients.mean(
            dim=(2,3),
            keepdim=True
        )



        cam = (
            weights *
            activations
        ).sum(
            dim=1
        )



        cam = F.relu(
            cam
        )



        ################################################
        # NORMALIZE
        ################################################

        cam = cam.detach()


        cam = cam.view(
            -1,
            H,
            W
        )


        cam_min = cam.min()

        cam_max = cam.max()


        if cam_max-cam_min > 1e-8:

            cam = (
                cam-cam_min
            ) / (
                cam_max-cam_min
            )


        cam = cam.cpu().numpy()



        return cam




    ################################################
    # TOP SUSPICIOUS FRAMES
    ################################################

    def suspicious_frames(
        self,
        cam,
        top_k=5
    ):


        scores=[]


        for i,frame_cam in enumerate(cam):


            score=float(
                frame_cam.mean()
            )


            scores.append(

                {
                    "frame":i+1,
                    "score":round(
                        score,
                        4
                    )
                }

            )



        scores.sort(
            key=lambda x:x["score"],
            reverse=True
        )


        return scores[:top_k]



    ################################################
    # SAVE HEATMAPS
    ################################################

    def save_heatmaps(
        self,
        cam,
        frames,
        output_dir
    ):


        os.makedirs(
            output_dir,
            exist_ok=True
        )


        paths=[]


        for idx,c in enumerate(cam):


            heat=cv2.resize(
                c,
                (
                    frames[idx].shape[1],
                    frames[idx].shape[0]
                )
            )


            heatmap=cv2.applyColorMap(
                np.uint8(
                    heat*255
                ),
                cv2.COLORMAP_JET
            )


            overlay=cv2.addWeighted(
                frames[idx],
                0.6,
                heatmap,
                0.4,
                0
            )


            path=os.path.join(
                output_dir,
                f"frame_{idx+1}.jpg"
            )


            cv2.imwrite(
                path,
                overlay
            )


            paths.append(path)



        return paths