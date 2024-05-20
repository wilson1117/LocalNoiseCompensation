from .metric import Metric
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity

class AttackMetric(Metric):
    def __init__(self):
        super(AttackMetric, self).__init__()
        self.lpips_loss_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
        self.ssim_loss_fn = StructuralSimilarityIndexMeasure(data_range=255.0)
        self.psnr_loss_fn = PeakSignalNoiseRatio(data_range=255.0)

    def reset(self):
        self.records = []
        self.best = None

    def __call__(self, dummy_imgs, target_imgs):
        psnr = self.psnr_loss_fn(dummy_imgs, target_imgs).item()
        ssim = self.ssim_loss_fn(dummy_imgs, target_imgs).item()
        if dummy_imgs.shape[1] == 1:
            dummy_imgs = dummy_imgs.repeat(1, 3, 1, 1)
            target_imgs = target_imgs.repeat(1, 3, 1, 1)

        lpips = self.lpips_loss_fn(dummy_imgs, target_imgs).item()

        if self.best is None or lpips < self.records[self.best][2]:
            self.best = len(self.records)

        self.current_record = (psnr, ssim, lpips)

        self.records.append(self.current_record)

        return psnr, ssim, lpips
            
    def output_best(self):
        return dict(
            best_round=self.best,
            psnr=self.records[self.best][0],
            ssim=self.records[self.best][1],
            lpips=self.records[self.best][2]
        )


    def __str__(self):
        return ""

    @staticmethod
    def get_log_title():
        return "PSNR,SSIM,LPIPS"

    def log(self):
        return f"{self.current_record[0]},{self.current_record[1]},{self.current_record[2]}"