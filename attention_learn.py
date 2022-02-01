import numpy as np
import scipy.io
import json
import glob
import os
import cv2

CORE_PATH = "lsp_dataset"

def load_and_understand_mat_file():
    image_paths = glob.iglob(os.path.join(CORE_PATH, "images", "*.jpg"))
    test_images = list(image_paths)[:5]
    f = scipy.io.loadmat(CORE_PATH + "/joints.mat")
    joints = f['joints'].T
    for test_image in test_images:
        basename = os.path.basename(test_image)
        index = int(basename[2:6]) - 1

        data = joints[index]
        right_ankle = data[0]
        right_knee = data[1]
        right_hip = data[2]

        """
        Find right ankle, right knee, right hip
        http://sam.johnson.io/research/lsp.html
        """
        ankle_x, ankle_y = int(right_ankle[0]), int(right_ankle[1])
        knee_x, knee_y = int(right_knee[0]), int(right_knee[1])
        hip_x, hip_y = int(right_hip[0]), int(right_hip[1])

        img = cv2.imread(test_image)
        print(img.shape)

        circle = cv2.circle(
            img,
            (ankle_x, ankle_y),
            4,
            (255, 0, 0),
            2
        )

        circle = cv2.circle(
            circle,
            (knee_x, knee_y),
            4,
            (255, 0, 0),
            2
        )

        circle = cv2.circle(
            circle,
            (hip_x, hip_y),
            4,
            (255, 0, 0),
            2
        )

        cv2.imshow(basename, circle)
        cv2.imwrite(f"evaluated_images/{basename}", circle)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


load_and_understand_mat_file()
