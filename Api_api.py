from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
import os
from ultralytics import YOLO
from skimage import measure
from io import BytesIO
from fastapi.responses import StreamingResponse

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize YOLO model
pt_name = 'best.pt'
pt_yv8n = YOLO(pt_name)

def get_lines(X1, Y1, X2, Y2):
    A = np.array([[X1, 1], [X2, 1]])
    B = np.array([Y1, Y2])
    ab = np.linalg.solve(A, B)
    a, b = ab[0], ab[1]
    return a, b

def array_corners(xys):
    x, y = np.zeros(4, np.uint32), np.zeros(4, np.uint32)
    x[0], y[0] = xys[0][0], xys[0][1]
    x[1], y[1] = xys[1][0], xys[1][1]
    x[2], y[2] = xys[2][0], xys[2][1]
    x[3], y[3] = xys[3][0], xys[3][1]

    lefts = []
    rghts = []
    for i in range(4):
        if x[i] < np.mean(x):
            lefts.append(i)
        else:
            rghts.append(i)
    if y[lefts[0]] > y[lefts[1]]:
        x1, y1 = x[lefts[1]], y[lefts[1]]
        x4, y4 = x[lefts[0]], y[lefts[0]]
    else:
        x1, y1 = x[lefts[0]], y[lefts[0]]
        x4, y4 = x[lefts[1]], y[lefts[1]]

    if y[rghts[0]] > y[rghts[1]]:
        x2, y2 = x[rghts[1]], y[rghts[1]]
        x3, y3 = x[rghts[0]], y[rghts[0]]
    else:
        x2, y2 = x[rghts[0]], y[rghts[0]]
        x3, y3 = x[rghts[1]], y[rghts[1]]
    
    return x1, y1, x2, y2, x3, y3, x4, y4

@app.post("/process-image")
async def process_images(car_image: UploadFile = File(...), logo_image: UploadFile = File(...)):
    try:
        # Read images directly in BGR format using cv2
        car_content = await car_image.read()
        logo_content = await logo_image.read()

        car_np = np.frombuffer(car_content, np.uint8)
        logo_np = np.frombuffer(logo_content, np.uint8)

        I = cv2.imdecode(car_np, cv2.IMREAD_COLOR)  # Reads in BGR
        I_logo = cv2.imdecode(logo_np, cv2.IMREAD_COLOR)  # Reads in BGR

        if I is None or I_logo is None:
            return Response(content="Failed to load images", status_code=400)

        # Detect license plates
        results = pt_yv8n(I)[0]
        boxes = results.obb

        if boxes is not None:
            for j in range(len(boxes)):
                xys = boxes.xyxyxyxy[0]
                x1, y1, x2, y2, x3, y3, x4, y4 = array_corners(xys)
                xy0 = boxes.xyxy[0]
                xbs, ybs, xbe, ybe = int(xy0[0]), int(xy0[1]), int(xy0[2]), int(xy0[3])

                I_rt = I[ybs:ybe, xbs:xbe].copy()
                xb1, yb1, xb2, yb2, xb3, yb3, xb4, yb4 = x1-xbs, y1-ybs, x2-xbs, y2-ybs, x3-xbs, y3-ybs, x4-xbs, y4-ybs

                if abs(yb1 + yb4 - yb3 - yb2) > 40:
                    a1, b1 = get_lines(xb1, yb1, xb2, yb2)
                    a2, b2 = get_lines(yb2, xb2, yb3, xb3)
                    a3, b3 = get_lines(xb3, yb3, xb4, yb4)
                    a4, b4 = get_lines(yb1, xb1, yb4, xb4)

                    for k1 in range(I_rt.shape[0]):
                        for k2 in range(I_rt.shape[1]):
                            if ~((k1 > a1*k2 + b1) and (k1 < a3*k2 + b3)):
                                I_rt[k1, k2, :] = 0
                            if ~((k2 < a2*k1 + b2) and (k2 > a4*k1 + b4)):
                                I_rt[k1, k2, :] = 0

                    I_rt = cv2.GaussianBlur(I_rt, (3, 3), 2)
                    I_rt = cv2.cvtColor(I_rt, cv2.COLOR_BGR2GRAY)
                    _, im_bw = cv2.threshold(I_rt, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                    labels = measure.label(im_bw)
                    props = measure.regionprops(labels)

                    area = [prop.area for prop in props]
                    if area:
                        max_value = max(area)
                        max_index = area.index(max_value) + 1
                        new_img = np.uint8(labels == max_index)*255
                        new_img = cv2.erode(new_img, np.ones((5, 5), np.uint8))
                        new_img = cv2.dilate(new_img, np.ones((5, 5), np.uint8))

                        pxsum = np.sum(new_img, 0)/255
                        pxsum = pxsum > 10

                        start = next((i for i in range(len(pxsum)) if pxsum[i]), 0)
                        finish = len(pxsum) - next((i for i in range(len(pxsum)) if pxsum[-i-1]), 0)

                        if abs(xb1 - start) > abs(xb4 - start):
                            xb1 = start
                        else:
                            xb4 = start

                        if abs(xb2 - finish) > abs(xb3 - finish):
                            xb2 = finish
                        else:
                            xb3 = finish

                        x1, y1, x2, y2, x3, y3, x4, y4 = xb1+xbs, yb1+ybs, xb2+xbs, yb2+ybs, xb3+xbs, yb3+ybs, xb4+xbs, yb4+ybs

                pts1 = np.float32([[0, 0], [I_logo.shape[1], 0], [I_logo.shape[1], I_logo.shape[0]], [0, I_logo.shape[0]]])
                pts2 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                result = cv2.warpPerspective(I_logo, matrix, (I.shape[1], I.shape[0]))

                mask00 = ~(cv2.GaussianBlur(result, (7, 7), 1) > 0)
                I = I*mask00 + result

        # Convert to JPEG without color space conversion
        _, buffer = cv2.imencode('.jpg', I)
        io_buffer = BytesIO(buffer)

        return StreamingResponse(io_buffer, media_type="image/jpeg")

    except Exception as e:
        return Response(content=str(e), status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
