FROM agrigorev/model-2025-hairstyle:v1

# install dependencies
RUN pip install pillow numpy onnxruntime

# copy lambda code
COPY lambda_function.py .

# set lambda handler
CMD ["lambda_function.lambda_handler"]
