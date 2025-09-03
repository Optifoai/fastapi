from fastapi import FastAPI, File, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from rembg import remove
from PIL import Image
import io

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (replace with specific domains in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

@app.post("/remove-background/")
async def remove_background(file: UploadFile = File(...)):
    # Read the uploaded image
    input_image = Image.open(file.file)

    # Remove the background (output will have transparent background)
    output_image = remove(input_image)

    # Save the output image with transparent background to a bytes buffer
    output_buffer = io.BytesIO()
    output_image.save(output_buffer, format="PNG")  # Save as PNG to preserve transparency
    output_buffer.seek(0)  # Move the buffer's cursor to the beginning

    # Return the image as a response with transparent background
    return Response(content=output_buffer.getvalue(), media_type="image/png")

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
