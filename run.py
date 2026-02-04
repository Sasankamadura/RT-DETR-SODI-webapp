import uvicorn
import os
import sys

# Ensure we can import backend
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("Starting RT-DETR Backend...")
    print("http://localhost:8000")
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)



#J.M Sasanka Madhura Bandara
#20220496
#w1998841