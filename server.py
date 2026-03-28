import io
import re
import sys
import argparse
import http.server
import cgi
import torch
import numpy as np
from PIL import Image
from dataset import reconstruct_color
from pix2pix_gan import ResnetGenerator
import torchvision.transforms.functional as TF

class CycleGANHandler(http.server.BaseHTTPRequestHandler):
    _path_pattern = re.compile("^(/[^?\s]*)(\?\S*)?$")

    def do_POST(self):
        self._handle_request()
        sys.stdout.flush()
        sys.stderr.flush()

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST")
        self.send_header("Access-Control-Allow-Headers", "Keep-Alive,User-Agent,Authorization,Content-Type")
        super(CycleGANHandler, self).end_headers()

    def _handle_request(self):
        m = self._path_pattern.match(self.path)
        if not m or m.group(1) != "/cycle_gan/fake":
            self.send_error(http.HTTPStatus.NOT_FOUND)
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers["Content-Type"]
            }
        )

        if "real" not in form:
            self.send_error(http.HTTPStatus.BAD_REQUEST)
            return

        # 1. Load Image from POST data
        image_data = form["real"].value
        img = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 2. Preprocess
        real = TF.resize(img, self.server.resize, transforms.InterpolationMode.BICUBIC)
        real = TF.to_tensor(real)
        real = TF.normalize(real, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        real = real.unsqueeze(0).to(self.server.device)

        # 3. Inference
        with torch.no_grad():
            self.server.net.eval()
            fake, _ = self.server.net(real)
        
        # 4. Post-process & Encode to PNG
        out_np = reconstruct_color(fake[0])
        out_pil = Image.fromarray(out_np)
        
        out_buffer = io.BytesIO()
        out_pil.save(out_buffer, format="PNG")
        out_bytes = out_buffer.getvalue()

        # 5. Send Response
        self.protocol_version = "HTTP/1.1"
        self.send_response(http.HTTPStatus.OK)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Disposition", 'attachment; filename="fake.png"')
        self.send_header("Content-Length", str(len(out_bytes)))
        self.end_headers()
        self.wfile.write(out_bytes)

def run_server():
    parser = argparse.ArgumentParser(description="PyTorch CycleGAN demo server.")
    parser.add_argument("--reversed", help="reverse transformation (B to A)", action="store_true")
    parser.add_argument("--model", help="model prefix (default: vangogh2photo)", type=str, default="vangogh2photo")
    parser.add_argument("--resize", help="image resize size (default: 256)", type=int, default=256)
    parser.add_argument("--addr", help="address (default: 0.0.0.0)", type=str, default="0.0.0.0")
    parser.add_argument("--port", help="port (default: 8080)", type=int, default=8080)
    parser.add_argument("--device", help="cuda, mps, or cpu", type=str, default=None)
    args = parser.parse_args()

    # Device detection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Server using device: {device}")

    print("Loading model...", flush=True)
    net = ResnetGenerator().to(device)
    suffix = "gen_ba" if args.reversed else "gen_ab"
    model_path = f"model/{args.model}.{suffix}.pth"
    
    try:
        net.load_state_dict(torch.load(model_path, map_location=device))
        net.eval()
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return

    # Create server
    server_address = (args.addr, args.port)
    httpd = http.server.HTTPServer(server_address, CycleGANHandler)
    
    # Store settings in server object
    httpd.resize = args.resize
    httpd.device = device
    httpd.net = net
    
    print(f"Starting server on {args.addr}:{args.port}...")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server()
