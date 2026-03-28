import torch
import torch.nn as nn
import torch.nn.functional as F

def spectral_norm_conv(in_channels, out_channels, kernel_size, stride, padding, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))

class ResBlock(nn.Module):
    def __init__(self, filters):
        super(ResBlock, self).__init__()
        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            spectral_norm_conv(filters, filters, 3, 1, 0, bias=False),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            spectral_norm_conv(filters, filters, 3, 1, 0, bias=False)
        )

    def forward(self, x):
        return x + self.net(x)

class ClassActivationMapping(nn.Module):
    def __init__(self, units, activation_fn):
        super(ClassActivationMapping, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap_linear = nn.Conv2d(units, units, 1, bias=False)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gmp_linear = nn.Conv2d(units, units, 1, bias=False)
        self.out = nn.Conv2d(units * 2, units, 1)
        self.activation = activation_fn

    def forward(self, x):
        # Global Avg/Max Pool + 1x1 Conv
        gap_y = self.gap_linear(self.gap(x))
        gap_m = self.gap_linear(x)
        gmp_y = self.gmp_linear(self.gmp(x))
        gmp_m = self.gmp_linear(x)
        
        # Concat gap_m and gmp_m along channel dimension
        # In MXNet code: mx.nd.concat(gap_m, gmp_m, dim=1)
        # Note: units was 2^downsample_layers * filters. 
        # The output of concat has units*2 channels.
        combined = torch.cat([gap_m, gmp_m], dim=1)
        x_out = self.activation(self.out(combined))
        
        # Concat gap_y and gmp_y for CAM features
        y_out = torch.cat([gap_y, gmp_y], dim=1)
        return x_out, y_out

class ResnetGenerator(nn.Module):
    def __init__(self, channels=3, filters=64, res_blocks=9, downsample_layers=2):
        super(ResnetGenerator, self).__init__()
        
        # Encoder
        model_enc = [
            nn.ReflectionPad2d(3),
            spectral_norm_conv(channels, filters, 7, 1, 0, bias=False),
            nn.ReLU(inplace=True)
        ]
        
        for i in range(downsample_layers):
            in_f = (2**i) * filters
            out_f = (2**(i+1)) * filters
            model_enc += [
                spectral_norm_conv(in_f, out_f, 3, 2, 1, bias=False),
                nn.ReLU(inplace=True)
            ]
            
        units = (2**downsample_layers) * filters
        for _ in range(res_blocks):
            model_enc.append(ResBlock(units))
            
        self.enc = nn.Sequential(*model_enc)
        self.cam = ClassActivationMapping(units, nn.ReLU(inplace=True))
        
        # Decoder
        model_dec = []
        for i in range(downsample_layers):
            in_f = (2**(downsample_layers - i)) * filters
            out_f = (2**(downsample_layers - i - 1)) * filters
            model_dec += [
                nn.Upsample(scale_factor=2, mode='nearest'),
                spectral_norm_conv(in_f, out_f, 3, 1, 1, bias=False),
                nn.ReLU(inplace=True)
            ]
            
        model_dec += [
            nn.ReflectionPad2d(3),
            spectral_norm_conv(filters, channels, 7, 1, 0, bias=False),
            nn.Tanh()
        ]
        self.dec = nn.Sequential(*model_dec)

    def forward(self, x):
        x, cam_y = self.cam(self.enc(x))
        return self.dec(x), cam_y

class PatchDiscriminator(nn.Module):
    def __init__(self, channels=3, filters=64, layers=3):
        super(PatchDiscriminator, self).__init__()
        
        model_enc = [
            spectral_norm_conv(channels, filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        for i in range(1, layers):
            in_f = min(2**(i-1), 8) * filters
            out_f = min(2**i, 8) * filters
            model_enc += [
                spectral_norm_conv(in_f, out_f, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            
        units = min(2**layers, 8) * filters
        in_f_final = min(2**(layers-1), 8) * filters
        model_enc += [
            spectral_norm_conv(in_f_final, units, 4, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        self.enc = nn.Sequential(*model_enc)
        self.cam = ClassActivationMapping(units, nn.LeakyReLU(0.2, inplace=True))
        self.dec = spectral_norm_conv(units, 1, 4, 1, 1, bias=False)

    def forward(self, x):
        x, cam_y = self.cam(self.enc(x))
        return self.dec(x), cam_y

if __name__ == "__main__":
    net_g = ResnetGenerator()
    net_d = PatchDiscriminator()
    dummy_input = torch.zeros((1, 3, 256, 256))
    
    with torch.no_grad():
        fake_out, gen_cam_y = net_g(dummy_input)
        print("Generator output shape:", fake_out.shape)
        print("Generator CAM shape:", gen_cam_y.shape)
        
        real_y, real_cam_y = net_d(dummy_input)
        print("Discriminator output shape:", real_y.shape)
        print("Discriminator CAM shape:", real_cam_y.shape)
