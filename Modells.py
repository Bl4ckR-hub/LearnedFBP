import torch.nn as nn
import torch


class Clamper(nn.Module):
    """
    Wraps torch.clamp as an nn.Module for use inside nn.Sequential pipelines.

    Args:
        min_val (float): Lower bound of the clamp range.
        max_val (float): Upper bound of the clamp range.
    """
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min = min_val
        self.max = max_val
    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

class UNet_pre(nn.Module):
    """
    UNet for sinogram-domain preprocessing.

    Structurally identical to UNet but with adjusted output_padding in the
    bridge and decoder blocks to handle the non-square sinogram dimensions
    (1000 x 513). The final activation is a Sigmoid, so outputs are in [0, 1].

    Use this when the network operates on sinograms before reconstruction,
    e.g. as the preprocess_net in CompleteReconstruct.

    Args:
        in_channels (int): Number of input/output channels (typically 1).

    Input:  (B, in_channels, 1000, 513)
    Output: (B, in_channels, 1000, 513)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,0))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128)
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64, output_padding=(0,1))

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)


        return self.sigmoid(X)

class UNet(nn.Module):
    """
    Standard UNet for image-domain post-processing.

    4-level encoder-decoder architecture with skip connections. Designed for
    square image inputs (362 x 362). The final activation is a Sigmoid, so
    outputs are in [0, 1] — suitable as the post_processing_module in LearnableFBP.

    Channel progression: in → 64 → 128 → 256 → 512 → 1024 (bridge) → ... → in

    Args:
        in_channels (int): Number of input/output channels (typically 1).

    Input:  (B, in_channels, 362, 362)
    Output: (B, in_channels, 362, 362)
    """
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,1))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128, output_padding=(1,1))
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64)

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)

        return self.sigmoid(X)

class CNNBlock(nn.Module):
    """
    Basic convolutional building block: Conv → ReLU → Conv → ReLU.

    Applies two 3x3 convolutions with same-padding (no spatial downsampling).
    Used as the feature extraction unit inside EncoderBlock, DecoderBlock, and the bridge.

    Args:
        in_channels (int):  Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        )

    def forward(self, X):
        return self.layer(X)


class EncoderBlock(nn.Module):
    """
    UNet encoder step: CNNBlock followed by 2x2 MaxPooling.

    Returns both the pre-pool feature map (for the skip connection) and
    the pooled output (passed to the next encoder level).

    Args:
        in_channels (int):  Number of input channels.
        out_channels (int): Number of output channels after convolution.

    Returns:
        tuple[Tensor, Tensor]: (skip, pooled)
            - skip:   feature map before pooling, shape (B, out_channels, H, W)
            - pooled: downsampled output,          shape (B, out_channels, H/2, W/2)
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = CNNBlock(in_channels=in_channels, out_channels=out_channels)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, X):
        X_old = self.conv(X)  # for skip connections
        P = self.pool(X_old)
        return X_old, P


class DecoderBlock(nn.Module):
    """
    UNet decoder step: CNNBlock followed by transposed convolution (2x upsampling).

    Receives the concatenation of the skip connection and the upsampled tensor
    from the previous decoder level, processes it with a CNNBlock, then upsamples.

    Args:
        in_channels (int):   Number of input channels (skip + previous decoder output).
        out_channels1 (int): Intermediate channels after CNNBlock.
        out_channels2 (int): Output channels after upsampling.
        output_padding (int or tuple): Passed to ConvTranspose2d to resolve spatial
            size mismatches caused by odd input dimensions. Default: 0.
    """
    def __init__(self, in_channels, out_channels1, out_channels2, output_padding=0):
        super().__init__()
        self.conv = CNNBlock(in_channels=in_channels, out_channels=out_channels1)
        self.upsample = nn.ConvTranspose2d(in_channels=out_channels1, out_channels=out_channels2, kernel_size=2,
                                           stride=2, output_padding=output_padding)

    def forward(self, X):
        X = self.conv(X)
        return self.upsample(X)


####################

class TrainableFourierSeries(nn.Module):
    def __init__(self, freqs, init_filter, L=50):
        super().__init__()

        # Step23: Store Fourier Series Coefficients
        a, b, a_0 = self.cos_sin_coeffs(init_filter, L)

        # Step 3: Normalize the frequencies
        freqs_range = freqs.max() - freqs.min()
        normalized_freqs = (freqs - freqs.min()) / freqs_range

        # Step 4: Create cos and sin matrices
        i = torch.arange(1, L + 1, dtype=normalized_freqs.dtype)
        cos_terms = torch.cos(2 * torch.pi * normalized_freqs.unsqueeze(1) * i)
        sin_terms = torch.sin(2 * torch.pi * normalized_freqs.unsqueeze(1) * i)

        self.register_buffer('cos_sin_stuff', torch.cat([cos_terms, sin_terms], dim=1))
        self.coeffs = torch.nn.Parameter(torch.cat([a, b]), requires_grad=True)
        self.const = torch.nn.Parameter(a_0, requires_grad=True)

    def forward(self, X):
        filter = self.const + torch.matmul(self.cos_sin_stuff, self.coeffs)
        filter = torch.fft.fftshift(filter)
        return filter.unsqueeze(0).unsqueeze(0).unsqueeze(0) # (1,1,1,513)

    def cos_sin_coeffs(self, f, L):
        N = f.shape[0]

        # Step 1: Perform the FFT
        fft_result = torch.fft.fft(f)

        # Step 2: Extract the coefficients
        a0 = fft_result[0].real / N

        # Initialize lists for a_i and b_i
        a_i_list = []
        b_i_list = []

        for i in range(1, L + 1):
            # a_i is derived from the real part
            ai = (2 / N) * fft_result[i].real
            a_i_list.append(ai.item())

            # b_i is derived from the imaginary part
            bi = (-2 / N) * fft_result[i].imag
            b_i_list.append(bi.item())

        a_i_list = torch.Tensor(a_i_list)
        b_i_list = torch.Tensor(b_i_list)

        return a_i_list, b_i_list, a0


#############
class LearnableWindow(nn.Module):
    def __init__(self, init_tensor=torch.ones(513)):
        super().__init__()
        self.weights = nn.Parameter(init_tensor)  # or random init
    def forward(self, x):
        return self.weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)
##############

class LearnableWindowII(nn.Module):
    def __init__(self, init_tensor=torch.ones((1000,513))):
        super(LearnableWindowII, self).__init__()
        self.weights = nn.Parameter(init_tensor)
    def forward(self, x):
        return self.weights.unsqueeze(0).unsqueeze(0) # (1,1,1000,513)
    

class UNet_no_activation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,1))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128, output_padding=(1,1))
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64)

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)

        return X
    
class UNet_pre_no_activation(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.encoder_block1 = EncoderBlock(in_channels=in_channels, out_channels=64)
        self.encoder_block2 = EncoderBlock(in_channels=64, out_channels=128)
        self.encoder_block3 = EncoderBlock(in_channels=128, out_channels=256)
        self.encoder_block4 = EncoderBlock(in_channels=256, out_channels=512)

        self.bridge = nn.Sequential(
            CNNBlock(in_channels=512, out_channels=1024),
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, output_padding=(1,0))
        )

        self.decoder_block1 = DecoderBlock(in_channels=1024, out_channels1=512, out_channels2=256)
        self.decoder_block2 = DecoderBlock(in_channels=512, out_channels1=256, out_channels2=128)
        self.decoder_block3 = DecoderBlock(in_channels=256, out_channels1=128, out_channels2=64, output_padding=(0,1))

        self.last1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1)
        self.last2 = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=1)

    def forward(self, X):
        X_old1, X = self.encoder_block1(X)
        X_old2, X = self.encoder_block2(X)
        X_old3, X = self.encoder_block3(X)
        X_old4, X = self.encoder_block4(X)


        X = self.bridge(X)
        X = torch.cat([X_old4, X], dim=1)  # I exprect the X to have the shape ( (batch) x C x H x W)
        X = self.decoder_block1(X)
        X = torch.cat([X_old3, X], dim=1)
        X = self.decoder_block2(X)
        X = torch.cat([X_old2, X], dim=1)
        X = self.decoder_block3(X)
        X = torch.cat([X_old1, X], dim=1)
        X = self.last1(X)
        X = self.last2(X)


        return X