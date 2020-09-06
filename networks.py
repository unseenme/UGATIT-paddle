import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D, Pool2D, BatchNorm, Linear,InstanceNorm,PRelu,SpectralNorm,Sequential
import paddle.fluid.dygraph.nn as nn


class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2d(3),
                      Conv2D(num_channels=input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      ReLU(True)]
        self.conv1=Conv2D(input_nc, ngf, 7)
        self.instance_norm=InstanceNorm(ngf)

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2d(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          ReLU(True)]
        mult = 2**0
        self.conv2=Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False)
        self.instance_norm1=InstanceNorm(ngf * mult * 2)
        mult = 2**1
        self.conv3=Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False)

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]
        self.renetblock=ResnetBlock(ngf * mult, use_bias=False)

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 = Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True,act='relu')

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu')
                  ]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False,act='relu'),
                  Linear(ngf * mult, ngf * mult, bias_attr=False,act='relu')
                  ]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(),
                         ReflectionPad2d(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False,act='relu'),
                         ILN(int(ngf * mult / 2))]

        UpBlock2 += [ReflectionPad2d(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False,act='tanh')]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x)
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1])
        gap_logit = self.gap_fc(gap)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])
        gap = x * gap_weight
        gmp =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])     
        gmp = x * gmp_weight
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.conv1x1(x)
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        if self.light:
            x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg')
            x_=fluid.layers.reshape(x_, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        else:
            x_=fluid.layers.reshape(x, shape=[x.shape[0], -1])
            x_ = self.FC(x_)
        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)
        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim),
                       ReLU(True)]

        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=use_bias),
                       InstanceNorm(dim)]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=1, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.conv1(x)
        out = self.norm1(out, gamma, beta)
        out=fluid.layers.relu(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)
        return out + x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, in_channels, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho =self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.9))

    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input, gamma, beta):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1 - self.rho)*out_ln
        out=out*fluid.layers.unsqueeze(fluid.layers.unsqueeze(gamma,2),3)+fluid.layers.unsqueeze(fluid.layers.unsqueeze(beta,2),3)
        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, in_channels, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
        self.gamma = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=True,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
        
    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1 - self.rho)*out_ln
        out = out * self.gamma + self.beta
        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2d(1),
                 Spectralnorm(Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True,act='leaky_relu'))]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2d(1),
                      Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True,act='leaky_relu'))]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2d(1),
                  Spectralnorm(Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True,act='leaky_relu')),]        

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 =Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.pad=ReflectionPad2d(1)
        self.conv = Spectralnorm(Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))   
        self.model = Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='avg')(x) #[1, 2048, 1, 1]
        gap=fluid.layers.reshape(gap, shape=[x.shape[0], -1]) 
        gap_logit = self.gap_fc(gap)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[0])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[3])   
        gap = x * gap_weight
        gmp =Pool2D(pool_size=x.shape[-1],pool_stride=x.shape[-1],pool_type='max')(x)
        gmp=fluid.layers.reshape(gmp, shape=[x.shape[0], -1])        
        gmp_logit = self.gmp_fc(gmp)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[0])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[3])          
        gmp = x * gmp_weight
        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x=fluid.layers.leaky_relu(self.conv1x1(x))
        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)
        x = self.pad(x)
        out = self.conv(x)
        return out, cam_logit, heatmap


class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale)
        return out


class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w


class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out


class Spectralnorm(fluid.dygraph.Layer):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = nn.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")


class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace=inplace

    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.relu(x))
            return x
        else:
            y=fluid.layers.relu(x)
            return y
