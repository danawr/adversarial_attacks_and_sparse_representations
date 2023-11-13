import torch
import torch.nn.functional as F
from itertools import cycle

class SoftTh(torch.nn.Module):
    def __init__(self):
        super(SoftTh, self).__init__()
        # self.eta = torch.nn.Parameter(torch.randn(1))
        self.eta = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.sign(x) * torch.maximum(torch.zeros(1).cuda(), torch.abs(x) - self.eta)

class ListaBlock(torch.nn.Module):
    """One iteration of LISTA."""

    def __init__(self, measurement_dim, input_dim):
        super(ListaBlock, self).__init__()
        self.W_1 = torch.nn.Linear(measurement_dim, input_dim, bias=False)
        self.W_2 = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.soft_th = SoftTh()

    def forward(self, x, y):
        return self.soft_th(self.W_1(y) + self.W_2(x))


class Lista(torch.nn.Module):
    """y = Ax : restore x from y.
       y: measurement, of length measurement_dim (m).
       x: input, of len input_dim (n).
       Holds num_iterations x 2 matrices, and num_iterations thresholds."""
    def __init__(self, measurement_dim, input_dim, num_iterations):
        super(Lista, self).__init__()
        self.layers = torch.nn.ModuleList([ListaBlock(measurement_dim, input_dim) for _ in range(num_iterations)])

    def forward(self, y, iterations=None):
        if iterations:
            max_iteration = iterations
        else:
            max_iteration = len(self.layers)
        x = torch.zeros((y.shape[0], 500)).to(y.device)
        for layer in self.layers[:max_iteration]:
            x = layer(x, y)
        return x


class SharedWeightsLista(torch.nn.Module):
    """y = Ax : restore x from y.
       y: measurement, of length measurement_dim (m).
       x: input, of len input_dim (n).
       Holds two matrices, and num_iterations thresholds,
       corresponding to the original (unfolded) Lista."""
    def __init__(self, measurement_dim, input_dim, num_iterations):
        super(SharedWeightsLista, self).__init__()
        self.W_1 = torch.nn.Linear(measurement_dim, input_dim, bias=False)
        self.W_2 = torch.nn.Linear(input_dim, input_dim, bias=False)
        self.shrinkers = torch.nn.ModuleList([SoftTh() for _ in range(num_iterations)])

    def forward(self, x, y):
        y = self.W_1(y)
        for shrinker in self.shrinkers:
            x = shrinker(y + self.W_2(x))
        return x


class LISTAConvDict(torch.nn.Module):
    """
    LISTA ConvDict encoder based on paper:
    https://arxiv.org/pdf/1711.00328.pdf
    """
    def __init__(self, num_input_channels=3, num_output_channels=3,
                 kc=64, ks=7, ista_iters=3, iter_weight_share=True,
                 share_decoder=False):

        super(LISTAConvDict, self).__init__()
        self._ista_iters = ista_iters
        self._layers = 1 if iter_weight_share else ista_iters

        def build_softthrsh():
            return SoftshrinkTrainable(
                torch.nn.Parameter(0.1 * torch.ones(1, kc), requires_grad=True)
            )

        self.softthrsh0 = build_softthrsh()
        if iter_weight_share:
            self.softthrsh1 = torch.nn.ModuleList([self.softthrsh0
                                             for _ in range(self._layers)])
        else:
            self.softthrsh1 = torch.nn.ModuleList([build_softthrsh()
                                             for _ in range(self._layers)])

        def build_conv_layers(in_ch, out_ch, count):
            """Conv layer wrapper
            """
            return torch.nn.ModuleList(
                [torch.nn.Conv2d(in_ch, out_ch, ks,
                           stride=1, padding=ks//2, bias=False) for _ in
                 range(count)])

        # Encoder
        self.encode_conv0 = build_conv_layers(num_input_channels, kc, 1)[0]
        if iter_weight_share:
            self.encode_conv1 = torch.nn.ModuleList(self.encode_conv0 for _ in
                                              range(self._layers))
        else:
            self.encode_conv1 = build_conv_layers(num_input_channels, kc,
                                                  self._layers)

        self.decode_conv0 = build_conv_layers(kc, num_input_channels,
                                              self._layers if not share_decoder
                                              else 1)
        # Decoder
        if share_decoder:
            self.decode_conv1 = self.decode_conv0[0]
            self.decode_conv0 = torch.nn.ModuleList([self.decode_conv0[0] for _ in
                                               range(self._layers)])
        else:
            self.decode_conv1 = build_conv_layers(kc, num_output_channels, 1)[0]

    @property
    def ista_iters(self):
        """Amount of ista iterations
        """
        return self._ista_iters

    @property
    def layers(self):
        """Amount of layers with free parameters.
        """
        return self._layers

    @property
    def conv_dictionary(self):
        """Get the weights of convolutoinal dictionary
        """
        return self.decode_conv1.weight#.data

    def forward_enc(self, inputs):
        """Conv LISTA forwrd pass
        """
        csc = self.softthrsh0(self.encode_conv0(inputs))

        for _itr, lyr in\
            zip(range(self._ista_iters),
                    cycle(range(self._layers))):

            sc_residual = self.encode_conv1[lyr](
                inputs - self.decode_conv0[lyr](csc)
            )
            csc = self.softthrsh1[lyr](csc + sc_residual)
        return csc

    def forward_enc_generataor(self, inputs):
        """forwar encoder generator
        Use for debug and anylize model.
        """
        csc = self.softthrsh0(self.encode_conv0(inputs))

        for itr, lyr in\
            zip(range(self._ista_iters),
                    cycle(range(self._layers))):

            sc_residual = self.encode_conv1[lyr](
                inputs - self.decode_conv0[lyr](csc)
            )
            csc = self.softthrsh1[lyr](csc + sc_residual)
            yield csc, sc_residual, itr

    def forward_dec(self, csc):
        """
        Decoder foward  csc --> input
        """
        return self.decode_conv1(csc)

    #pylint: disable=arguments-differ
    def forward(self, inputs):
        csc = self.forward_enc(inputs)
        outputs = self.forward_dec(csc)
        return outputs, csc


class SCClassifier(torch.nn.Module):
    def __init__(self, sparse_coder: LISTAConvDict, code_dim, num_classes=10):

        super(SCClassifier, self).__init__()

        self.num_classes = num_classes
        self.code_dim = code_dim
        self._ista_iters = sparse_coder._ista_iters
        self._layers = sparse_coder._layers

        self.softthrsh0 = sparse_coder.softthrsh0
        self.softthrsh1 = sparse_coder.softthrsh1

        # Encoder
        self.encode_conv0 = sparse_coder.encode_conv0
        self.encode_conv1 = sparse_coder.encode_conv1
        self.decode_conv0 = sparse_coder.decode_conv0

        # Decoder
        self.decode_conv1 = sparse_coder.decode_conv1

        # Classifier
        # self.pre_classifier = torch.nn.Linear(code_dim, 128)
        # self.non_linearity = torch.nn.ReLU()
        # self.classifier = torch.nn.Linear(128, num_classes)
        self.classifier = torch.nn.Linear(code_dim, num_classes)


    def forward_enc(self, inputs):
        """Conv LISTA forwrd pass
        """
        csc = self.softthrsh0(self.encode_conv0(inputs))

        for _itr, lyr in\
            zip(range(self._ista_iters),
                    cycle(range(self._layers))):

            sc_residual = self.encode_conv1[lyr](
                inputs - self.decode_conv0[lyr](csc)
            )
            csc = self.softthrsh1[lyr](csc + sc_residual)
        return csc


    def forward_dec(self, csc):
        """
        Decoder foward  csc --> input
        """
        return self.decode_conv1(csc)


    # def sprse_code_to_logits(self, sparse_code):
    #     logits = self.pre_classifier(sparse_code.flatten(start_dim=1))
    #     logits = self.non_linearity(logits)
    #     logits = self.classifier(logits)
    #     return logits

    def forward(self, inputs):
        sparse_code = self.forward_enc(inputs)
        # logits = self.sprse_code_to_logits(sparse_code)
        logits = self.classifier(sparse_code.flatten(start_dim=1))
        # rec = self.forward_dec(sparse_code)
        return logits#, rec, sparse_code

    def classify_code(self, sparse_code):
        logits = self.classifier(sparse_code.flatten(start_dim=1))
        # logits = self.sprse_code_to_logits(sparse_code)
        rec = self.forward_dec(sparse_code)
        return logits, rec, sparse_code


    def conv_dictionary(self):
        """Get the weights of convolutoinal dictionary
        """
        return self.decode_conv1.weight#.data

class SoftshrinkTrainable(torch.nn.Module):
    """
    Learn threshold (lambda)
    """

    def __init__(self, _lambd):
        super(SoftshrinkTrainable, self).__init__()
        self._lambd = _lambd

    @property
    def thrshold(self):
        return self._lambd
#        self._lambd.register_hook(print)

    def forward(self, inputs):
        """ sign(inputs) * (abs(inputs)  - thrshold)"""
        _inputs = inputs
        _lambd = self._lambd.clamp(0).unsqueeze(-1).unsqueeze(-1)
        result = torch.sign(_inputs) * (F.relu(torch.abs(_inputs) - _lambd))
        return result

    def _forward(self, inputs):
        """ sign(inputs) * (abs(inputs)  - thrshold)"""
        _lambd = self._lambd.clamp(0)
        pos = (inputs - _lambd.unsqueeze(-1).unsqueeze(-1))
        neg = ((-1) * inputs - _lambd.unsqueeze(-1).unsqueeze(-1))
        return (pos.clamp(min=0) - neg.clamp(min=0))
