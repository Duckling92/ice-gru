import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RnnBlock(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_parallel,
        num_layers,
        rnn_type="LSTM",
        bidir=False,
        residual=False,
        batch_first=True,
        learn_init=False,
        dropout=0.0,
    ):

        super(RnnBlock, self).__init__()

        self._batch_first = batch_first
        self.hidden_size = n_out
        self.n_in = n_in
        self.residual = residual
        self.bidir = bidir
        self.n_dirs = 2 if bidir else 1
        self.n_layers = num_layers
        self.learn_init = learn_init
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.par_RNNs = nn.ModuleList()
        if learn_init:
            self.init_hidden_states = nn.ParameterList()
            if rnn_type == "LSTM":
                self.init_cell_states = nn.ParameterList()

        if rnn_type == "LSTM":
            for i_par in range(n_parallel):
                self.par_RNNs.append(
                    nn.LSTM(
                        input_size=n_in,
                        hidden_size=n_out,
                        bidirectional=bidir,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        dropout=self.dropout,
                    )
                )
                if self.learn_init:
                    n_parameters = self.n_dirs * self.hidden_size * self.n_layers
                    self.init_hidden_states.append(
                        nn.Parameter(torch.empty(n_parameters).normal_(mean=0, std=1.0))
                    )
                    self.init_cell_states.append(
                        nn.Parameter(torch.empty(n_parameters).normal_(mean=0, std=1.0))
                    )

        elif rnn_type == "GRU":
            for i_par in range(n_parallel):
                self.par_RNNs.append(
                    nn.GRU(
                        input_size=n_in,
                        hidden_size=n_out,
                        bidirectional=bidir,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        dropout=self.dropout,
                    )
                )
                if self.learn_init:
                    n_parameters = self.n_dirs * self.hidden_size * self.n_layers
                    self.init_hidden_states.append(
                        nn.Parameter(torch.empty(n_parameters).normal_(mean=0, std=1.0))
                    )
                    # self.init_cell_states.append(nn.Parameter(torch.empty(n_parameters).normal_(mean=0,std=1.0)))
        elif rnn_type == "Vanilla":
            for i_par in range(n_parallel):
                self.par_RNNs.append(
                    nn.RNN(
                        input_size=n_in,
                        hidden_size=n_out,
                        bidirectional=bidir,
                        num_layers=num_layers,
                        batch_first=batch_first,
                        dropout=self.dropout,
                    )
                )
                if self.learn_init:
                    n_parameters = self.n_dirs * self.hidden_size * self.n_layers
                    self.init_hidden_states.append(
                        nn.Parameter(torch.empty(n_parameters).normal_(mean=0, std=1.0))
                    )

        else:
            raise KeyError("UNKNOWN RNN TYPE (%s) PASSED TO MAKEMODEL" % (rnn_type))

    def forward(self, seq, lengths, device=None):
        if self._batch_first:
            longest_seq = seq.shape[1]
            batch_size = seq.shape[0]
        else:
            longest_seq = seq.shape[0]
            batch_size = seq.shape[1]

        seq_packed = pack_padded_sequence(seq, lengths, batch_first=self._batch_first)

        for i_par in range(len(self.par_RNNs)):
            if self.learn_init:

                if self.rnn_type == "LSTM":
                    # Dont know why, but the .contiguous call is needed, else an error is thrown
                    hidden = (
                        self.init_hidden_states[i_par]
                        .view(self.n_layers * self.n_dirs, 1, -1)
                        .expand(-1, batch_size, -1)
                        .contiguous()
                    )
                    cell = (
                        self.init_cell_states[i_par]
                        .view(self.n_layers * self.n_dirs, 1, -1)
                        .expand(-1, batch_size, -1)
                        .contiguous()
                    )
                    h = (hidden, cell)
                elif self.rnn_type in ["GRU", "Vanilla"]:
                    h = (
                        self.init_hidden_states[i_par]
                        .view(self.n_layers * self.n_dirs, 1, -1)
                        .expand(-1, batch_size, -1)
                        .contiguous()
                    )

            else:
                h = self.init_hidden(batch_size, self.par_RNNs[i_par], device)

            self.par_RNNs[i_par].flatten_parameters()
            seq_par, h_par = self.par_RNNs[i_par](seq_packed, h)
            seq_par_post, lengths = pad_packed_sequence(
                seq_par, batch_first=True, total_length=longest_seq
            )

            if self.rnn_type == "LSTM":
                h_out = h_par[0].view(
                    self.n_layers, self.n_dirs, batch_size, self.hidden_size
                )
            if self.rnn_type in ["GRU", "Vanilla"]:
                h_out = h_par.view(
                    self.n_layers, self.n_dirs, batch_size, self.hidden_size
                )

            if self.bidir:
                h_out = torch.cat(
                    (
                        h_out[self.n_layers - 1, 0, :, :],
                        h_out[self.n_layers - 1, 1, :, :],
                    ),
                    axis=-1,
                )
            else:
                h_out = h_out[
                    self.n_layers - 1, 0, :, :
                ]  # torch.cat((, h_out[self.n_layers-1, 1, :, :]), axis=-1)

            if self.residual:
                seq_par_post = seq_par_post + seq

            if i_par == 0:
                x = h_out
                seq_out = seq_par_post
            else:
                x = torch.cat((x, h_out), -1)
                seq_out = torch.cat((seq_out, seq_par_post), -1)

        return seq_out, x.squeeze(0)

    def init_hidden(self, batch_size, layer, device):
        hidden_size = int(layer.weight_ih_l0.shape[0] / 4)

        if self.rnn_type == "LSTM":
            output = (
                torch.zeros(
                    self.n_dirs * self.n_layers, batch_size, hidden_size, device=device
                ),
                torch.zeros(
                    self.n_dirs * self.n_layers, batch_size, hidden_size, device=device
                ),
            )
        elif self.rnn_type == "GRU":
            output = torch.zeros(
                self.n_dirs * self.n_layers, batch_size, hidden_size, device=device
            )

        return output


class MakeModel(nn.Module):
    """A modular PyTorch model builder
    """

    def __init__(self, arch_dict, device=None):
        super(MakeModel, self).__init__()
        self.mods = make_model_architecture(arch_dict)
        self.layer_names = get_layer_names(arch_dict)
        self.arch_dict = arch_dict
        self.device = device
        self.count = 0

    def forward(self, batch):

        seq, lengths = batch
        device = "cuda:" + str(seq.get_device())

        for layer_name, entry in zip(self.layer_names, self.mods):

            if layer_name == "RnnBlock":
                seq, x = entry(seq, lengths, device=device)

            elif layer_name in ["ResBlock", "Linear"]:
                x = entry(x)

            elif layer_name == "ResBlockSeq":
                seq = entry(seq)

            else:
                raise ValueError(
                    "An unknown Module (%s) could not be processed." % (layer_name)
                )

        return x

    # def init_hidden(self, batch_size, layer, device):
    #     hidden_size = int(layer.weight_ih_l0.shape[0] / 4)
    #     if layer.bidirectional:
    #         num_dir = 2
    #     else:
    #         num_dir = 1

    #     return (
    #         torch.zeros(num_dir, batch_size, hidden_size, device=device),
    #         torch.zeros(num_dir, batch_size, hidden_size, device=device),
    #     )


class ResBlock(nn.Module):
    """A Residual block as proposed in 'Identity Mappings in Deep Residual Networks'
    """

    def __init__(self, arch_dict, layer_dict, n_in, n_out, norm=False):
        super(ResBlock, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        if n_in != n_out:
            self.linear0 = nn.Linear(in_features=n_in, out_features=n_out)

        if norm:
            self.norm1 = add_norm(arch_dict, layer_dict, n_out)
        self.non_lin1 = add_non_lin(arch_dict, arch_dict["nonlin"])
        self.linear1 = nn.Linear(in_features=n_out, out_features=n_out)
        init_weights(arch_dict, arch_dict["nonlin"], self.linear1)
        if norm:
            self.norm2 = add_norm(arch_dict, layer_dict, n_out)
        self.non_lin2 = add_non_lin(arch_dict, arch_dict["nonlin"])
        self.linear2 = nn.Linear(in_features=n_out, out_features=n_out)
        init_weights(arch_dict, arch_dict["nonlin"], self.linear2)

    def forward(self, seq, device=None):

        if self.n_in != self.n_out:
            seq = self.linear0(seq)

        res = self.linear1(self.non_lin1(self.norm1(seq)))
        res = self.linear2(self.non_lin2(self.norm2(res)))

        return seq + res


class SoftPlusSigma(nn.Module):
    def __init__(self, min_sigma=1e-3):
        super(SoftPlusSigma, self).__init__()
        self._min_sigma = min_sigma
        self._softplus = torch.nn.Softplus()

    def forward(self, x, device=None):

        # if not device:
        #     raise ValueError('A device must be supplied!')

        n_features = x.shape[-1] // 2
        mean = x[:, :n_features] + 0.0
        sigma = self._min_sigma + self._softplus(x[:, n_features:])
        out = torch.cat((mean, sigma), dim=-1)
        return out


def add_ResBlock(arch_dict, layer_dict):
    n_ins = layer_dict["input_sizes"][:-1]
    n_outs = layer_dict["input_sizes"][1:]

    layers = []
    for n_in, n_out in zip(n_ins, n_outs):
        layers.append(
            ResBlock(arch_dict, layer_dict, n_in, n_out, layer_dict.get("norm", False))
        )

    return nn.Sequential(*layers)


def add_linear_layers(arch_dict, layer_dict):
    n_layers = len(layer_dict["input_sizes"]) - 1

    # Add n_layers linear layers with non-linearity and normalization
    layers = []
    for i_layer in range(n_layers):
        isize = layer_dict["input_sizes"][i_layer]
        hsize = layer_dict["input_sizes"][i_layer + 1]

        # Add layer and initialize its weights
        layers.append(nn.Linear(in_features=isize, out_features=hsize))
        init_weights(arch_dict, arch_dict["nonlin"], layers[-1])

        # If last layer, do not add non-linearities or normalization
        if i_layer + 1 == n_layers:
            continue

        # If not, add non-linearities and normalization in required order
        else:
            if layer_dict["norm_before_nonlin"]:

                # Only add normalization layer if wanted!
                if arch_dict["norm"]["norm"] != None:
                    layers.append(add_norm(arch_dict, arch_dict["norm"], hsize))
                layers.append(add_non_lin(arch_dict, arch_dict["nonlin"]))

            else:
                layers.append(add_non_lin(arch_dict, arch_dict["nonlin"]))
                if arch_dict["norm"]["norm"] != None:
                    layers.append(add_norm(arch_dict, arch_dict["norm"], hsize))

    return nn.Sequential(*layers)


def add_non_lin(arch_dict, layer_dict):
    if arch_dict["nonlin"]["func"] == "ReLU":
        return nn.ReLU()

    elif arch_dict["nonlin"]["func"] == "LeakyReLU":
        negslope = arch_dict["nonlin"].get("negslope", 0.01)
        return nn.LeakyReLU(negative_slope=negslope)

    elif arch_dict["nonlin"]["func"] == "Mish":
        return Mish()

    else:
        raise ValueError(
            "An unknown nonlinearity could not be added in model generation."
        )


def add_norm(arch_dict, layer_dict, n_features):

    if layer_dict["norm"] == "BatchNorm1D":

        if "momentum" in layer_dict:
            mom = layer_dict["momentum"]
        else:
            mom = 0.1

        if "eps" in layer_dict:
            eps = layer_dict["eps"]
        else:
            eps = 1e-05

        return nn.BatchNorm1d(n_features, eps=eps, momentum=mom)

    elif layer_dict["norm"] == "LayerNorm":
        return nn.LayerNorm(n_features)

    else:
        raise ValueError(
            "An unknown normalization could not be added in model generation."
        )


def init_weights(arch_dict, layer_dict, layer):

    if type(layer) == torch.nn.modules.linear.Linear:
        if layer_dict["func"] == "ReLU":

            nn.init.kaiming_normal_(
                layer.weight, a=0, mode="fan_in", nonlinearity="relu"
            )

        elif layer_dict["func"] == "LeakyReLU" or layer_dict["func"] == "Mish":

            if "negative_slope" in layer_dict:
                negslope = layer_dict["negative_slope"]
            else:
                negslope = 0.01

            nn.init.kaiming_normal_(
                layer.weight, a=negslope, mode="fan_in", nonlinearity="leaky_relu"
            )

        else:
            raise ValueError("An unknown initialization was encountered.")
    else:
        raise ValueError("An unknown initialization was encountered.")


def make_model_architecture(arch_dict):

    modules = nn.ModuleList()
    for layer in arch_dict["layers"]:
        for key, layer_dict in layer.items():

            if key == "ResBlock":
                modules.append(add_ResBlock(arch_dict, layer_dict))
            elif key == "Linear":
                modules.append(add_linear_layers(arch_dict, layer_dict))
            elif key == "RnnBlock":
                modules.append(RnnBlock(**layer_dict))
            elif key == "SoftPlusSigma":
                modules.append(SoftPlusSigma())
            else:
                raise ValueError(
                    "An unknown module (%s) could not be added in model generation."
                    % (key)
                )

    return modules


def get_layer_names(arch_dict):
    """Extracts layer names from an arch_dict
    """
    layer_names = []
    for layer in arch_dict["layers"]:
        for layer_name, dicts in layer.items():

            if layer_name == "ResBlock":
                if dicts["type"] == "seq":
                    layer_names.append("ResBlockSeq")
                elif dicts["type"] == "x":
                    layer_names.append("ResBlock")
                else:
                    raise KeyError('ResBlock: "type" MUST be supplied!')
            else:
                layer_names.append(layer_name)

    return layer_names
