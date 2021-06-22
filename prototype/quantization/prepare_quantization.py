import copy
import time
from typing import Any, List, Dict

import torch
import torch.nn as nn
from torch.nn.qat.modules import Conv2d, Linear
from torch.fx import symbolic_trace, GraphModule, Node
from torch.fx.graph import _format_target  # noqa

from torch.quantization import propagate_qconfig_
from torch.quantization.quantize import _convert  # noqa
from torch.quantization.fx.qconfig_utils import get_flattened_qconfig_dict
from prototype.quantization.qconfig import get_qconfig
from prototype.quantization.base_quantizer import toggle_fake_quant, enable_param_learning, QuantizeBase, \
    enable_static_estimate, enable_static_observation

import torch.quantization.quantize_fx as quantize_fx
from prototype.quantization.bn_fold import ConvBNNaiveFold, ConvBNReLUNaiveFold, ConvBNLookAheadFold, \
    ConvBNReLULookAheadFold, ConvBNTorchFold, ConvBNReLUTorchFold, ConvBNBase, ConvBNReLUBase, search_fold_bn,\
    ConvBNWPFold, ConvBNReLUWPFold, ConvBNFreeze, ConvBNReLUFreeze, ConvBNMerge, ConvBNReLUMerge
from torch.quantization.fx.quantization_patterns import ConvRelu


# _LSQ_INPUT_OUTPUT_QCONFIG = get_lsq_qconfig(bit=8, signed=True)
_MAPPING = {nn.Conv2d: Conv2d, nn.Linear: Linear}
_QUANTIZABLE_LAYER_TYPE = (nn.Conv2d, nn.Linear)


def prepare_quant_academic(model: nn.Module, **qconfig_dict):
    graph_module = symbolic_trace(model)

    qconfig_normal = get_qconfig(**qconfig_dict)
    qconfig_dict['bit'] = 8
    qconfig_8bit = get_qconfig(**qconfig_dict)

    flattened_qconfig_dict = get_flattened_qconfig_dict({"": qconfig_normal})
    # put quantization config on the weight for all layers
    propagate_qconfig_(graph_module, flattened_qconfig_dict)
    # replace head & stem layer with 8-bit weight qconfig
    skip_input_output(graph_module, qconfig_8bit)

    # TODO: need a more general implementation, instead qconfig_dict[""]
    # insert activation node
    insert_pre_process(graph_module, qconfig_normal.activation, qconfig_8bit.activation)
    # convert weight qconfig to quantization weight
    graph_module = _convert(graph_module, _MAPPING, inplace=False, convert_custom_config_dict=None)
    return graph_module


def skip_input_output(gm: GraphModule, qconfig_8bit):
    """
    Find input and output layer, and replace original qconfig with 8-bit qconfig.
    """
    for node in gm.graph.nodes:
        if node.op == "placeholder" or node.op == "output":
            to_process = (list(node.users.keys()), node.all_input_nodes)[node.op != "placeholder"]
            for process_node in to_process:
                if process_node.op == "call_module":
                    orig_mod = eval(_format_target("gm", process_node.target))
                    if isinstance(orig_mod, _QUANTIZABLE_LAYER_TYPE):
                        orig_mod.qconfig = qconfig_8bit


def insert_pre_process(gm: GraphModule, q_mod, qconfig_8bit_act):
    """
        Insert fake-quantize nodes before layers in _LSQ_QUANTIZABLE_LAYER_TYPE. To keep consistent with
        LSQ implementation, the input layer and output layer are kept in 8-bit.

        Args:
            gm: a graph module that needs to be quantized.
            q_mod: a LearnableFakeQuantize module, it will be instantiated and insterted to the gm.
        Returns:
            None
    """
    node_list = list(gm.graph.nodes)

    ind = 0
    for node in node_list:
        # TODO: need to deal with call_function
        if node.op == "call_module":
            # TODO: maybe other way to replace eval()?
            orig_mod = eval(_format_target("gm", node.target))
            if isinstance(orig_mod, _QUANTIZABLE_LAYER_TYPE):
                insert_name = "conv_pre_process_{}".format(ind)

                # If this layer is the input or output layer,
                # we keep its pre-activation in 8-bit.
                # TODO: need to refactor: use pattern match to dispatch fake-quantize node.
                if _is_input_layer(node):
                    inserted_mod = qconfig_8bit_act
                elif _is_output_layer(node):
                    inserted_mod = qconfig_8bit_act
                else:
                    inserted_mod = q_mod
                setattr(gm, insert_name, inserted_mod())

                inp = node.all_input_nodes
                with gm.graph.inserting_before(node):
                    inserted_node = gm.graph.create_node("call_module", insert_name, (inp[0], ), {})
                node.args = (inserted_node, )
                ind += 1

    gm.recompile()
    gm.graph.lint()


def _is_output_layer(node: Node):
    for n in list(node.users.keys()):
        if n.op == "output":
            return True
    return False


def _is_input_layer(node: Node):
    for n in node.all_input_nodes:
        if n.op == "placeholder":
            return True
    return False


def insert_all():
    import warnings
    import torchvision
    entry_points = torch.hub.list('pytorch/vision', force_reload=False)
    exclude_list = [
        'googlenet',  # existing error
        'mnasnet0_75', 'mnasnet1_3'  # no checkpint in torch hub
    ]

    for entrypoint in entry_points:  # noqa
        print(f'testing {entrypoint}')
        if entrypoint in exclude_list:
            continue
        try:
            model = getattr(torchvision.models, entrypoint)(pretrained=False)
        except AttributeError:
            warnings.warn("Can load model {}".format(entrypoint), RuntimeWarning)
            continue

        try:
            prepare_lsq(model, {"": get_lsq_qconfig(bit=4)})  # noqa
            print("Model prepare success for {}.".format(entrypoint))
        except RuntimeError as e:
            warnings.warn("Prepare fault {}: {}.".format(entrypoint, e))
            continue


def backtrack_find_quant(node: Node, modules: Dict):
    """
    This function backtracks the graph from a given node to find the first quantization node.

    node: the node we will begin our backtrack at.
    modules: the module dict created by dict(model.named_modules()).
    """
    if node.op == "call_module":
        if isinstance(modules[node.target], QuantizeBase):
            return node
    # TODO: are these conditions sufficient ?
    if ((node.op == "call_module" or node.op == "call_function") and len(node.all_input_nodes) == 1) or \
            node.op == "call_method":
        return backtrack_find_quant(node.all_input_nodes[0], modules)
    else:
        return None


def bitwidth_refactor(model: GraphModule, input_quant=None):
    """
    Bitwidth refactor is a helper function to deal with input and output quantization configurations.
    As most academic and some industrial settings allow input and output layers are represented in 8-bit,
    it is necessary to do a refactor operation to ensure that these layers are not quantized to low bit-width.

    The current version does the following steps to achieve bit-width modification:
        1) visit all nodes in the graph and find input and output nodes.
        2) if the input node is a quantize node, simply change its bit-width config.
            if the input node is a param layer, try to find the weight_fake_quant property and change its config.
        3) similar processing is done with the output node, except that it needs to backtrack the whole
            graph to find the input activation to the final output node, then change the input activation to 8-bit.

    Considering that there are two types defined in FX graph node, i.e. call_module and call_function, this
    function addresses these cases separately, which may be somewhat confusing and error-prone.
    """
    import warnings

    def set_layer_to_8bit(_module):
        _module.quant_max = 127
        _module.quant_min = -128
        _module.bitwidth = 8
        _module.activation_post_process.quant_max = 127
        _module.activation_post_process.quant_min = -128

    modules = dict(model.named_modules())
    for node in model.graph.nodes:
        if node.op == "call_module":
            module = modules[node.target]
            if _is_input_layer(node):
                if isinstance(module, nn.Module):
                    if hasattr(module, "weight_fake_quant"):
                        set_layer_to_8bit(module.weight_fake_quant)
                        inp: Node = node.all_input_nodes[0]
                        with model.graph.inserting_after(inp):
                            insert_name = "input_activation_quant"
                            setattr(model, insert_name, input_quant())
                            set_layer_to_8bit(model.input_activation_quant)
                            inserted_node = model.graph.create_node("call_module", insert_name, (inp, ), {})
                        inp.replace_all_uses_with(inserted_node)
                        inserted_node.args = (inp, )
                    # If input is quantized, we try to find the first param layer and change its qconfig.
                    elif isinstance(module, torch.quantization.FakeQuantizeBase):
                        set_layer_to_8bit(module)
                        users: List[Node] = list(node.users.keys())
                        for user in users:
                            if user.op == "call_module":
                                param_layer = modules[user.target]
                                if hasattr(param_layer, "weight_fake_quant"):
                                    set_layer_to_8bit(param_layer.weight_fake_quant)
                                else:
                                    warnings.warn("Fail to find the first quantizable layer after layer {},"
                                                  "is this normal ?".format(node.target),
                                                  RuntimeWarning)
                    else:
                        warnings.warn("Input layer: {} is not a quantizable layer ?".format(node.target),
                                      RuntimeWarning)
            if _is_output_layer(node):
                if isinstance(module, QuantizeBase):
                    # Skip output layer.
                    delattr(model, node.target)
                    setattr(model, node.target, nn.Identity())
                else:
                    warnings.warn("Output layer: {} is not quantized ?"
                                  "The output quantization node is not found.".format(node.target),
                                  RuntimeWarning)
                    continue
                inp = node.all_input_nodes
                if len(inp) == 1:
                    if inp[0].op == "call_module":
                        input_module = modules[inp[0].target]
                        set_layer_to_8bit(input_module.weight_fake_quant)
                        inp_quant_node = backtrack_find_quant(inp[0], modules)
                        if inp_quant_node is not None:
                            set_layer_to_8bit(modules[inp_quant_node.target])
                        else:
                            warnings.warn("Output layer: {} has no input quantization ?".format(node.target),
                                          RuntimeWarning)
                    # TODO: need to refactor
                    elif inp[0].op == "call_function":
                        if inp[0].target == torch.nn.functional.linear or inp[0].target == torch.nn.functional.conv2d:
                            activation = inp[0].all_input_nodes[0]
                            weight = inp[0].all_input_nodes[1]
                            if activation.op == "call_module":
                                if isinstance(modules[activation.target], QuantizeBase):
                                    set_layer_to_8bit(modules[activation.target])
                            if weight.op == "call_module":
                                if isinstance(modules[weight.target], QuantizeBase):
                                    set_layer_to_8bit(modules[weight.target])
                    else:
                        warnings.warn("{} is unknown type, check the graph module code ?".format(inp[0].target),
                                      RuntimeWarning)
                else:
                    warnings.warn("Output layer: {} has multiple branches ?".format(node.target),
                                  RuntimeWarning)
    model.recompile()


def get_foldbn_config(strategy=3):
    assert -1 <= strategy <= 4, 'Wrong BN folding strategy!'
    additional_config = {}
    if strategy == -1:
        # no fold bn, normal conv-and-bn
        additional_config['additional_qat_module_mapping'] = {torch.nn.intrinsic.ConvBn2d: ConvBNBase,
                                                              torch.nn.intrinsic.ConvBnReLU2d: ConvBNReLUBase}
        additional_config['additional_quant_pattern'] = {ConvBNBase: ConvRelu}
    elif strategy == 0:
        # freeze bn
        additional_config['additional_qat_module_mapping'] = {torch.nn.intrinsic.ConvBn2d: ConvBNMerge,
                                                              torch.nn.intrinsic.ConvBnReLU2d: ConvBNReLUMerge}
        additional_config['additional_quant_pattern'] = {ConvBNMerge: ConvRelu}
    elif strategy == 1:
        # freeze bn
        additional_config['additional_qat_module_mapping'] = {torch.nn.intrinsic.ConvBn2d: ConvBNFreeze,
                                                              torch.nn.intrinsic.ConvBnReLU2d: ConvBNReLUFreeze}
        additional_config['additional_quant_pattern'] = {ConvBNFreeze: ConvRelu}
    elif strategy == 2:
        # naive fold bn: proposed in integer-only quantization
        additional_config['additional_qat_module_mapping'] = {torch.nn.intrinsic.ConvBn2d: ConvBNNaiveFold,
                                                         torch.nn.intrinsic.ConvBnReLU2d: ConvBNReLUNaiveFold}
        additional_config['additional_quant_pattern'] = {ConvBNNaiveFold: ConvRelu}
    elif strategy == 3:
        # white paper fold bn
        additional_config['additional_qat_module_mapping'] = {torch.nn.intrinsic.ConvBn2d: ConvBNWPFold,
                                                             torch.nn.intrinsic.ConvBnReLU2d: ConvBNReLUWPFold}
        additional_config['additional_quant_pattern'] = {ConvBNWPFold: ConvRelu}
    else:
        # torch fold bn
        additional_config['additional_qat_module_mapping'] = {torch.nn.intrinsic.ConvBn2d: ConvBNTorchFold,
                                                              torch.nn.intrinsic.ConvBnReLU2d: ConvBNReLUTorchFold}
        additional_config['additional_quant_pattern'] = {ConvBNTorchFold: ConvRelu}
    return additional_config


if __name__ == "__main__":
    from prototype.model.mobilenet_v2 import mobilenetv2
    from prototype.model.resnet import resnet50, resnet18

    net = mobilenetv2()
    qconfig_params = dict(
        w_method='lsq', a_method='lsq', bit=4, ada_sign=False, symmetry=True, per_channel=True, pot_scale=False,
    )
    # net = prepare_quant_academic(net, **qconfig_params)

    net = quantize_fx.prepare_qat_fx(net, {"": get_qconfig(**qconfig_params)}, get_foldbn_config(strategy=4))
    # search_fold_bn(net, strategy=4)
    bitwidth_refactor(net, get_qconfig(**qconfig_params).activation)

    # from prototype.quantization.prepare_tensorrt import tensorrt_refactor
    # # print(get_foldbn_config(strategy=4)['additional_qat_module_mapping'])
    # tensorrt_refactor(net, "dsq", get_foldbn_config(strategy=0)['additional_qat_module_mapping'])

    dummy_input = torch.randn((1, 3, 224, 224))
    net(dummy_input)
    enable_param_learning(net)

    fwd_time = []
    bwd_time = []

    for i in range(5):
        net.zero_grad()
        dummy_input = torch.randn((1, 3, 224, 224))
        time1 = time.time()
        out = net(dummy_input)
        fwd_time += [time.time() - time1]
        time2 = time.time()
        out.mean().backward()
        bwd_time += [time.time() - time2]
    # for name, p in net.named_parameters():
    #     if 'zero_point' in name:
    #         print(name, p)

    print(fwd_time)
    print(bwd_time)


