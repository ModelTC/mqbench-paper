import operator
from typing import Dict, List, Tuple

import torch
from torch.fx import GraphModule, Node
from torch.quantization.fx.pattern_utils import is_match, MatchAllNode
from torch.quantization.fake_quantize import FakeQuantizeBase
from torch.nn.intrinsic.qat.modules.conv_fused import ConvBnReLU2d, ConvBn2d, ConvReLU2d

from prototype.quantization.bn_fold import ConvBNTorchFold, ConvBNReLUTorchFold
from prototype.quantization.qconfig import get_activation_fake_quantize


_TENSORRT_SKIP_PATTERNS = [
                           (operator.add, (FakeQuantizeBase, ConvReLU2d), MatchAllNode),
                           (operator.add, (FakeQuantizeBase, ConvBn2d), MatchAllNode),
                           (operator.add, (FakeQuantizeBase, ConvBnReLU2d), MatchAllNode),

                           (operator.add, MatchAllNode, (FakeQuantizeBase, ConvReLU2d)),
                           (operator.add, MatchAllNode, (FakeQuantizeBase, ConvBn2d)),
                           (operator.add, MatchAllNode, (FakeQuantizeBase, ConvBnReLU2d)),

                           (torch.add, (FakeQuantizeBase, ConvReLU2d), MatchAllNode),
                           (torch.add, (FakeQuantizeBase, ConvBn2d), MatchAllNode),
                           (torch.add, (FakeQuantizeBase, ConvBnReLU2d), MatchAllNode),

                           (torch.add, MatchAllNode, (FakeQuantizeBase, ConvReLU2d)),
                           (torch.add, MatchAllNode, (FakeQuantizeBase, ConvBn2d)),
                           (torch.add, MatchAllNode, (FakeQuantizeBase, ConvBnReLU2d)),
                           ]


def _find_matches_tensorrt(root: GraphModule, patterns):
    modules = dict(root.named_modules())
    excluded = set()
    matches = []

    for node in reversed(root.graph.nodes):
        for pat_id, pattern in enumerate(patterns):
            if node not in excluded:
                if is_match(modules, node, pattern):
                    matches.append((node, pat_id))
                    excluded.update({node})
            else:
                break
    return list(reversed(matches))


def _remove_which_branch(gm: GraphModule,
                         matches: List[Tuple],
                         skip_pattern=_TENSORRT_SKIP_PATTERNS, # noqa
                         quantize_node=FakeQuantizeBase):

    modules = dict(gm.named_modules())

    for match_nodes in matches:
        node, pat_id = match_nodes
        matched_pattern = skip_pattern[pat_id]

        # TODO: need a more general method to find the branch where the fake quantization node will be removed.
        # remove_id = matched_pattern.index(MatchAllNode) - 1
        remove_id = 0  # Do not include add operator itself.

        input_args = node.args
        removed_fake_node: Node = input_args[remove_id]

        if isinstance(modules[removed_fake_node.target], quantize_node):
            removed_fake_node.replace_all_uses_with(removed_fake_node.all_input_nodes[0])
            gm.graph.erase_node(removed_fake_node)

    gm.recompile()
    gm.graph.lint()


def remove_redundant_fake_node_tensorrt(net: GraphModule,
                                        skip_pattern=_TENSORRT_SKIP_PATTERNS, # noqa
                                        quantize_node=FakeQuantizeBase): # noqa
    """
    This function removes redundant fake-quantize node between add and conv operations.

    It first matches the pattern in a given graph module, then gets the operand cascaded after
    a fake-quantize node (lhs in this function for example). Afterward, we remove the fake quantize
    node between add and conv, and recompile.

    Args:
        net: Network that needs to remove redundant fake quantize node.
        skip_pattern: print net info or not.
    """

    matches = _find_matches_tensorrt(net, skip_pattern)
    _remove_which_branch(net, matches, skip_pattern=skip_pattern, quantize_node=quantize_node)
    net.recompile()


def pattern_find_replace(input_tuple: Tuple, replace_pattern: Dict) -> List:
    """
    This function unpacks a pattern and locates elements given in replace_pattern,
    then replaces the original element with the updated element.

    Example:
        input_tuple: (Add, (ConvBNFused, ReLU), Conv2d)
        replace_pattern: {ConvBNFused: CustomConvBNFused}

        output: (Add, (CustomConvBNFused, ReLU), Conv2d)

    Args:
        input_tuple: the original pattern.
        replace_pattern: the replace mapping dict.
    """
    orig_list = list(input_tuple)
    replace_tuple = tuple(replace_pattern.keys())

    def match_id(_ele, pattern_replace_tuple):
        for _id, item in enumerate(pattern_replace_tuple):
            if _ele == item:
                return _id
        return -1

    for idx, ele in enumerate(orig_list):
        if not isinstance(ele, (tuple, list)):
            matched_idx = match_id(ele, replace_tuple)
            if matched_idx != -1:
                orig_list[idx] = replace_pattern[replace_tuple[matched_idx]]
        else:
            orig_list[idx] = pattern_find_replace(ele, replace_pattern)
    return orig_list


def reform_pattern(pattern_list: List) -> Tuple:
    """
    The updated pattern is a nested list which can not be used in pattern match. This function recursively
    converts a nested list to a nested tuple.

    Example:
        input pattern_list: [add, [conv, relu], conv]
        output: (add, (conv, relu), conv)

    Args:
        pattern_list: the pattern list requiring conversion.
    """
    if len(pattern_list) == 1:
        return tuple(pattern_list)
    else:
        for item in pattern_list:
            if isinstance(item, list):
                idx = pattern_list.index(item)
                pattern_list[idx] = reform_pattern(item)
            else:
                continue
        return tuple(pattern_list)


def quant_qat_pattern_update(orig_patterns: List, additional_conf: Dict):
    replaced_patterns, final_patterns = [], []
    for pattern in orig_patterns:
        replaced_patterns.append(pattern_find_replace(pattern, additional_conf))
    for pattern in replaced_patterns:
        final_patterns.append(reform_pattern(pattern))
    return final_patterns


def tensorrt_refactor(net: GraphModule, a_method, additional_mapping: Dict):
    """
    This funtion refactors the given net using tensorrt quantization pattern. Meanwhile, it also
    supports custom conv-bn fold methods.

    Args:
        net: the graph module that has been processed by prepare_qat_fx.
        qconfig: QConfig containing weight and activation fake quantization methods.
        additional_mapping: custom fold bn methods.
    Return:
        None
    """
    replace_quantize_node = get_activation_fake_quantize(a_method)
    mapping = {torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d: ConvBNTorchFold,
               torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d: ConvBNReLUTorchFold,
               torch.quantization.FakeQuantizeBase: replace_quantize_node}

    if additional_mapping is not None:
        assert isinstance(additional_mapping, dict)
        for k in additional_mapping.keys():
            if k == torch.nn.intrinsic.ConvBn2d:
                mapping.update({torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d: additional_mapping[k]})
            elif k == torch.nn.intrinsic.ConvBnReLU2d:
                mapping.update({torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d: additional_mapping[k]})
            else:
                raise RuntimeError("Unsupported additional fold bn mapping.")
    print(mapping)

    quant_pattern_tensorrt = quant_qat_pattern_update(_TENSORRT_SKIP_PATTERNS, mapping)
    remove_redundant_fake_node_tensorrt(net, quant_pattern_tensorrt,
                                        quantize_node=replace_quantize_node)