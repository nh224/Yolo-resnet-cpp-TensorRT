#!/usr/bin/env python3
"""
ResNet ONNX 导出脚本
支持从 PyTorch .pth/.pt 权重导出 ONNX 模型
"""

import torch
import torch.nn as nn
import torchvision.models as models
import argparse
import os


class ResNetExporter:
    """ResNet ONNX 导出器"""

    # ResNet 架构映射
    ARCHITECTURES = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
    }

    def __init__(self, arch: str, num_classes: int = 1000, weights_path: str = None):
        """
        初始化导出器

        Args:
            arch: ResNet 架构名称 (resnet18, resnet50, etc.)
            num_classes: 分类数量，默认 1000 (ImageNet)
            weights_path: 预训练权重路径，None 则使用官方权重
        """
        if arch not in self.ARCHITECTURES:
            raise ValueError(f"不支持的架构: {arch}，支持: {list(self.ARCHITECTURES.keys())}")

        self.arch = arch
        self.num_classes = num_classes
        self.weights_path = weights_path

        self.model = self._load_model()

    def _load_model(self) -> nn.Module:
        """加载模型"""
        # 创建模型
        model_fn = self.ARCHITECTURES[self.arch]

        if self.num_classes == 1000 and self.weights_path is None:
            # 使用官方预训练权重
            model = model_fn(weights=models.ResNet18_Weights.IMAGENET1K_V1
                           if self.arch == 'resnet18' else
                           models.ResNet50_Weights.IMAGENET1K_V1)
            print(f"✓ 加载官方 {self.arch} ImageNet 预训练权重")
        else:
            # 自定义类别数或自定义权重
            model = model_fn(num_classes=self.num_classes)
            if self.weights_path and os.path.exists(self.weights_path):
                state_dict = torch.load(self.weights_path, map_location='cpu')
                model.load_state_dict(state_dict)
                print(f"✓ 从 {self.weights_path} 加载权重")
            else:
                print(f"⚠ 使用随机初始化权重")

        model.eval()
        return model

    def export_onnx(self,
                    output_path: str,
                    input_size: int = 224,
                    batch_size: int = 1,
                    opset_version: int = 17,
                    dynamic: bool = False) -> None:
        """
        导出 ONNX 模型

        Args:
            output_path: 输出 ONNX 文件路径
            input_size: 输入图像尺寸 (默认 224)
            batch_size: batch size (默认 1)
            opset_version: ONNX opset 版本 (默认 17)
            dynamic: 是否使用动态 batch size
        """
        dummy_input = torch.randn(batch_size, 3, input_size, input_size)

        # 导出配置
        export_kwargs = {
            'f': output_path,
            'opset_version': opset_version,
            'input_names': ['images'],
            'output_names': ['output'],
            'dynamic_axes': None,
        }

        if dynamic:
            export_kwargs['dynamic_axes'] = {
                'images': {0: 'batch'},
                'output': {0: 'batch'},
            }

        print(f"\n{'='*50}")
        print(f"导出 {self.arch} 到 ONNX")
        print(f"{'='*50}")
        print(f"输入尺寸: {batch_size} x 3 x {input_size} x {input_size}")
        print(f"类别数: {self.num_classes}")
        print(f"Opset 版本: {opset_version}")
        print(f"动态 Batch: {dynamic}")
        print(f"输出路径: {output_path}")

        # 执行导出
        torch.onnx.export(
            self.model,
            dummy_input,
            **export_kwargs,
            export_params=True,
            do_constant_folding=True,
        )

        print(f"✓ ONNX 导出成功!")

        # 验证 ONNX
        self._verify_onnx(output_path, dummy_input)

    def _verify_onnx(self, onnx_path: str, dummy_input: torch.Tensor) -> None:
        """验证导出的 ONNX 模型"""
        import onnx
        import onnxruntime as ort

        print(f"\n{'='*50}")
        print("验证 ONNX 模型")
        print(f"{'='*50}")

        # 加载并检查模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX 模型检查通过")

        # 使用 ONNXRuntime 推理验证
        ort_session = ort.InferenceSession(onnx_path)
        outputs = ort_session.run(None, {'images': dummy_input.numpy()})

        # 与 PyTorch 输出对比
        with torch.no_grad():
            torch_output = self.model(dummy_input)

        diff = abs(torch_output.numpy() - outputs[0]).max()
        print(f"✓ ONNXRuntime 推理成功")
        print(f"✓ PyTorch 与 ONNX 输出差异: {diff:.6f}")

        if diff < 1e-5:
            print("✓ 输出一致，导出正确!")
        else:
            print("⚠ 输出差异较大，请检查")


def main():
    parser = argparse.ArgumentParser(description='ResNet ONNX 导出工具')
    parser.add_argument('--arch', type=str, default='resnet18',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'],
                       help='ResNet 架构')
    parser.add_argument('--weights', type=str, default=None,
                       help='预训练权重路径 (.pth/.pt)，不指定则使用官方权重')
    parser.add_argument('--num-classes', type=int, default=1000,
                       help='分类数量 (默认: 1000)')
    parser.add_argument('--input-size', type=int, default=224,
                       help='输入图像尺寸 (默认: 224)')
    parser.add_argument('--output', type=str, default='resnet18.onnx',
                       help='输出 ONNX 文件路径')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size (默认: 1)')
    parser.add_argument('--opset', type=int, default=17,
                       help='ONNX opset 版本 (默认: 17)')
    parser.add_argument('--dynamic', action='store_true',
                       help='启用动态 batch size')

    args = parser.parse_args()

    # 自动命名输出文件
    if args.output == 'resnet18.onnx' and args.arch != 'resnet18':
        args.output = f'{args.arch}.onnx'

    # 创建导出器并导出
    exporter = ResNetExporter(
        arch=args.arch,
        num_classes=args.num_classes,
        weights_path=args.weights,
    )

    exporter.export_onnx(
        output_path=args.output,
        input_size=args.input_size,
        batch_size=args.batch_size,
        opset_version=args.opset,
        dynamic=args.dynamic,
    )


if __name__ == '__main__':
    main()
