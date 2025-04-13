import argparse
from pprint import pprint

import yaml


def modify_yaml_fields(yaml_path, field_updates, save_path):
    """
    修改 YAML 文件中的多个字段。

    :param yaml_path: YAML 文件路径
    :param field_updates: 字典，包含字段名和对应的新值
    :param save_path: 保存修改后的 YAML 文件路径
    """
    # 读取 YAML 文件
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)  # 加载 YAML 内容
    print("Original YAML content:")
    pprint(data)

    # 修改多个字段
    for field_name, new_value in field_updates.items():
        if field_name in data:  # 检查字段是否存在
            data[field_name] = new_value
        else:
            print(f"Warning: Field '{field_name}' not found in the YAML file. Adding it.")
            data[field_name] = new_value  # 如果字段不存在，则添加新字段

    print("Modified YAML content:")
    pprint(data)
    # 将修改后的内容保存到指定路径
    with open(save_path, 'w') as file:
        yaml.safe_dump(data, file, default_flow_style=False)  # 保存 YAML 内容

    print(f"YAML file has been modified and saved to {save_path}")

if __name__ == "__main__":
    # 设置命令行参数
    parser = argparse.ArgumentParser(description="Modify multiple fields in a YAML file.")
    parser.add_argument("--yaml_path", type=str, help="Path to the input YAML file")
    parser.add_argument("--fields", nargs='+', help="Field names and new values in the format 'field_name=new_value'")
    parser.add_argument("--save_path", type=str, help="Path to save the modified YAML file")

    # 解析命令行参数
    args = parser.parse_args()

    # 解析字段更新
    field_updates = {}
    for field in args.fields:
        field_name, new_value = field.split('=', 1)  # 按等号分割字段名和新值
        try:
            new_value = eval(new_value)  # 尝试将新值转换为 Python 对象
        except (NameError, SyntaxError):
            if new_value.lower() == 'true':
                new_value = True
            elif new_value.lower() == 'false':
                new_value = False
            elif new_value.lower() == 'none':
                new_value = None
        except Exception as e:
            pass
        field_updates[field_name] = new_value

    # 调用函数修改 YAML 文件
    modify_yaml_fields(args.yaml_path, field_updates, args.save_path)