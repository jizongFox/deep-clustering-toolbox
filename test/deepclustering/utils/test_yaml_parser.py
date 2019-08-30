from unittest import TestCase

from deepclustering.utils.yaml_parser import YAMLArgParser, yaml_load

CONFIG_PATH = "test/deepclustering/utils/config.yaml"


# class TestYamlParser(TestCase):
#     def test_yaml_loader(self):
#         yaml_load(CONFIG_PATH, verbose=True)
#         yaml_load(CONFIG_PATH, verbose=False)
#
#     def test_yaml_argparser(self):
#         YAMLArgParser()
