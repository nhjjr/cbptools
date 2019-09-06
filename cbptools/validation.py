from cbptools.utils import pyyaml_ordereddict
from collections import OrderedDict
from functools import partial as fp
import logging
import fnmatch
import yaml


class DocumentError(Exception):
    """ Raised when the target document is missing or has the wrong format """
    pass


class ModalityError(Exception):
    """ Raised when the target document is missing or has the wrong format """
    pass


class RuleError(Exception):
    """ Raised when a rule is violated """
    pass


class SetDefault(Exception):
    """ Raised when an input value is not required and not given """
    pass


class Validator(object):
    priority = ('required', 'type', 'contains', 'allowed', 'min',
                'max', 'minlength', 'maxlength')

    def __init__(self, schema: str):
        yaml.add_representer(OrderedDict, pyyaml_ordereddict)
        self.schema_path = schema
        self.schema = None
        self.document = None
        self.errors = 0

    @staticmethod
    def load(path) -> dict:
        with open(path, 'r') as stream:
            try:
                return OrderedDict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                raise DocumentError(exc)

    @staticmethod
    def construct_schema(modality: str, schema_template: dict) -> dict:
        def merge(a, b, path=None):
            "merges b into a"
            if path is None:
                path = []

            for k in b:
                if k in a:
                    if isinstance(a[k], dict) and isinstance(b[k], dict):
                        merge(a[k], b[k], path + [str(k)])
                    elif a[k] == b[k]:
                        pass  # same leaf value
                    else:
                        raise Exception(
                            'Conflict at %s' % '.'.join(path + [str(k)]))
                else:
                    a[k] = b[k]
            return a

        general_schema = schema_template.get('schema-general')
        specific_schema = 'schema-%s' % modality

        if specific_schema in schema_template.keys():
            modality = {'modality': schema_template.get('modality')}
            general_schema = merge(modality, general_schema)
            return merge(general_schema, schema_template.get(specific_schema))

        raise ModalityError('Modality %s not recognized' % modality)

    def _rule_parser(self, field: str, value: str, schema: dict):
        """Return a list containing all rule functions to be executed"""
        rulelst = []

        if 'required' not in schema.keys():
            schema['required'] = False

        order = list(schema.keys())
        priority = list(self.priority) + list(set(order) - set(self.priority))
        order.sort(key=lambda x: priority.index(x))

        for k in order:
            if hasattr(self, '_rule_%s' % k):
                rule = getattr(self, '_rule_%s' % k)
                rule = fp(rule, field, value, schema.get(k))
                rulelst.append(rule)

        if 'custom' in schema.keys():
            for k in schema['custom']:
                if hasattr(self, '_rule_custom_%s' % k):
                    rule = getattr(self, '_rule_custom_%s' % k)
                    rule = fp(rule, field, value)
                    rulelst.append(rule)

        return rulelst

    def depth(self, d, level=0):
        if not isinstance(d, dict) or not d:
            return level

        return max(self.depth(d[k], level + 1) for k in d.keys())

    def _validate_against(self, document, schema, path: list = []):
        for k, v in schema.items():
            path.append(k)

            if isinstance(v, dict) and self.depth(v) == 1 and not isinstance(
                    document, dict):
                self.errors += 1
                logging.error('%s must be %s, not %s'
                              % (type(document).__name__, '.'.join(path[:-1])))
                break

            elif isinstance(v, dict) and self.depth(v) == 1:
                value = document.get(k, None) if isinstance(document,
                                                            dict) else None
                rules = self._rule_parser('.'.join(path), value, v)

                for rule in rules:
                    try:
                        rule()
                    except RuleError as exc:
                        self.errors += 1
                        logging.error(exc)
                        break
                    except SetDefault as exc:
                        default = schema[k].get('default', None)
                        if value is None and default is not None:
                            logging.warning(
                                'using default value for %s' % '.'.join(path))
                            document[k] = default
                        break

            elif isinstance(v, dict) and self.depth(v) > 1:
                if not isinstance(document, dict) and document is None:
                    document = {}

                elif not isinstance(document, dict):
                    self.errors += 1
                    logging.error('%s must be dict, not %s'
                                  % ('.'.join(path[:-1]),
                                     type(document).__name__))
                    del path[-1]
                    break

                document[k] = self._validate_against(document.get(k, {}), v,
                                                     path)

            del path[-1]

        return document

    def _del_invalid(self, document: dict, schema: dict, path: list = []):
        for k, v in list(document.items()):
            path.append(k)
            if k not in schema.keys():
                logging.warning('Invalid key: %s' % ('.'.join(path)))
                del document[k]
                del path[-1]
                continue

            if isinstance(v, dict):
                document[k] = self._del_invalid(v, schema.get(k), path)

            del path[-1]

        return document

    def validate(self, document) -> bool:
        """Validate YAML and then schema"""
        document = OrderedDict(self.load(document))
        modality = document.get('modality', None)
        schema_template = self.load(self.schema_path)

        if modality not in schema_template['modality']['allowed']:
            self.errors += 1
            logging.error('modality must be %s, not %s'
                          % (', '.join(schema_template['modality']['allowed']),
                             modality))
            return False

        # Create the validation schema
        schema = self.construct_schema(modality, schema_template)

        # Validate and set defaults
        document = self._validate_against(document, schema)

        # Remove invalid keys
        document = self._del_invalid(document, schema)

        if self.errors == 0:
            self.document = document
            return True

        else:
            return False

    def _set_defaults(self, d):
        for k, v in d.items():
            if isinstance(v, dict) and self.depth(v) <= 1:
                default = v.get('default', None)
                d[k] = default

            elif isinstance(v, dict):
                d[k] = self._set_defaults(v)

        return d

    def example(self, modality: str, out: str = None):
        schema_template = self.load(self.schema_path)

        if modality not in schema_template['modality']['allowed']:
            return False

        schema = OrderedDict(self.construct_schema(modality, schema_template))
        self.document = self._set_defaults(schema)
        self.document['modality'] = modality

        if out:
            with open(out, 'w') as f:
                yaml.dump(self.document, f, default_flow_style=False)

    @staticmethod
    def _rule_required(field, value, *args) -> bool:
        if args[0] and value is None:
            raise RuleError('%s is a required field' % field)
        elif not args[0] and value is None:
            raise SetDefault()
        else:
            return True

    @staticmethod
    def _rule_type(field, value, *args) -> bool:
        types = {'integer': int, 'string': str, 'boolean': bool,
                 'float': (int, float), 'list': list}
        oftype = args[0]

        if oftype.startswith('list'):
            ofsubtype = args[0][args[0].find('[') + 1:args[0].find(']')]
            oftype = 'list'

        else:
            ofsubtype = None

        if not isinstance(value, types.get(oftype)):
            raise RuleError('%s should be of type %s' % (field, oftype))
        else:
            if ofsubtype:
                if not all(isinstance(i, types.get(ofsubtype)) for i in value):
                    raise RuleError('%s list items should be of type %s'
                                    % (field, ofsubtype))

            return True

    @staticmethod
    def _rule_contains(field, value, *args) -> bool:
        if args[0] not in value:
            raise RuleError('%s must contain %s' % (field, args[0]))
        else:
            return True

    @staticmethod
    def _rule_allowed(field, value, *args) -> bool:
        readable_allowed = " or ".join([", ".join(args[0][:-1]), args[0][-1]]
                                       if len(args[0]) > 1 else args[0])

        if isinstance(value, list):
            if not set(value).issubset(set(args[0])) or len(set(value)) != len(
                    value):
                raise RuleError('%s must be %s, not %s'
                                % (field, readable_allowed, value))
            else:
                return True

        if not any(fnmatch.filter([value], i) for i in args[0]):
            raise RuleError('%s must be %s, not %s'
                            % (field, readable_allowed, value))
        else:
            return True

    @staticmethod
    def _rule_min(field, value, *args) -> bool:
        if isinstance(value, list):
            if any(i < args[0] for i in value):
                raise RuleError('minimum value of %s is %s' % (field, args[0]))

        elif value < args[0]:
            raise RuleError('minimum value of %s is %s' % (field, args[0]))

        else:
            return True

    @staticmethod
    def _rule_max(field, value, *args) -> bool:
        if isinstance(value, list):
            if any(i > args[0] for i in value):
                raise RuleError('maximum value of %s is %s' % (field, args[0]))

        elif value > args[0]:
            raise RuleError('maximum value of %s is %s' % (field, args[0]))

        else:
            return True

    @staticmethod
    def _rule_minlength(field, value, *args) -> bool:
        if len(value) < args[0]:
            raise RuleError('minimum length of %s is %s' % (field, args[0]))
        else:
            return True

    @staticmethod
    def _rule_maxlength(field, value, *args) -> bool:
        if len(value) > args[0]:
            raise RuleError('maximum length of %s is %s' % (field, args[0]))
        else:
            return True

    @staticmethod
    def _rule_custom_bandpass(field, value) -> bool:
        high_pass, low_pass = value
        if high_pass >= low_pass:
            raise RuleError(
                'high-pass mus be smaller than low-pass in %s' % field)
        else:
            return True

    @staticmethod
    def _rule_custom_voxdim(field, value) -> bool:
        if len(value) == 2:
            raise RuleError('only 1 or 3 values allowed in %s, not 2' % field)
        else:
            return True

    @staticmethod
    def _rule_custom_tr(field, value) -> bool:
        if value >= 100:
            logging.warning(
                '%s is large. Are you sure it is repetition time in seconds?'
                % field)
        return True
