from .exceptions import DocumentError, RuleError, DependencyError, SetDefault
from .utils import pyyaml_ordereddict, config_get
from collections import OrderedDict
from functools import partial as fp, reduce
import operator
import logging
import fnmatch
import string
import yaml


class MetaData(object):
    """Meta data storage for a configuration parameter/field"""
    priority = ('required', 'type', 'contains', 'allowed', 'min',
                'max', 'minlength', 'maxlength')

    def __init__(self, field: str, schema: dict):
        self.field = field
        self.value = None
        self.required = schema.get('required', False)
        self.type = schema.get('type', None)
        self.allowed = schema.get('allowed', None)
        self.contains = schema.get('contains', None)
        self.default = schema.get('default', None)
        self.custom = schema.get('custom', None)
        self.description = schema.get('desc', None)
        self.min = schema.get('min', None)
        self.max = schema.get('max', None)
        self.minlength = schema.get('minlength', None)
        self.maxlength = schema.get('maxlength', None)
        self.dependency = schema.get('dependency', None)

        if self.dependency is not None:
            self.dependency = {k: v for d in self.dependency
                               for k, v in d.items()}

        # Rule validation order
        self.rules = list(schema.keys())

        if 'required' not in self.rules:
            self.rules.append('required')

        priority = list(self.priority) + list(
            set(self.rules) - set(self.priority))
        self.rules.sort(key=lambda x: priority.index(x))


class Validator(object):
    """Validation for configuration files using a schema"""
    priority = ('required', 'type', 'contains', 'allowed', 'min',
                'max', 'minlength', 'maxlength')

    def __init__(self, schema: str):
        yaml.add_representer(OrderedDict, pyyaml_ordereddict)
        self.schema_path = schema
        self.schema = None
        self.document = None
        self.errors = 0
        self._document = None  # temporary (non-validated) input data

    def get(self, keymap, default=None):
        return config_get(keymap, self._document, default)

    @staticmethod
    def load(path) -> dict:
        """Load a YML document"""
        with open(path, 'r', encoding='utf-8') as stream:
            try:
                return OrderedDict(yaml.safe_load(stream))
            except yaml.YAMLError as exc:
                raise DocumentError(exc)

    def depth(self, d, level=0):
        """Traverse through a dictionary"""
        if not isinstance(d, dict) or not d:
            return level

        return max(self.depth(d[k], level + 1) for k in d.keys())

    def _rule_parser(self, this):
        """Return a list containing all rule functions to be executed"""
        partials = []
        for k in this.rules:
            if hasattr(self, '_rule_%s' % k):
                rule = getattr(self, '_rule_%s' % k)
                rule = fp(rule, this)
                partials.append(rule)

        if this.custom is not None:
            for k in this.custom:
                if hasattr(self, '_rule_custom_%s' % k):
                    rule = getattr(self, '_rule_custom_%s' % k)
                    rule = fp(rule, this)
                    partials.append(rule)

        return partials

    def _validate_against(self, document, schema, path: list = []):
        """Execute validation rules per field"""
        for k, v in schema.items():
            path.append(k)
            field = '.'.join(path)
            this = MetaData(field, schema[k])
            this.rules = self._rule_parser(this)

            if isinstance(v, dict) and self.depth(v) == 1 and not isinstance(
                    document, dict):
                self.errors += 1
                logging.error(
                    '%s must be %s, not %s'
                    % (type(document).__name__, '.'.join(path[:-1]), field)
                )
                break

            elif isinstance(v, dict) and self.depth(v) == 1:
                this.value = document.get(k, None) \
                    if isinstance(document, dict) else None

                # Remove escape characters added by yaml.safe_load
                if isinstance(this.value, str):
                    document[k] = this.value.encode('utf-8').decode(
                        'unicode_escape')

                for rule in this.rules:
                    try:
                        rule()
                    except RuleError as exc:
                        self.errors += 1
                        logging.error(exc)
                        break
                    except SetDefault as exc:
                        default = schema[k].get('default', None)
                        if this.value is None and default is not None:
                            logging.warning('using default value for %s (%s)'
                                            % (field, default))
                            document[k] = default
                        break
                    except DependencyError as exc:
                        # Dependencies are not met, parameter will not be used
                        if this.value is not None:
                            logging.warning(
                                '%s is defined but will not be used' % field)

                        if k in document.keys():
                            del document[k]

                        break

            elif isinstance(v, dict) and self.depth(v) > 1:
                if not isinstance(document, dict) and document is None:
                    document = {}

                elif not isinstance(document, dict):
                    self.errors += 1
                    logging.error(
                        '%s must be dict, not %s'
                        % ('.'.join(path[:-1]), type(document).__name__))
                    del path[-1]
                    break

                document[k] = self._validate_against(
                    document.get(k, {}), v, path)

            del path[-1]

        return document

    def _del_invalid(self, document: dict, schema: dict, path: list = []):
        """Delete invalid fields"""
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
        """Validate YML structure and then against a validation schema"""
        if isinstance(document, str):
            document = self.load(document)

        self._document = document
        modality = document.get('modality', None)
        schema = self.load(self.schema_path)

        if modality not in schema['modality']['allowed']:
            self.errors += 1
            logging.error('modality must be %s, not %s'
                          % (', '.join(schema['modality']['allowed']),
                             modality))
            return False

        # Validate and set defaults
        document = self._validate_against(document, schema)

        # Remove invalid keys
        document = self._del_invalid(document, schema)
        document = self._strip_document(document)

        if self.errors == 0:
            self.document = document
            return True

        else:
            return False

    def _set_defaults(self, data):
        """Return a document with all default values"""
        data_copy = data.copy()

        for k, v in data.items():
            if isinstance(v, dict) and self.depth(v) <= 1:
                default = v.get('default', None)
                data_copy[k] = default

            elif isinstance(v, dict):
                data_copy[k] = self._set_defaults(v)

        return data_copy

    def _strip_document(self, data):
        """Strip a document of empty fields"""
        data_copy = OrderedDict()

        for k, v in data.items():
            if isinstance(v, dict):
                v = self._strip_document(v)

            if v not in (u'', None, {}):
                data_copy[k] = v

        return data_copy

    def example(self, modality: str, out: str = None):
        """Create an example configuration file for the requested modality"""
        schema = self.load(self.schema_path)

        if modality not in schema['modality']['allowed']:
            return False

        document = self._set_defaults(schema)
        document['modality'] = modality
        self.validate(document)
        document = self._strip_document(document)

        if out:
            with open(out, 'w') as f:
                yaml.dump(document, f, default_flow_style=False)

    def _rule_required(self, this) -> bool:
        """Check if entry will be used and if so, is required"""
        if this.dependency is not None:
            for k, v in this.dependency.items():
                mapping = k.split('.')
                try:
                    dep_value = reduce(operator.getitem, mapping,
                                       self._document)
                except (KeyError, TypeError) as exc:
                    # Dependency is not provided, try default value
                    mapping = mapping.append('default')

                    try:
                        dep_value = reduce(operator.getitem, mapping,
                                           self.schema)
                    except (KeyError, TypeError) as exc:
                        # No default value, assuming dependency not met
                        raise DependencyError()

                if (isinstance(v, str) and dep_value != v) or \
                        (isinstance(v, list) and dep_value not in v) or \
                        (isinstance(v, bool) and dep_value != v):

                    raise DependencyError()

        if this.required and this.value is None:
            raise RuleError('%s is a required field' % this.field)
        elif not this.required and this.value is None:
            raise SetDefault()
        else:
            return True

    @staticmethod
    def _rule_type(this) -> bool:
        """Check if the field type is correct"""
        types = {'integer': int, 'string': str, 'boolean': bool,
                 'float': (int, float), 'list': list}
        oftype = this.type

        if oftype.startswith('list'):
            ofsubtype = this.type[this.type.find('[') + 1:this.type.find(']')]
            oftype = 'list'

        else:
            ofsubtype = None

        if oftype == 'float' and isinstance(this.value, str):
            # Catch scientific notation
            try:
                this.value = float(this.value)
            except ValueError:
                pass

        if not isinstance(this.value, types.get(oftype)):
            raise RuleError('%s should be of type %s' % (this.field, oftype))
        else:
            if ofsubtype:
                if not all(isinstance(i, types.get(ofsubtype))
                           for i in this.value):
                    raise RuleError('%s list items should be of type %s'
                                    % (this.field, ofsubtype))

            return True

    @staticmethod
    def _rule_contains(this) -> bool:
        """Check if the field contains a must-have value"""
        if this.contains not in this.value:
            raise RuleError('%s must contain %s' % (this.field, this.contains))
        else:
            return True

    @staticmethod
    def _rule_allowed(this) -> bool:
        """Check if the field consists only out of allowed values"""
        readable_allowed = " or ".join(
            [", ".join(repr(this.allowed[:-1])),
             repr(this.allowed[-1])]
            if len(this.allowed) > 1 else repr(this.allowed)
        )

        if None in this.allowed:
            if this.value is None:
                return True

            this.allowed.remove(None)

        expansion = True if any(fnmatch.filter(this.allowed, '*')) else False

        if isinstance(this.value, list) and not expansion:
            if not set(this.value).issubset(set(this.allowed)) or len(
                    set(this.value)) != len(this.value):
                raise RuleError('%s must be %s, not %s'
                                % (this.field, readable_allowed, this.value))
            else:
                return True

        if not isinstance(this.value, list):
            filter_value = [this.value]
        else:
            filter_value = this.value

        if not any(fnmatch.filter(filter_value, i) for i in this.allowed):
            raise RuleError('%s must be %s, not %s'
                            % (this.field, readable_allowed, this.value))
        else:
            return True

    @staticmethod
    def _rule_min(this) -> bool:
        """Check if the field is above the minimum allowed value"""
        if isinstance(this.value, list):
            if any(i < this.min for i in this.value):
                raise RuleError('minimum value of %s is %s'
                                % (this.field, this.min))

        elif this.value < this.min:
            raise RuleError('minimum value of %s is %s'
                            % (this.field, this.min))

        else:
            return True

    @staticmethod
    def _rule_max(this) -> bool:
        """Check if the field is below the maximum allowed value"""
        if isinstance(this.value, list):
            if any(i > this.max for i in this.value):
                raise RuleError('maximum value of %s is %s'
                                % (this.field, this.max))

        elif this.value > this.max:
            raise RuleError('maximum value of %s is %s'
                            % (this.field, this.max))

        else:
            return True

    @staticmethod
    def _rule_minlength(this) -> bool:
        """Check if the field is above the minimum length"""
        if len(this.value) < this.minlength:
            raise RuleError('minimum length of %s is %s'
                            % (this.field, this.minlength))
        else:
            return True

    @staticmethod
    def _rule_maxlength(this) -> bool:
        """Check if the field is below the maximum length"""
        if len(this.value) > this.maxlength:
            raise RuleError('maximum length of %s is %s'
                            % (this.field, this.maxlength))
        else:
            return True

    @staticmethod
    def _rule_custom_bandpass(this) -> bool:
        """Custom rule for the bandpass field"""
        high_pass, low_pass = this.value
        if high_pass >= low_pass:
            raise RuleError(
                'high-pass mus be smaller than low-pass in %s' % this.field)
        else:
            return True

    @staticmethod
    def _rule_custom_voxdim(this) -> bool:
        """Custom rule for the voxel dimensions field"""
        if len(this.value) == 2:
            raise RuleError('only 1 or 3 values allowed in %s, not 2'
                            % this.field)
        else:
            return True

    @staticmethod
    def _rule_custom_tr(this) -> bool:
        """Custom rule for the repetition-time field"""
        if this.value >= 100:
            logging.warning(
                '%s is large. Are you sure it is repetition time in seconds?'
                % this.field)
        return True

    def _rule_custom_agglomerative_linkage(self, this) -> bool:
        """Custom rule for the linkage field for agglomerative clustering"""
        if this.value == 'ward':
            distance_metric = self._document.get('parameters', {})\
                .get('clustering', {})\
                .get('cluster_options', {})\
                .get('distance_metric', 'euclidean')

            if distance_metric != 'euclidean':
                raise RuleError(
                    'parameters.clustering.cluster_options.distance_metric '
                    'must be "euclidean" if the %s is set to "ward"'
                    % this.field)

        return True

    def _rule_custom_has_sessions(self, this) -> bool:
        """Custom rule for the sessions field"""
        sessions = self._document.get('data', {}).get('session', [])
        wildcards = [
            wildcard[1] for wildcard in string.Formatter().parse(this.value)
            if wildcard[1] is not None
        ]
        has_wildcard = True if 'session' in wildcards else False

        if sessions and not has_wildcard:
            # sessions is defined, but wildcard is not provided
            raise RuleError('%s must contain {session}' % this.field)

        if not sessions and has_wildcard:
            # sessions is not defined, but wildcard is provided
            raise RuleError('%s contains {session} but data.session is not '
                            'defined' % this.field)

        return True

    def _rule_custom_space_match(self, this) -> bool:
        """Custom rule for the mask space field"""
        if this.value == 'native':
            seed_keymap = 'data.masks.seed'
            target_keymap = 'data.masks.target'
            seed_mask = self.get(seed_keymap, None)
            target_mask = self.get(target_keymap, None)
            sessions = self.get('data.session', None)

            if target_mask is None:
                raise RuleError('%s must be defined when native space is '
                                'used.' % target_keymap)

            masks = ((seed_keymap, seed_mask), (target_keymap, target_mask))
            for name, mask in masks:
                if mask is not None:
                    if '{participant_id}' not in mask:
                        raise RuleError(
                            '%s must contain {participant_id} when native '
                            'space is used' % name
                        )

                    if sessions is not None and '{session}' in mask:
                        raise RuleError(
                            '%s cannot contain {session} because session data '
                            'will be merged' % name
                        )

        return True

    @staticmethod
    def _rule_custom_benchmarking(this):
        """Custom rule for benchmarking"""
        if this.value is True:
            try:
                import psutil
            except ImportError as exc:
                raise RuleError('Python 3 package psutil needs to be '
                                'installed for benchmarking')

        return True

    def _rule_custom_spectral_kernel(self, this):
        """Custom rule for kernel selection for spectral clustering"""
        modality = self.get('modality')

        if this.value == 'precomputed' and modality != 'connectivity':
            raise RuleError('%s can only be \'precomputed\' if connectivity is'
                            'set as the modality and an adjacency matrix is'
                            'given instead of a connectivity matrix'
                            % this.field)

        return True

    def _rule_custom_references(self, this):
        """Custom rule to check if median filtering is false when using
        reference images"""
        mf_keymap = 'parameters.masking.seed.median_filtering.apply'
        median_filter = self.get(mf_keymap, False)

        if median_filter:
            raise RuleError('%s cannot be used when %s is set to True, '
                            'because this option will make the seed mask '
                            'different from the reference images'
                            % (this.field, mf_keymap))

        return True

    def _rule_custom_has_xfm(self, this):
        """Custom rule to check if an XFM is defined when using the dMRI
        modality"""
        xfm = self.get('data.xfm', None)

        if xfm is not None and this.value is None:
            raise RuleError('%s must be defined if data.xfm is defined'
                            % this.field)

        return True
