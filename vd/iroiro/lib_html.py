import re
import pathlib

from html.parser import HTMLParser

from .internal_utils import exporter
export, __all__ = exporter()

from . import lib_fs
from .lib_math import interval


self_closing_tags = {
        'area', 'base', 'br', 'col', 'embed', 'hr', 'img',
        'input', 'link', 'meta', 'param', 'source', 'track', 'wbr',
        }


@export
class HTML(HTMLParser):
    def __init__(self, source=None, *, keep_comments=False, pre='pre'):
        super().__init__()

        if isinstance(pre, str):
            pre = {pre}
        else:
            try:
                pre = set(pre)
            except TypeError:
                pre = bool(pre)

        self.keep_comments = keep_comments
        self.pre = pre

        self.decl = None
        self.roots = []
        self.stack = []

        if isinstance(source, pathlib.Path):
            with lib_fs.open(source) as f:
                self.feed(f.read())
                self.close()
        elif hasattr(source, 'read') and callable(source.read):
            self.feed(source.read())
            self.close()
        elif isinstance(source, str):
            self.feed(source)
            self.close()
        elif source is None:
            pass
        else:
            raise TypeError('Unrecognized source:', repr(source))

    @property
    def root(self):
        if self.roots:
            return self.roots[0]

    def __getattr__(self, name):
        for root in self.roots:
            if name == root.name:
                return root

        for root in self.roots:
            try:
                return getattr(root, name)
            except AttributeError:
                pass

        raise AttributeError(name)

    def handle_decl(self, decl):
        self.decl = decl

    def handle_comment(self, data):
        if not self.keep_comments:
            return

        elem = HTMLComment(data)

        if not self.stack:
            self.roots.append(elem)

        if self.stack:
            self.stack[-1].append(elem)

    def handle_starttag(self, tag, attrs):
        elem = HTMLElement(tag, attrs)

        if not self.stack:
            self.roots.append(elem)

        if self.stack:
            self.stack[-1].append(elem)

        if tag in self_closing_tags:
            elem.closed = True
        else:
            self.stack.append(elem)

    def handle_endtag(self, tag):
        if not self.stack:
            return

        if tag in self_closing_tags:
            return

        if self.stack[-1].name == tag:
            self.stack[-1].closed = True
            self.stack.pop()
            return

        for i in interval(len(self.stack) - 1, 0, close=True):
            if self.stack[i].name == tag:
                self.stack[i:] = []
                break

    def handle_data(self, data):
        pre = None
        if isinstance(self.pre, set):
            pre = self.pre & set(node.name for node in self.stack)
        else:
            pre = bool(self.pre)

        if not pre:
            data = re.sub(r' +', r' ', data.replace('\n', ' '))
            data = re.sub(r'(^ +| +$)', '', data)

        if not data:
            return

        if self.stack:
            self.stack[-1].append(data)


class HTMLComment:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return '<!--' + self.data + '-->'

    def __str__(self):
        return self.data

    def __eq__(self, other):
        return self is other or str(self) == other


class HTMLElementDataSetProxy:
    def __init__(self, elem):
        self.elem = elem

    def __getattr__(self, name):
        return self.elem.attrs.get('data-' + name)


class HTMLElement:
    def __init__(self, name, attrs):
        self.name = name
        self.attrs = dict(attrs)
        self.childnodes = []
        self.closed = False

    def __repr__(self):
        if self.attrs:
            attr = ' ' + ' '.join([f'{attr}="{value}"'
                                   for attr, value in self.attrs.items()])
        else:
            attr = ''
        if self.name in self_closing_tags:
            return f'<{self.name}{attr}>'

        return (f'<{self.name}{attr}>' +
                ''.join(child if isinstance(child, str) else repr(child) for child in self.childnodes) +
                (f'</{self.name}>' if self.closed else ''))

    @property
    def tagname(self):
        return self.name

    @property
    def classlist(self):
        return self.attrs.get('class', '').split()

    @property
    def dataset(self):
        return HTMLElementDataSetProxy(self)

    @property
    def children(self):
        return [child
                for child in self.childnodes
                if not isinstance(child, str)
                ]

    @property
    def innerText(self):
        tokens = [child if isinstance(child, str) else child.innerText
                  for child in self.childnodes]

        if not tokens:
            return ''

        # Join child innerTexts with
        # - empty, if one of the connecting ends already have space
        # - space, otherwise
        ret, *tokens = tokens
        for token in tokens:
            if (not ret or ret.endswith((' ', '\n'))) or (not token or token.startswith((' ', '\n'))):
                ret += token
            else:
                ret += ' ' + token
        return ret

    def __getattr__(self, name):
        if name in self.attrs:
            return self.attrs[name]

        for child in self.children:
            if child.name == name:
                return child

        raise AttributeError(name)

    def append(self, elem):
        self.childnodes.append(elem)
