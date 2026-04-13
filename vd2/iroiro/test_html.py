from .lib_test_utils import *

from .lib_html import HTML


class TestHTMLInputSource(TestCase):
    def test_read_from_empty_string(self):
        document = HTML('')
        self.eq(document.root, None)

    def test_read_from_string(self):
        document = HTML('<html></html>')
        self.ne(document.root, None)
        self.eq(document.root.name, 'html')

    def test_read_from_file_obj(self):
        import io
        fake_file = io.StringIO('<html><head></head><body><div id="container"></div></body></html>')
        document = HTML(fake_file)
        self.eq(document.html.body.div.id, 'container')

    def test_read_from_file_path(self):
        import pathlib
        fake_path = pathlib.Path('some/path')

        def mock_open(path):
            self.eq(path, fake_path)
            import io
            return io.StringIO('<html><head></head><body><div id="container"></div></body></html>')
        self.patch('iroiro.fs.open', mock_open)

        document = HTML(fake_path)
        self.eq(document.html.body.div.id, 'container')

    def test_no_source(self):
        document = HTML()

    def test_invalid_input(self):
        with self.raises(TypeError):
            document = HTML(42)


class TestHTMLContent(TestCase):
    def test_only_doctype(self):
        document = HTML('<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" '
                        '"http://www.w3.org/TR/html4/strict.dtd">')
        self.eq(document.decl, 'DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01//EN" "http://www.w3.org/TR/html4/strict.dtd"')

    def test_only_html_tag(self):
        document = HTML('<html></html>')
        self.ne(document.root, None)
        self.eq(document.root.name, 'html')

    def test_ignore_naked_strings(self):
        document = HTML('data')
        self.eq(document.root, None)

    def test_minimal_html_doc(self):
        document = HTML('<!DOCTYPE html><title>a</title>')
        self.eq(document.decl, 'DOCTYPE html')
        self.eq(document.root.name, 'title')
        self.eq(document.root.innerText, 'a')

    def test_small_html_doc(self):
        document = HTML('''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <link href="style.css" rel="stylesheet" />
        <script type="text/javascript" src="main.js"></script>
        <title>Title</title>
    </head>
    <body onload="main()">
        <div id="container">text</div>
    </body>
</html>
''')
        self.eq(document.decl, 'DOCTYPE html')
        self.eq(document.root.name, 'html')
        self.eq(document.html.innerText.split(), ['Title', 'text'])

        self.eq([x.name for x in document.head.children],
                ['meta', 'link', 'script', 'title'])
        self.eq(document.head.meta.charset, 'utf-8')
        self.eq(document.head.link.href, 'style.css')
        self.eq(document.head.link.rel, 'stylesheet')
        self.eq(document.head.script.type, 'text/javascript')
        self.eq(document.head.script.src, 'main.js')
        self.eq(document.head.title.innerText, 'Title')

        self.eq(document.body.onload, 'main()')
        self.eq(document.body.innerText, 'text')
        self.eq(document.body.div.id, 'container')
        self.eq(document.body.div.innerText, 'text')

    def test_multiple_roots(self):
        document = HTML('<head><title>title</title></head><body><div id="container"></div></body>')
        self.eq(document.head.title.innerText, 'title')
        self.eq(document.body.div.id, 'container')

        document = HTML('<div id="first"></div>'
                        '<div id="second"></div>')
        self.eq(document.roots[0].id, 'first')
        self.eq(document.roots[1].id, 'second')

        with self.raises(AttributeError):
            document.table

    def test_only_closing_tag(self):
        document = HTML('</table>')
        self.eq(document.roots, [])

    def test_redundent_closing_tag(self):
        document = HTML('<table><tr><td>td</td></td></tr><tr><td>td2</td></tr></tr></table>')
        self.eq(document.table.tr.td.innerText, 'td')
        self.eq(document.table.children[1].td.innerText, 'td2')

    def test_redundent_closing_parent_tag(self):
        document = HTML('<table><tr><td>td</tr><tr><td>td2</td></tr></tr></table>')
        self.eq(document.table.tr.td.innerText, 'td')
        self.eq(document.table.children[1].td.innerText, 'td2')

    def test_redundent_closing_whole_root(self):
        document = HTML('<table><tr><td>td</table>')
        self.eq(document.table.tr.td.innerText, 'td')

    def test_comment(self):
        doc = '''
<!DOCTYPE html>
<html>
    <head>
        <!--comment1-->
    </head>
    <body>
        <!-- comment2 -->
    </body>
</html>'''
        document = HTML(doc, keep_comments=False)
        self.eq(document.head.childnodes, [])
        self.eq(document.body.childnodes, [])

        document = HTML(doc, keep_comments=True)
        self.eq(document.head.childnodes[0], 'comment1')
        self.eq(document.body.childnodes[0], ' comment2 ')

        document = HTML('<!-- comment? -->', keep_comments=True)
        self.eq(document.root, ' comment? ')
        self.eq(repr(document.root), '<!-- comment? -->')

    def test_pre(self):
        doc = '''
<!DOCTYPE html>
<html>
    <body>  <pre>
        <div>text  </div><span> iroiro </span>  </pre>  </body>
</html>'''
        document = HTML(doc, pre=False)
        self.eq(document.body.innerText, 'text iroiro')

        document = HTML(doc, pre=True)
        self.eq(document.body.innerText, '  \n        text   iroiro     ')

        document = HTML(doc, pre='pre')
        self.eq(document.body.innerText, '\n        text   iroiro   ')

        document = HTML(doc, pre='div')
        self.eq(document.body.innerText, 'text  iroiro')

        document = HTML(doc, pre={'div', 'span'})
        self.eq(document.body.innerText, 'text   iroiro ')

    def test_feed_on_demand(self):
        document = HTML()
        document.feed('<html>')
        self.eq(repr(document.html), '<html>')
        document.feed('<head></head>')
        document.feed('<body>')
        self.eq(repr(document.html), '<html><head></head><body>')
        document.feed('<div id="container"></div>')
        self.eq(repr(document.html), '<html><head></head><body><div id="container"></div>')
        self.eq(document.html.body.div.id, 'container')
        self.eq(document.html.body.closed, False)
        document.feed('</body>')
        document.feed('</html>')
        document.close()
        self.eq(repr(document.html), '<html><head></head><body><div id="container"></div></body></html>')
        self.eq(document.html.body.closed, True)


class TestHTMLElementAttributes(TestCase):
    def test_classlist(self):
        document = HTML('<div class="container centered hidden"></div>')
        self.eq(document.div.classlist, ['container', 'centered', 'hidden'])

    def test_tagname(self):
        document = HTML('<div class="container centered hidden"></div>')
        self.eq(document.div.tagname, 'div')

    def test_repr(self):
        document = HTML('<div class="container centered hidden">text</div>')
        self.eq(repr(document.div), '<div class="container centered hidden">text</div>')

        document = HTML('<br>')
        self.eq(repr(document.br), "<br>")

    def test_dataset(self):
        document = HTML('<div data-what="iroiro">text</div>')
        self.eq(document.div.dataset.what, 'iroiro')
        self.eq(document.div.dataset.what2, None)
