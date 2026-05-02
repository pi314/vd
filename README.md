VDIR - Edit directory with vim
===============================================================================

Heavily inspired by `vidir` from [moreutils](https://joeyh.name/code/moreutils/) and
[edir](https://github.com/bulletmark/edir).

With a few different design choices:

*   `vdir` always prompts before applying changes

*   `vdir` doesn't integrate with `git` (for not accidentally messing up your staging area)

*   `vdir` displays an (arguably) pretty-looking preview of the changes

*   `vdir` pads sequence number to same width for visual block operations

*   `vdir` references `LS_COLORS` for coloring

*   `vdir` supports stage/unstage items dynamically

*   `vdir` supports stage/unstage items dynamically from shell commands' output

    -   Shell commands could be piped

*   `vdir` supports sorting staged items by several attributes, each of them could be reversed

*   `vdir` treats symbolic links as files instead of resolving them

*   `vdir` supports editing symbolic links


Examples (shell)
-------------------------------------------------------------------------------
Check version
```console
sh$ vdir --version
vdir 0.4.1
```

Manage current diretory:
```console
sh$ vdir
```

Manage everything including hidden files in current diretory:
```console
sh$ vdir -a
```

Manage certain files:
```console
sh$ vdir *.txt
```

Manage found files:
```console
sh$ fd --type f | vdir
```


Examples (vim)
-------------------------------------------------------------------------------
A `vdir`'s vim session looks like this:

```console
# ═════════════════════════════════════════════════════════════════════════════
# - Add paths to stage them. Globs are recognized.
# - Add a '#' before id to untrack an item.
# - Add a '+' before id to expand non-hidden items under the directory.
# - Add a '*' before id to expand all items under the directory.
# - Add a '@' before id to resolve the soft link.
# - Stage items by shell command output: (globs are not supported here)
#   $ find . -type f | grep py
# - Sort with:
#   :sort [-][type|isdir|isfile|isfifo|islink|path|basename|name|dirname|size|atime|mtime|ctime|birthtime] ...
# - Setup default vd.vimrc with:
#   $ vdir --vimrc
# ═════════════════════════════════════════════════════════════════════════════

110 ││ LICENSE
120 ││ README.md
131 ││ __pycache__/
140 ││ pyproject.toml
151 ││ vdir/
```

Paths could be added directly for next round editing.

Shell commands could be used to add paths in batch.
Shell commands could be piped, but note that commands are ran one-by-one,
i.e. each command's stdout is collected, returncode is checked,
and then all pipe to stdin of the next command.

If one command fails (i.e. returncode != 0), the pipeline stops, and stderr is appended in comment.

The inventory could be sorted with `:sort` command.
Several attributes are available, each of them could be prefixed with `-` for reversing the order.
For example, `:sort dirname -type basename` sorts the inventory with `dirname` ascending,
`type` decending, and `basename` ascending.

When you're done, save and quit, and `vdir` prompts you the changes like this:

```
[info] Delete:[LICENSE]
[info] Rename:[README.mdd]
Continue? [(Y)es / (n)o / (e)dit / (r)edo] _
```

*   `y` to apply and continue with another `vim` session
*   Empty input to apply and quit
*   `n` to cancel the edit
*   `ctrl`+`c` to cancel the edit if you're scared of the change
*   `e` to continue editing
*   `r` to restart with initial contents


Installation
-------------------------------------------------------------------------------

[![vdir](https://img.shields.io/pypi/v/vdir)](https://pypi.org/project/vdir/)


```console
sh$ pipx install vdir
```
