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

*   `vdir` treats symbolic links as files instead of resolving them

*   `vdir` supports editing symbolic links


Examples (shell)
-------------------------------------------------------------------------------
Edit the current diretory:
```console
sh$ vdir
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
# - Add a path to track it.
# - Sort the paths as you want.
# - Add a '#' before the id to untrack an item.
# - Add a '+' before the id to expand non-hidden items under the directory.
# - Add a '*' before the id to expand all items under the directory.
# - Add a '@' before the id to resolve the soft link.
# - Setup default vdir.vimrc with
#   $ vdir --vimrc
# ═════════════════════════════════════════════════════════════════════════════

110 ││ LICENSE
120 ││ README.md
131 ││ __pycache__/
140 ││ pyproject.toml
151 ││ vdir/
```

Edit the content, save and quit, and `vdir` prompts you the changes like this:

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
